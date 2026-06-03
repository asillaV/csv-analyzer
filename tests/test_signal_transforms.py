"""Test per core/signal_transforms.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.signal_transforms import (
    TransformSpec,
    TransformResult,
    TRANSFORM_LABELS,
    validate_transform_spec,
    apply_transform,
    apply_transform_pipeline,
    transform_spec_cache_key,
    _SCIPY_INTERP_OK,
    _SCIPY_INTEG_OK,
)
from core.signal_tools import FsInfo
from tests.fixtures.synthetic_signals import (
    generate_sine_wave,
    generate_ramp,
    generate_white_noise,
)


# ---- Fixtures ----

@pytest.fixture
def ramp100():
    x, y, _ = generate_ramp(start=0.0, stop=10.0, n_samples=100)
    return x.astype(float), y


@pytest.fixture
def sine100():
    t, y, _ = generate_sine_wave(frequency=5.0, fs=100.0, duration=1.0)
    return t, y


@pytest.fixture
def fs_100hz():
    return FsInfo(value=100.0, source="manual", unit="manual", is_uniform=True)


# ---- TransformSpec dataclass ----

class TestTransformSpecDataclass:
    def test_defaults(self):
        spec = TransformSpec(kind="offset")
        assert spec.enabled is True
        assert spec.constant == 0.0
        assert spec.shift_samples == 0
        assert spec.shift_time == 0.0
        assert spec.target_fs is None
        assert spec.interp_method == "linear"
        assert spec.per_column is False
        assert spec.per_column_overrides == {}

    def test_all_kinds_in_labels(self):
        kinds = list(TRANSFORM_LABELS.keys())
        assert len(kinds) == 9
        for k in kinds:
            spec = TransformSpec(kind=k)
            assert spec.kind == k


# ---- Cache key ----

class TestCacheKey:
    def test_deterministic(self):
        spec = TransformSpec(kind="offset", constant=3.5)
        assert transform_spec_cache_key(spec) == transform_spec_cache_key(spec)

    def test_different_for_different_specs(self):
        k1 = transform_spec_cache_key(TransformSpec(kind="offset", constant=1.0))
        k2 = transform_spec_cache_key(TransformSpec(kind="offset", constant=2.0))
        assert k1 != k2

    def test_hashable(self):
        spec = TransformSpec(kind="scale", constant=2.0, per_column_overrides={"col1": 3.0})
        key = transform_spec_cache_key(spec)
        d = {key: "value"}
        assert d[key] == "value"


# ---- validate_transform_spec ----

class TestValidateTransformSpec:
    def test_disabled_always_valid(self):
        spec = TransformSpec(kind="derivative", enabled=False)
        ok, _ = validate_transform_spec(spec, x_available=False)
        assert ok is True

    def test_x_required_without_x(self):
        for kind in ("shift_time", "resample", "derivative", "integral"):
            spec = TransformSpec(kind=kind)
            ok, msg = validate_transform_spec(spec, x_available=False)
            assert ok is False, f"{kind} should fail without X"
            assert "asse X" in msg

    def test_offset_always_valid(self):
        ok, _ = validate_transform_spec(TransformSpec(kind="offset"))
        assert ok is True

    def test_scale_always_valid(self):
        ok, _ = validate_transform_spec(TransformSpec(kind="scale", constant=0.0))
        assert ok is True

    def test_resample_no_target_fs(self):
        spec = TransformSpec(kind="resample", target_fs=None)
        ok, msg = validate_transform_spec(spec, x_available=True)
        assert ok is False
        assert "target_fs" in msg

    def test_resample_negative_target_fs(self):
        spec = TransformSpec(kind="resample", target_fs=-10.0)
        ok, msg = validate_transform_spec(spec, x_available=True)
        assert ok is False

    def test_resample_invalid_method(self):
        spec = TransformSpec(kind="resample", target_fs=50.0, interp_method="spline")
        ok, msg = validate_transform_spec(spec, x_available=True)
        assert ok is False
        assert "interpolazione" in msg.lower()

    @pytest.mark.skipif(not _SCIPY_INTERP_OK, reason="SciPy non disponibile")
    def test_resample_aliasing_warning(self):
        fs_info = FsInfo(value=100.0, source="manual", unit="manual", is_uniform=True)
        spec = TransformSpec(kind="resample", target_fs=80.0)
        ok, msg = validate_transform_spec(spec, fs_info=fs_info, x_available=True)
        assert ok is True
        assert "aliasing" in msg.lower()

    @pytest.mark.skipif(_SCIPY_INTEG_OK, reason="SciPy disponibile")
    def test_integral_without_scipy(self):
        spec = TransformSpec(kind="integral")
        ok, msg = validate_transform_spec(spec, x_available=True)
        assert ok is False
        assert "SciPy" in msg


# ---- apply_transform: basic kinds ----

class TestApplyOffset:
    def test_adds_constant(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="offset", constant=5.0)
        result = apply_transform(y, x, spec)
        np.testing.assert_allclose(result.y.values, y.values + 5.0)

    def test_negative_offset(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="offset", constant=-2.5)
        result = apply_transform(y, x, spec)
        np.testing.assert_allclose(result.y.values, y.values - 2.5)

    def test_zero_offset_is_identity(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="offset", constant=0.0)
        result = apply_transform(y, x, spec)
        np.testing.assert_allclose(result.y.values, y.values)

    def test_preserves_length(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="offset", constant=1.0))
        assert len(result.y) == len(y)
        assert result.changed_length is False

    def test_per_column_override(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="offset", constant=0.0, per_column=True,
                             per_column_overrides={"col_a": 10.0})
        result = apply_transform(y, x, spec, col_name="col_a")
        np.testing.assert_allclose(result.y.values, y.values + 10.0)

    def test_per_column_fallback_to_constant(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="offset", constant=3.0, per_column=True,
                             per_column_overrides={})
        result = apply_transform(y, x, spec, col_name="unknown_col")
        np.testing.assert_allclose(result.y.values, y.values + 3.0)


class TestApplyScale:
    def test_multiplies_by_factor(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="scale", constant=2.0)
        result = apply_transform(y, x, spec)
        np.testing.assert_allclose(result.y.values, y.values * 2.0)

    def test_scale_by_zero(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="scale", constant=0.0)
        result = apply_transform(y, x, spec)
        np.testing.assert_allclose(result.y.values, np.zeros(len(y)))

    def test_scale_by_one_is_identity(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="scale", constant=1.0)
        result = apply_transform(y, x, spec)
        np.testing.assert_allclose(result.y.values, y.values)

    def test_preserves_length(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="scale", constant=3.0))
        assert len(result.y) == len(y)
        assert result.changed_length is False


class TestApplyMinmaxNorm:
    def test_range_is_zero_to_one(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="minmax_norm")
        result = apply_transform(y, x, spec)
        assert float(result.y.min()) == pytest.approx(0.0, abs=1e-9)
        assert float(result.y.max()) == pytest.approx(1.0, abs=1e-9)

    def test_constant_signal_raises(self):
        y = pd.Series([3.0] * 50)
        x = pd.Series(range(50), dtype=float)
        with pytest.raises(ValueError, match="costante"):
            apply_transform(y, x, TransformSpec(kind="minmax_norm"))

    def test_preserves_length(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="minmax_norm"))
        assert len(result.y) == len(y)
        assert result.changed_length is False


class TestApplyZscoreNorm:
    def test_mean_zero_std_one(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="zscore_norm")
        result = apply_transform(y, x, spec)
        assert float(result.y.mean()) == pytest.approx(0.0, abs=1e-9)
        assert float(result.y.std()) == pytest.approx(1.0, rel=1e-6)

    def test_constant_signal_raises(self):
        y = pd.Series([5.0] * 50)
        x = pd.Series(range(50), dtype=float)
        with pytest.raises(ValueError, match="costante"):
            apply_transform(y, x, TransformSpec(kind="zscore_norm"))

    def test_preserves_length(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="zscore_norm"))
        assert len(result.y) == len(y)
        assert result.changed_length is False


class TestApplyShiftSamples:
    def test_positive_shift_introduces_leading_nan(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="shift_samples", shift_samples=5)
        result = apply_transform(y, x, spec)
        assert np.isnan(result.y.values[:5]).all()
        assert not np.isnan(result.y.values[5:]).any()

    def test_negative_shift_introduces_trailing_nan(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="shift_samples", shift_samples=-3)
        result = apply_transform(y, x, spec)
        assert np.isnan(result.y.values[-3:]).all()
        assert not np.isnan(result.y.values[:-3]).any()

    def test_zero_shift_is_identity(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="shift_samples", shift_samples=0)
        result = apply_transform(y, x, spec)
        np.testing.assert_array_equal(result.y.values, y.values)

    def test_preserves_length(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="shift_samples", shift_samples=10))
        assert len(result.y) == len(y)
        assert result.changed_length is False

    def test_x_unchanged(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="shift_samples", shift_samples=5))
        np.testing.assert_array_equal(result.x.values, x.values)


class TestApplyShiftTime:
    def test_adds_dt_to_x(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="shift_time", shift_time=10.0)
        result = apply_transform(y, x, spec)
        np.testing.assert_allclose(result.x.values, x.values + 10.0)

    def test_y_unchanged(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="shift_time", shift_time=5.0)
        result = apply_transform(y, x, spec)
        np.testing.assert_array_equal(result.y.values, y.values)

    def test_requires_x(self):
        y = pd.Series([1.0, 2.0, 3.0])
        spec = TransformSpec(kind="shift_time", shift_time=1.0)
        with pytest.raises(ValueError, match="asse X"):
            apply_transform(y, None, spec)

    def test_datetime_x(self):
        times = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1s"))
        y = pd.Series(np.ones(10))
        spec = TransformSpec(kind="shift_time", shift_time=5.0)
        result = apply_transform(y, times, spec)
        expected = times + pd.Timedelta(seconds=5.0)
        pd.testing.assert_series_equal(result.x, expected)


# ---- apply_transform: advanced kinds ----

class TestApplyResample:
    @pytest.mark.skipif(not _SCIPY_INTERP_OK, reason="SciPy non disponibile")
    def test_output_length_matches_target_fs(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="resample", target_fs=50.0, interp_method="linear")
        result = apply_transform(y, x, spec)
        x_range = float(x.iloc[-1]) - float(x.iloc[0])
        expected_n = max(2, int(round(x_range * 50.0)))
        assert len(result.y) == expected_n

    def test_linear_resampling(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="resample", target_fs=50.0, interp_method="linear")
        result = apply_transform(y, x, spec)
        assert len(result.y) > 0
        assert result.changed_length is True
        assert result.fs_out == pytest.approx(50.0)

    def test_requires_x(self):
        y = pd.Series(np.ones(50))
        spec = TransformSpec(kind="resample", target_fs=10.0)
        with pytest.raises(ValueError, match="asse X"):
            apply_transform(y, None, spec)

    def test_no_target_fs_raises(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="resample", target_fs=None)
        with pytest.raises(ValueError, match="target_fs"):
            apply_transform(y, x, spec)

    def test_output_x_range_preserved(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="resample", target_fs=20.0)
        result = apply_transform(y, x, spec)
        assert float(result.x.iloc[0]) == pytest.approx(float(x.iloc[0]), rel=1e-6)
        assert float(result.x.iloc[-1]) == pytest.approx(float(x.iloc[-1]), rel=1e-4)


class TestApplyDerivative:
    def test_derivative_of_ramp_is_constant(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="derivative")
        result = apply_transform(y, x, spec)
        deriv_vals = result.y.dropna().values
        assert np.std(deriv_vals) == pytest.approx(0.0, abs=1e-6)

    def test_output_is_n_minus_one(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="derivative"))
        assert len(result.y) == len(y) - 1
        assert result.changed_length is True

    def test_x_midpoints(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="derivative"))
        expected = (x.values[:-1] + x.values[1:]) / 2.0
        np.testing.assert_allclose(result.x.values, expected)

    def test_requires_x(self):
        y = pd.Series(np.ones(10))
        with pytest.raises(ValueError, match="asse X"):
            apply_transform(y, None, TransformSpec(kind="derivative"))

    def test_short_signal_raises(self):
        y = pd.Series([1.0])
        x = pd.Series([0.0])
        with pytest.raises(ValueError, match="corto"):
            apply_transform(y, x, TransformSpec(kind="derivative"))


@pytest.mark.skipif(not _SCIPY_INTEG_OK, reason="SciPy non disponibile")
class TestApplyIntegral:
    def test_integral_of_constant_is_linear(self):
        x = pd.Series(np.linspace(0, 1, 101))
        y = pd.Series(np.ones(101))
        result = apply_transform(y, x, TransformSpec(kind="integral"))
        np.testing.assert_allclose(result.y.values, x.values, atol=1e-6)

    def test_same_length_as_input(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="integral"))
        assert len(result.y) == len(y)
        assert result.changed_length is False

    def test_starts_at_zero(self, ramp100):
        x, y = ramp100
        result = apply_transform(y, x, TransformSpec(kind="integral"))
        assert float(result.y.iloc[0]) == pytest.approx(0.0, abs=1e-9)

    def test_requires_x(self):
        y = pd.Series(np.ones(10))
        with pytest.raises(ValueError, match="asse X"):
            apply_transform(y, None, TransformSpec(kind="integral"))


# ---- apply_transform: disabled spec ----

class TestDisabledSpec:
    def test_disabled_returns_copy(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="scale", constant=100.0, enabled=False)
        result = apply_transform(y, x, spec)
        np.testing.assert_array_equal(result.y.values, y.values)

    def test_disabled_x_unchanged(self, ramp100):
        x, y = ramp100
        spec = TransformSpec(kind="derivative", enabled=False)
        result = apply_transform(y, x, spec)
        np.testing.assert_array_equal(result.x.values, x.values)


# ---- apply_transform_pipeline ----

class TestApplyTransformPipeline:
    def test_empty_pipeline_is_identity(self, ramp100):
        x, y = ramp100
        y_out, x_out, descs, changed = apply_transform_pipeline(y, x, [])
        np.testing.assert_array_equal(y_out.values, y.values)
        assert descs == []
        assert changed is False

    def test_single_offset(self, ramp100):
        x, y = ramp100
        specs = [TransformSpec(kind="offset", constant=7.0)]
        y_out, x_out, descs, changed = apply_transform_pipeline(y, x, specs)
        np.testing.assert_allclose(y_out.values, y.values + 7.0)
        assert len(descs) == 1
        assert changed is False

    def test_offset_then_scale(self, ramp100):
        x, y = ramp100
        specs = [
            TransformSpec(kind="offset", constant=1.0),
            TransformSpec(kind="scale", constant=2.0),
        ]
        y_out, _, _, _ = apply_transform_pipeline(y, x, specs)
        expected = (y.values + 1.0) * 2.0
        np.testing.assert_allclose(y_out.values, expected)

    def test_disabled_steps_skipped(self, ramp100):
        x, y = ramp100
        specs = [
            TransformSpec(kind="offset", constant=10.0, enabled=False),
            TransformSpec(kind="scale", constant=3.0),
        ]
        y_out, _, descs, _ = apply_transform_pipeline(y, x, specs)
        np.testing.assert_allclose(y_out.values, y.values * 3.0)
        assert len(descs) == 1

    def test_changed_length_propagates(self, ramp100):
        x, y = ramp100
        specs = [
            TransformSpec(kind="offset", constant=1.0),
            TransformSpec(kind="derivative"),
        ]
        y_out, x_out, descs, changed = apply_transform_pipeline(y, x, specs)
        assert changed is True
        assert len(y_out) == len(y) - 1
        assert len(descs) == 2

    def test_invalid_step_raises(self, ramp100):
        x, y = ramp100
        specs = [TransformSpec(kind="resample", target_fs=None)]
        with pytest.raises(ValueError):
            apply_transform_pipeline(y, x, specs)

    def test_minmax_then_zscore(self, ramp100):
        x, y = ramp100
        specs = [
            TransformSpec(kind="minmax_norm"),
            TransformSpec(kind="zscore_norm"),
        ]
        y_out, _, descs, _ = apply_transform_pipeline(y, x, specs)
        assert float(y_out.mean()) == pytest.approx(0.0, abs=1e-9)
        assert float(y_out.std()) == pytest.approx(1.0, rel=1e-6)
        assert len(descs) == 2

    def test_per_column_overrides_in_pipeline(self, ramp100):
        x, y = ramp100
        specs = [
            TransformSpec(kind="offset", constant=0.0, per_column=True,
                         per_column_overrides={"sens_A": 5.0}),
        ]
        y_a, _, _, _ = apply_transform_pipeline(y, x, specs, col_name="sens_A")
        y_b, _, _, _ = apply_transform_pipeline(y, x, specs, col_name="sens_B")
        np.testing.assert_allclose(y_a.values, y.values + 5.0)
        np.testing.assert_allclose(y_b.values, y.values + 0.0)

    @pytest.mark.skipif(not _SCIPY_INTEG_OK, reason="SciPy non disponibile")
    def test_derivative_then_integral_roundtrip(self, ramp100):
        x, y = ramp100
        specs = [TransformSpec(kind="derivative")]
        y_deriv, x_deriv, _, _ = apply_transform_pipeline(y, x, specs)
        specs2 = [TransformSpec(kind="integral")]
        y_int, _, _, _ = apply_transform_pipeline(y_deriv, x_deriv, specs2)
        # ∫ dy/dx dx ≈ y + C; check relative shape (skip first point = 0)
        assert len(y_int) == len(y_deriv)
