"""Test di core.engine: AnalysisSession, ResultCache, Recipe (Parte 1, F5)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core import engine
from core.engine import AnalysisSession, Recipe, RecipeTransform, ResultCache
from core.signal_tools import FFTSpec, FilterSpec, apply_filter, compute_fft, resolve_fs
from core.signal_transforms import TransformSpec, apply_transform_pipeline

pytestmark = pytest.mark.unit

FS = 1000.0


@pytest.fixture
def signal_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t = np.arange(4000) / FS
    clean = np.sin(2 * np.pi * 50 * t)
    return pd.DataFrame({
        "t": t,
        "a": clean + rng.normal(0, 0.3, t.size),
        "b": 2 * np.cos(2 * np.pi * 120 * t),
        "testo": ["x"] * t.size,
    })


def _session(df, **recipe_kwargs) -> AnalysisSession:
    return AnalysisSession(df, recipe=Recipe(**recipe_kwargs))


# ---------------------------------------------------------------------------
# Conversioni
# ---------------------------------------------------------------------------
def test_parse_x_variants():
    assert engine.parse_x(None) is None
    numeric = engine.parse_x(pd.Series(["1", "2", "3", "x"] + ["4"] * 6))
    assert numeric.dtype.kind == "f" and numeric.iloc[0] == 1.0
    dates = engine.parse_x(pd.Series(["2025-01-01 00:00:00", "2025-01-01 00:00:01"]))
    assert pd.api.types.is_datetime64_any_dtype(dates)
    already = engine.parse_x(pd.Series(pd.date_range("2025-01-01", periods=3, freq="s")))
    assert pd.api.types.is_datetime64_any_dtype(already)


def test_numeric_series_keeps_name():
    df = pd.DataFrame({"v": ["1", "x", "3"]})
    y = engine.numeric_series(df, "v")
    assert y.name == "v" and y.isna().tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# Equivalenza con le chiamate dirette (quello che fa l'app)
# ---------------------------------------------------------------------------
def test_pipeline_matches_direct_calls(signal_df):
    specs = [TransformSpec(kind="offset", constant=1.5), TransformSpec(kind="scale", constant=2.0)]
    fspec = FilterSpec(kind="butter_lp", enabled=True, order=4, cutoff=(100.0, None))
    session = _session(
        signal_df,
        x="t",
        transforms=[RecipeTransform(s) for s in specs],
        filter=fspec,
        fft=FFTSpec(enabled=True),
    )

    x = signal_df["t"]
    fs_info = resolve_fs(x, None)
    y_tr, x_tr, _, _ = apply_transform_pipeline(signal_df["a"], x, specs, fs_info, col_name="a")
    y_filt, _ = apply_filter(y_tr, x_tr, fspec, fs_override=fs_info.value)
    freqs, amp = compute_fft(y_filt, fs_info.value, detrend=True, window="hann")

    assert session.fs == pytest.approx(fs_info.value)
    pd.testing.assert_series_equal(session.series("a")[0], y_tr, check_names=False)
    pd.testing.assert_series_equal(session.filtered("a").series, y_filt, check_names=False)
    result = session.fft("a")
    assert result.used_filtered
    np.testing.assert_allclose(result.freqs, freqs)
    np.testing.assert_allclose(result.amp, amp)


def test_fft_peak_and_original_signal(signal_df):
    session = _session(signal_df, x="t", fft=FFTSpec(enabled=True), fft_use_filtered=False)
    result = session.fft("b")
    assert result.ok and not result.used_filtered
    assert result.freqs[np.argmax(result.amp)] == pytest.approx(120.0, abs=0.5)


def test_manual_fs_wins(signal_df):
    session = _session(signal_df, x="t", manual_fs=500.0)
    assert session.fs == 500.0 and session.fs_info().source == "manual"
    session.recipe.manual_fs = None
    assert session.fs == pytest.approx(FS)


def test_without_x_no_fs_and_fft_message(signal_df):
    session = _session(signal_df, fft=FFTSpec(enabled=True))
    assert session.x_values() is None
    result = session.fft("a")
    assert not result.ok and "fs" in result.message


def test_irregular_x_blocks_fft(signal_df):
    df = signal_df.copy()
    df.loc[1000:, "t"] = df.loc[1000:, "t"] + 5.0  # salto: campionamento irregolare
    session = _session(df, x="t", fft=FFTSpec(enabled=True))
    if session.fs_info().is_uniform:
        pytest.skip("resolve_fs considera uniforme questo salto")
    result = session.fft("a")
    assert not result.ok and result.message


def test_filter_states(signal_df):
    off = _session(signal_df, x="t").filtered("a")
    assert not off.applied and off.series is None
    nyquist = _session(
        signal_df, x="t", filter=FilterSpec(kind="butter_lp", enabled=True, cutoff=(800.0, None))
    ).filtered("a")
    assert not nyquist.applied and nyquist.message
    ma = _session(signal_df, filter=FilterSpec(kind="ma", enabled=True, ma_window=5)).filtered("a")
    assert ma.applied and ma.series.std() < signal_df["a"].std()


def test_filter_error_is_reported(signal_df, monkeypatch):
    monkeypatch.setattr(engine, "apply_filter", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    result = _session(signal_df, filter=FilterSpec(kind="ma", enabled=True)).filtered("a")
    assert not result.applied and "Errore" in result.message


def test_fft_uses_original_when_filter_not_applicable(signal_df):
    session = _session(
        signal_df,
        x="t",
        filter=FilterSpec(kind="butter_lp", enabled=True, cutoff=(900.0, None)),
        fft=FFTSpec(enabled=True),
    )
    result = session.fft("a")
    assert result.ok and not result.used_filtered


def test_fft_too_short_message():
    df = pd.DataFrame({"t": [0.0, 0.001, 0.002], "v": [1.0, 2.0, 3.0]})
    result = _session(df, x="t", fft=FFTSpec(enabled=True)).fft("v")
    assert not result.ok and result.message


def test_transforms_only_on_targets(signal_df):
    session = _session(
        signal_df, transforms=[RecipeTransform(TransformSpec(kind="offset", constant=10.0), targets=["b"])]
    )
    pd.testing.assert_series_equal(session.series("a")[0], signal_df["a"], check_names=False)
    assert session.series("b")[0].iloc[0] == pytest.approx(signal_df["b"].iloc[0] + 10.0)


def test_invalid_transform_raises(signal_df):
    session = _session(signal_df, transforms=[RecipeTransform(TransformSpec(kind="resample", target_fs=None))])
    with pytest.raises(ValueError):
        session.series("a")


def test_fill_and_interpolate_x():
    df = pd.DataFrame({"t": [0.0, np.nan, 2.0, 3.0], "v": [1.0, np.nan, np.nan, 4.0]})
    session = _session(df, x="t", fill_missing=True, interpolate_x=True)
    y, x = session.series("v")
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert y.tolist() == [1.0, 1.0, 1.0, 4.0]
    raw = _session(df, x="t")
    assert raw.series("v")[0].isna().sum() == 2


# ---------------------------------------------------------------------------
# Riepiloghi
# ---------------------------------------------------------------------------
def test_stats_and_quality(signal_df):
    session = _session(signal_df, x="t")
    stats = session.stats()
    assert list(stats.index) == ["a", "b"]
    assert stats.loc["b", "max"] == pytest.approx(2.0, abs=1e-6)
    assert stats.loc["b", "rms"] == pytest.approx(np.sqrt(2.0), rel=1e-2)
    report = session.quality()
    assert report is session.quality()  # cache
    filtered_stats = _session(signal_df, filter=FilterSpec(kind="ma", enabled=True, ma_window=9)).stats(["a"])
    assert filtered_stats.loc["a", "std"] < stats.loc["a", "std"]
    assert _session(signal_df).stats([]).empty


def test_stats_on_empty_column():
    df = pd.DataFrame({"v": [np.nan, np.nan]})
    stats = _session(df).stats(["v"])
    assert stats.loc["v", "count"] == 0 and np.isnan(stats.loc["v", "mean"])


def test_frame_unknown_column(signal_df):
    with pytest.raises(KeyError):
        _session(signal_df).frame(["manca"])


def test_dataframe_session_properties(signal_df):
    session = _session(signal_df)
    assert session.columns == ["t", "a", "b", "testo"]
    assert session.numeric_columns == ["t", "a", "b"]
    assert session.num_rows == len(signal_df) and session.report is None and session.meta == {}


# ---------------------------------------------------------------------------
# Sessione su cache Parquet
# ---------------------------------------------------------------------------
def test_open_loads_only_needed_columns(signal_df, tmp_path):
    path = tmp_path / "sig.csv"
    signal_df.to_csv(path, index=False)
    session = AnalysisSession.open(path, cache_dir=tmp_path / "cache", recipe=Recipe(x="t", fft=FFTSpec(enabled=True)))
    assert session.num_rows == len(signal_df) and session.meta["delimiter"] == ","
    assert "a" in session.numeric_columns and session.report is not None
    result = session.fft("b")
    assert result.freqs[np.argmax(result.amp)] == pytest.approx(120.0, abs=0.5)
    assert set(session._loaded.columns) == {"t", "b"}

    reference = _session(signal_df, x="t", fft=FFTSpec(enabled=True)).fft("b")
    np.testing.assert_allclose(result.amp, reference.amp, rtol=1e-9)


# ---------------------------------------------------------------------------
# Ricetta
# ---------------------------------------------------------------------------
def test_recipe_round_trip():
    recipe = Recipe(
        x="t",
        manual_fs=250.0,
        fill_missing=True,
        transforms=[RecipeTransform(TransformSpec(kind="scale", constant=3.0), targets=["a"])],
        filter=FilterSpec(kind="butter_bp", enabled=True, order=2, cutoff=(5.0, 50.0)),
        fft=FFTSpec(enabled=True, detrend=False, window="hann"),
        fft_use_filtered=False,
    )
    data = json.loads(recipe.to_json())
    assert data["filter"]["cutoff"] == [5.0, 50.0] and data["recipe_version"] == engine.RECIPE_VERSION
    again = Recipe.from_json(recipe.to_json())
    assert again == recipe
    assert again.filter.cutoff == (5.0, 50.0)


def test_recipe_from_preset_like_dict():
    recipe = Recipe.from_dict({
        "manual_fs": 0,
        "filter": {"kind": "butter_lp", "enabled": True, "order": 4, "cutoff": [50.0], "ma_window": 5},
        "fft": {"enabled": True},
    })
    assert recipe.manual_fs is None
    assert recipe.filter.cutoff == (50.0, None)
    assert recipe.transforms == [] and recipe.fft.enabled
    with pytest.raises(ValueError):
        Recipe.from_dict({"recipe_version": "99"})
    assert Recipe.from_dict({"filter": {"kind": "ma", "cutoff": []}}).filter.cutoff is None


def test_session_recipe_api(signal_df):
    session = _session(signal_df, x="t")
    text = Recipe(x="a", manual_fs=10.0).to_json()
    assert session.from_recipe(text).recipe.x == "a"
    assert session.from_recipe({"x": "b"}).recipe.x == "b"
    assert session.from_recipe(Recipe(x="t")).to_recipe()["x"] == "t"


# ---------------------------------------------------------------------------
# ResultCache
# ---------------------------------------------------------------------------
def test_result_cache_filter_hits_and_index_change(signal_df):
    cache = ResultCache()
    fspec = FilterSpec(kind="ma", enabled=True, ma_window=3)
    y = signal_df["a"]
    first = cache.filter(y, None, fspec, None, None, "sig", "a", (False, False))
    second = cache.filter(y, None, fspec, None, None, "sig", "a", (False, False))
    pd.testing.assert_series_equal(first, second)
    assert cache.stats["filter_hits"] == 1 and cache.stats["filter_misses"] == 1
    shifted = y.iloc[::2]
    cache.filter(shifted, None, fspec, None, None, "sig", "a", (False, False))
    assert cache.stats["filter_misses"] == 2


def test_result_cache_fft_key_includes_filter(signal_df):
    cache = ResultCache()
    y = signal_df["b"]
    ma3 = FilterSpec(kind="ma", enabled=True, ma_window=3)
    ma9 = FilterSpec(kind="ma", enabled=True, ma_window=9)
    cache.fft(y, FS, "index", FFTSpec(), "s", "b", True, (False, False), ma3)
    cache.fft(y, FS, "index", FFTSpec(), "s", "b", True, (False, False), ma9)
    assert cache.stats["fft_misses"] == 2
    cache.fft(y, FS, "index", FFTSpec(), "s", "b", True, (False, False), ma3)
    assert cache.stats["fft_hits"] == 1


def test_result_cache_lru_and_clear(signal_df):
    cache = ResultCache()
    y = signal_df["a"]
    for i in range(engine.MAX_FFT_CACHE_SIZE + 1):
        cache.fft(y, FS + i, "index", FFTSpec(), "s", "a", False, (False, False))
    assert len(cache._fft) == engine.MAX_FFT_CACHE_SIZE
    cache.fft(y, FS, "index", FFTSpec(), "s", "a", False, (False, False))  # la prima è stata eliminata
    assert cache.stats["fft_hits"] == 0
    spec = [TransformSpec(kind="offset", constant=1.0)]
    out = cache.transform(y, None, spec, None, "s", "a", (False, False))
    assert cache.transform(y, None, spec, None, "s", "a", (False, False)) is out
    cache.clear()
    assert len(cache._fft) == 0 and len(cache._transform) == 0
