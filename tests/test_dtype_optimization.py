"""
Test per optimize_dtypes() - Issue #53

Verifica che la conversione float64→float32 avvenga correttamente e in sicurezza.
"""
import numpy as np
import pandas as pd
import pytest

from core.loader import optimize_dtypes, FLOAT32_MAX


def test_optimize_dtypes_converts_float64_to_float32():
    """Verifica che colonne float64 vengano convertite a float32 quando i valori rientrano nei limiti."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [10.5, 20.7, 30.9]
    })
    assert df["a"].dtype == "float64"
    assert df["b"].dtype == "float64"

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    # Verifico che siano stati convertiti
    assert df_opt["a"].dtype == "float32"
    assert df_opt["b"].dtype == "float32"

    # Verifico che i valori siano invariati (dentro tolleranza float32)
    np.testing.assert_allclose(df_opt["a"].values, df["a"].values, rtol=1e-6)
    np.testing.assert_allclose(df_opt["b"].values, df["b"].values, rtol=1e-6)

    # Verifico conversions map
    assert conversions == {"a": "float64→float32", "b": "float64→float32"}


def test_optimize_dtypes_preserves_values_with_nan():
    """Verifica che NaN siano preservati correttamente."""
    df = pd.DataFrame({
        "x": [1.0, np.nan, 3.0, np.nan, 5.0]
    })

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    assert df_opt["x"].dtype == "float32"
    assert df_opt["x"].isna().sum() == 2
    assert df_opt.loc[0, "x"] == pytest.approx(1.0, rel=1e-6)
    assert pd.isna(df_opt.loc[1, "x"])
    assert df_opt.loc[2, "x"] == pytest.approx(3.0, rel=1e-6)


def test_optimize_dtypes_handles_all_nan_column():
    """Verifica che colonne con tutti NaN siano convertite senza errori."""
    df = pd.DataFrame({
        "all_nan": [np.nan, np.nan, np.nan]
    })

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    assert df_opt["all_nan"].dtype == "float32"
    assert df_opt["all_nan"].isna().all()
    assert "all_nan" in conversions


def test_optimize_dtypes_handles_inf():
    """Verifica che Inf sia gestito correttamente."""
    df = pd.DataFrame({
        "with_inf": [1.0, np.inf, -np.inf, 10.0]
    })

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    assert df_opt["with_inf"].dtype == "float32"
    assert np.isinf(df_opt.loc[1, "with_inf"])
    assert np.isinf(df_opt.loc[2, "with_inf"])
    assert df_opt.loc[3, "with_inf"] == pytest.approx(10.0, rel=1e-6)


def test_optimize_dtypes_preserves_large_values():
    """Verifica che valori troppo grandi per float32 NON siano convertiti."""
    large_value = FLOAT32_MAX * 1.5  # Oltre il limite float32
    df = pd.DataFrame({
        "too_large": [1.0, large_value, 3.0]
    })

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    # Colonna NON convertita
    assert df_opt["too_large"].dtype == "float64"
    assert "too_large" not in conversions


def test_optimize_dtypes_mixed_types():
    """Verifica che colonne non-float64 siano preservate."""
    df = pd.DataFrame({
        "float64_col": [1.0, 2.0, 3.0],
        "int_col": [1, 2, 3],
        "str_col": ["a", "b", "c"],
        "float32_col": pd.Series([1.0, 2.0, 3.0], dtype="float32")
    })

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    # Solo float64_col convertita
    assert df_opt["float64_col"].dtype == "float32"
    assert df_opt["int_col"].dtype == df["int_col"].dtype  # Invariata
    assert df_opt["str_col"].dtype == df["str_col"].dtype  # Invariata
    assert df_opt["float32_col"].dtype == "float32"  # Invariata

    assert conversions == {"float64_col": "float64→float32"}


def test_optimize_dtypes_disabled():
    """Verifica che con enabled=False non avvenga conversione."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0]
    })

    df_opt, conversions = optimize_dtypes(df, enabled=False)

    assert df_opt["a"].dtype == "float64"
    assert conversions == {}


def test_optimize_dtypes_negative_values():
    """Verifica che valori negativi siano gestiti correttamente."""
    df = pd.DataFrame({
        "neg": [-1e10, -500.0, -0.001, 0.0, 0.001, 500.0, 1e10]
    })

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    assert df_opt["neg"].dtype == "float32"
    np.testing.assert_allclose(df_opt["neg"].values, df["neg"].values, rtol=1e-6)


def test_optimize_dtypes_boundary_values():
    """Verifica comportamento al limite di float32."""
    # Valore appena sotto il limite
    safe_value = FLOAT32_MAX * 0.9
    df = pd.DataFrame({
        "safe": [safe_value, -safe_value, 1.0]
    })

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    assert df_opt["safe"].dtype == "float32"
    assert "safe" in conversions


def test_optimize_dtypes_precision_loss():
    """Verifica che la perdita di precisione float64→float32 sia accettabile."""
    # Valore con alta precisione (>7 cifre significative)
    high_precision = 1.23456789012345  # float64 precision

    df = pd.DataFrame({
        "precise": [high_precision, high_precision * 2]
    })

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    # Conversione avvenuta
    assert df_opt["precise"].dtype == "float32"

    # Precisione ridotta ma entro tolleranza float32 (~7 cifre significative)
    # float32 ha ~6-7 cifre decimali di precisione
    np.testing.assert_allclose(df_opt["precise"].values, df["precise"].values, rtol=1e-6)


def test_optimize_dtypes_empty_dataframe():
    """Verifica che DataFrame vuoti siano gestiti correttamente."""
    df = pd.DataFrame()

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    assert len(df_opt) == 0
    assert conversions == {}


def test_optimize_dtypes_no_float64_columns():
    """Verifica che DF senza colonne float64 siano gestiti correttamente."""
    df = pd.DataFrame({
        "int": [1, 2, 3],
        "str": ["a", "b", "c"]
    })

    df_opt, conversions = optimize_dtypes(df, enabled=True)

    assert conversions == {}
    assert df_opt.equals(df)


def test_optimize_dtypes_immutability():
    """Verifica che il DataFrame originale non sia mutato."""
    df_original = pd.DataFrame({
        "a": [1.0, 2.0, 3.0]
    })
    df_copy_before = df_original.copy()

    df_opt, _ = optimize_dtypes(df_original, enabled=True)

    # DataFrame originale invariato
    assert df_original["a"].dtype == "float64"
    pd.testing.assert_frame_equal(df_original, df_copy_before)

    # DataFrame ottimizzato è diverso
    assert df_opt["a"].dtype == "float32"
