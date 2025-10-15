"""
Test suite for data quality checks module.

Tests cover acceptance criteria:
1. X monotonicity violations
2. Gap detection in X
3. Spike detection in Y with NaN handling
4. Short sample handling
5. Missing column handling
6. Performance with large datasets
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.quality import run_quality_checks, check_x_monotonic, check_x_gaps, check_y_spikes


def test_x_monotonic_detects_duplicates():
    """Test that duplicate timestamps are detected as non-monotonic."""
    # Create series with duplicate value
    x_series = pd.Series([0.0, 1.0, 1.0, 2.0, 3.0])

    issue = check_x_monotonic(x_series, max_examples=5)

    assert issue is not None
    assert issue.issue_type == 'x_non_monotonic'
    assert issue.count == 1  # One violation (index 2 <= index 1)
    assert len(issue.examples) == 1
    assert issue.examples[0]['index'] == 2
    assert issue.examples[0]['prev_index'] == 1


def test_x_monotonic_detects_decreasing():
    """Test that decreasing values are detected as non-monotonic."""
    x_series = pd.Series([0.0, 1.0, 2.0, 1.5, 3.0])

    issue = check_x_monotonic(x_series, max_examples=5)

    assert issue is not None
    assert issue.count == 1
    assert issue.examples[0]['index'] == 3


def test_x_gaps_large_gap_detected():
    """Test that large gaps are detected with 10x median."""
    # Regular spacing of 1, then a gap of 10
    x_series = pd.Series([0.0, 1.0, 2.0, 12.0, 13.0])

    issue = check_x_gaps(x_series, gap_factor_k=5.0, max_examples=5)

    assert issue is not None
    assert issue.issue_type == 'x_gap'
    assert issue.count == 1
    assert issue.details['median_dt'] == 1.0
    assert issue.details['threshold'] == 5.0
    assert issue.examples[0]['gap_size'] == 10.0


def test_x_gaps_datetime():
    """Test gap detection with datetime X axis."""
    # Create datetime series with gap
    dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-10'])
    x_series = pd.Series(dates)

    issue = check_x_gaps(x_series, gap_factor_k=3.0, max_examples=5)

    assert issue is not None
    assert issue.count == 1
    # Gap is 7 days = 7*86400 seconds, median is 1 day = 86400 seconds
    assert issue.details['median_dt'] == 86400.0
    assert issue.examples[0]['gap_ratio'] > 5.0  # 7 days vs 1 day


def test_y_spikes_single_outlier():
    """Test spike detection with single extreme value."""
    # Create signal with one extreme spike
    y_series = pd.Series([1.0, 1.1, 1.0, 15.0, 1.1, 1.0, 0.9])

    issue = check_y_spikes(y_series, 'test_col', spike_z=4.0, max_examples=5)

    assert issue is not None
    assert issue.issue_type == 'y_spike'
    assert issue.count == 1
    assert issue.column == 'test_col'
    assert abs(issue.examples[0]['z_score']) > 4.0
    assert issue.examples[0]['index'] == 3


def test_y_spikes_with_nan():
    """Test spike detection handles NaN values correctly."""
    y_series = pd.Series([1.0, np.nan, 1.1, 20.0, np.nan, 1.0])

    issue = check_y_spikes(y_series, 'test_col', spike_z=3.0, max_examples=5)

    assert issue is not None
    assert issue.details['nan_count'] == 2
    assert issue.count == 1  # Only the spike at index 3


def test_y_constant_signal():
    """Test constant signal returns special note."""
    y_series = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])

    issue = check_y_spikes(y_series, 'test_col', spike_z=4.0, max_examples=5)

    assert issue is not None
    assert issue.issue_type == 'y_constant'
    assert issue.count == 0
    assert 'no variability' in issue.details['note'].lower()


def test_run_quality_checks_all_ok():
    """Test full quality check with clean data."""
    df = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 3.0, 4.0],
        'value': [1.0, 1.1, 1.0, 0.9, 1.0]
    })

    report = run_quality_checks(
        df=df,
        x_col='time',
        y_cols=['value'],
        gap_factor_k=5.0,
        spike_z=4.0,
        min_points=3,
        max_examples=5
    )

    assert report.status == 'ok'
    assert not report.has_issues()


def test_run_quality_checks_mixed_issues():
    """Test quality check detects multiple issue types."""
    df = pd.DataFrame({
        'time': [0.0, 1.0, 1.0, 2.0, 12.0, 13.0],  # Duplicate and gap
        'value': [1.0, 1.1, 1.0, 20.0, 1.0, 0.9]   # Spike
    })

    report = run_quality_checks(
        df=df,
        x_col='time',
        y_cols=['value'],
        gap_factor_k=3.0,
        spike_z=3.0,
        min_points=3,
        max_examples=5
    )

    assert report.status == 'warning'
    assert report.has_issues()
    issue_types = {issue.issue_type for issue in report.issues}
    assert 'x_non_monotonic' in issue_types
    assert 'x_gap' in issue_types
    assert 'y_spike' in issue_types


def test_run_quality_checks_short_sample():
    """Test short sample adds note without failing."""
    df = pd.DataFrame({
        'time': [0.0, 1.0, 2.0],
        'value': [1.0, 1.1, 1.2]
    })

    report = run_quality_checks(
        df=df,
        x_col='time',
        y_cols=['value'],
        gap_factor_k=5.0,
        spike_z=4.0,
        min_points=20,
        max_examples=5
    )

    assert report.status == 'ok'  # No issues despite being short
    assert any('only 3 points' in note.lower() for note in report.notes)


def test_run_quality_checks_use_index():
    """Test quality check uses DataFrame index when x_col is None."""
    df = pd.DataFrame({
        'value': [1.0, 1.1, 1.0, 0.9]
    })

    report = run_quality_checks(
        df=df,
        x_col=None,
        y_cols=['value'],
        gap_factor_k=5.0,
        spike_z=4.0,
        min_points=3,
        max_examples=5
    )

    assert report.status == 'ok'
    assert any('index' in note.lower() for note in report.notes)


def test_run_quality_checks_missing_column():
    """Test handles missing Y column gracefully."""
    df = pd.DataFrame({
        'time': [0.0, 1.0, 2.0],
        'value': [1.0, 1.1, 1.2]
    })

    report = run_quality_checks(
        df=df,
        x_col='time',
        y_cols=['value', 'missing_col'],
        gap_factor_k=5.0,
        spike_z=4.0,
        min_points=3,
        max_examples=5
    )

    # Should only check existing column, no crash
    assert report.status == 'ok'


def test_run_quality_checks_irregular_sampling_warning():
    """Test adds note about irregular sampling when gaps > 5%."""
    # Create data with 2 gaps out of 10 intervals = 20%
    df = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 10.0, 11.0, 19.0, 20.0],
        'value': [1.0, 1.1, 1.0, 0.9, 1.0, 1.1, 1.0]
    })

    report = run_quality_checks(
        df=df,
        x_col='time',
        y_cols=['value'],
        gap_factor_k=3.0,
        spike_z=4.0,
        min_points=3,
        max_examples=5
    )

    assert report.status == 'warning'
    # Should have note about irregular sampling
    assert any('irregular sampling' in note.lower() for note in report.notes)
    # Should have high gap percentage
    gap_frac = report.get_gap_fraction()
    assert gap_frac > 0.05


def test_performance_large_dataset():
    """Smoke test: ensure quality checks complete in reasonable time for 1M rows."""
    rows = 1_000_000
    df = pd.DataFrame({
        'time': np.arange(rows, dtype=float),
        'value': np.sin(np.linspace(0, 50, rows))
    })

    import time
    start = time.time()

    report = run_quality_checks(
        df=df,
        x_col='time',
        y_cols=['value'],
        gap_factor_k=5.0,
        spike_z=4.0,
        min_points=20,
        max_examples=5
    )

    elapsed = time.time() - start

    assert report.status == 'ok'
    assert not report.has_issues()
    # Should complete in reasonable time (< 5 seconds for 1M rows)
    assert elapsed < 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
