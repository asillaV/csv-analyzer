"""
Data Quality Checking Module

Provides non-blocking data quality checks for CSV analysis:
- X-axis monotonicity (duplicates/decreasing values)
- Gap detection in X-axis (irregular sampling)
- Spike detection in Y columns (robust Z-score outliers)

Returns a compact report for UI display without blocking workflow.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class QualityIssue:
    """Single quality issue found in the data"""
    issue_type: str  # 'x_non_monotonic', 'x_gap', 'y_spike'
    column: Optional[str]  # Column name (None for X-axis issues)
    count: int  # Number of violations
    percentage: float  # % of rows impacted
    examples: List[Dict[str, Any]]  # Up to max_examples with index/value details
    details: Dict[str, Any] = field(default_factory=dict)  # Additional metrics


@dataclass
class DataQualityReport:
    """Complete data quality report"""
    status: str  # 'ok', 'warning'
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def has_issues(self) -> bool:
        """Check if any issues were found"""
        return len(self.issues) > 0

    def get_summary(self) -> str:
        """Single-line summary for logging"""
        if not self.has_issues():
            return "Data quality: OK"
        issue_counts = {}
        for issue in self.issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        parts = [f"{itype}={count}" for itype, count in issue_counts.items()]
        return f"Data quality: WARNING ({', '.join(parts)})"

    def get_gap_fraction(self) -> float:
        """Get fraction of gaps in X-axis (0.0 if no gap issues)"""
        for issue in self.issues:
            if issue.issue_type == 'x_gap':
                return issue.percentage / 100.0
        return 0.0


def check_x_monotonic(x_series: pd.Series, max_examples: int = 5) -> Optional[QualityIssue]:
    """
    Check if X values are strictly increasing.

    Args:
        x_series: X-axis values (numeric or datetime)
        max_examples: Maximum number of violation examples to return

    Returns:
        QualityIssue if violations found, None otherwise
    """
    # Remove NaN values
    x_clean = x_series.dropna()

    if len(x_clean) < 2:
        return None

    # Calculate first differences
    diffs = x_clean.diff()

    # Find violations (diff <= 0, excluding first NaN)
    violations = diffs[1:] <= 0
    violation_indices = violations[violations].index.tolist()

    if len(violation_indices) == 0:
        return None

    # Collect examples
    examples = []
    for idx in violation_indices[:max_examples]:
        prev_idx = x_clean.index[x_clean.index.get_loc(idx) - 1]
        examples.append({
            'index': int(idx),
            'prev_index': int(prev_idx),
            'value': str(x_clean.loc[idx]),
            'prev_value': str(x_clean.loc[prev_idx])
        })

    return QualityIssue(
        issue_type='x_non_monotonic',
        column=None,
        count=len(violation_indices),
        percentage=100.0 * len(violation_indices) / len(x_clean),
        examples=examples,
        details={
            'total_points': len(x_clean),
            'nan_count': len(x_series) - len(x_clean)
        }
    )


def check_x_gaps(x_series: pd.Series, gap_factor_k: float = 5.0,
                 max_examples: int = 5) -> Optional[QualityIssue]:
    """
    Detect gaps in X-axis where dt > k * median(dt).

    Args:
        x_series: X-axis values (numeric or datetime)
        gap_factor_k: Threshold multiplier for median
        max_examples: Maximum number of gap examples to return

    Returns:
        QualityIssue if gaps found, None otherwise
    """
    # Remove NaN values
    x_clean = x_series.dropna()

    if len(x_clean) < 2:
        return None

    # Calculate first differences
    # Handle datetime by converting to seconds
    if pd.api.types.is_datetime64_any_dtype(x_clean):
        diffs = x_clean.diff().dt.total_seconds()
    else:
        diffs = x_clean.diff()

    # Remove first NaN and negative/zero diffs
    diffs_clean = diffs[1:]
    diffs_positive = diffs_clean[diffs_clean > 0]

    if len(diffs_positive) == 0:
        return None

    # Calculate median and threshold
    median_dt = diffs_positive.median()
    threshold = gap_factor_k * median_dt

    # Find gaps
    gaps = diffs_clean > threshold
    gap_indices = gaps[gaps].index.tolist()

    if len(gap_indices) == 0:
        return None

    # Collect examples
    examples = []
    for idx in gap_indices[:max_examples]:
        prev_idx = x_clean.index[x_clean.index.get_loc(idx) - 1]
        gap_size = diffs_clean.loc[idx]
        examples.append({
            'index': int(idx),
            'prev_index': int(prev_idx),
            'gap_size': float(gap_size),
            'gap_ratio': float(gap_size / median_dt)
        })

    return QualityIssue(
        issue_type='x_gap',
        column=None,
        count=len(gap_indices),
        percentage=100.0 * len(gap_indices) / len(diffs_clean),
        examples=examples,
        details={
            'median_dt': float(median_dt),
            'threshold': float(threshold),
            'gap_factor_k': gap_factor_k,
            'total_intervals': len(diffs_clean),
            'nan_count': len(x_series) - len(x_clean)
        }
    )


def check_y_spikes(y_series: pd.Series, column_name: str, spike_z: float = 4.0,
                   max_examples: int = 5) -> Optional[QualityIssue]:
    """
    Detect outliers in Y using robust Z-score (median + MAD).

    Args:
        y_series: Y column values
        column_name: Name of the Y column
        spike_z: Z-score threshold (|z| >= threshold means spike)
        max_examples: Maximum number of spike examples to return

    Returns:
        QualityIssue if spikes found, None otherwise
    """
    # Remove NaN and try to convert to numeric
    y_clean = pd.to_numeric(y_series, errors='coerce').dropna()

    if len(y_clean) < 4:  # Need at least a few points for robust stats
        return None

    # Calculate robust statistics
    median_y = y_clean.median()
    mad = np.median(np.abs(y_clean - median_y))

    # Check for constant signal (MAD ~ 0)
    if mad < 1e-10:
        # Return special note for constant signal
        return QualityIssue(
            issue_type='y_constant',
            column=column_name,
            count=0,
            percentage=0.0,
            examples=[],
            details={
                'median': float(median_y),
                'mad': float(mad),
                'note': 'Signal has no variability (constant or near-constant values)'
            }
        )

    # Calculate robust Z-scores
    # MAD scaling factor: 1.4826 for normal distribution
    z_scores = 0.6745 * (y_clean - median_y) / mad

    # Find spikes
    spikes = np.abs(z_scores) >= spike_z
    spike_indices = z_scores[spikes].index.tolist()

    if len(spike_indices) == 0:
        return None

    # Collect examples (sorted by absolute z-score)
    spike_data = [(idx, float(z_scores.loc[idx]), float(y_clean.loc[idx]))
                  for idx in spike_indices]
    spike_data.sort(key=lambda x: abs(x[1]), reverse=True)

    examples = []
    for idx, z, val in spike_data[:max_examples]:
        examples.append({
            'index': int(idx),
            'value': val,
            'z_score': z
        })

    return QualityIssue(
        issue_type='y_spike',
        column=column_name,
        count=len(spike_indices),
        percentage=100.0 * len(spike_indices) / len(y_clean),
        examples=examples,
        details={
            'median': float(median_y),
            'mad': float(mad),
            'spike_z': spike_z,
            'total_points': len(y_clean),
            'nan_count': len(y_series) - len(y_clean)
        }
    )


def run_quality_checks(
    df: pd.DataFrame,
    x_col: Optional[str],
    y_cols: List[str],
    gap_factor_k: float = 5.0,
    spike_z: float = 4.0,
    min_points: int = 20,
    max_examples: int = 5
) -> DataQualityReport:
    """
    Run all quality checks on the dataset.

    Args:
        df: DataFrame to check
        x_col: X-axis column name (None = use index)
        y_cols: List of Y column names to check
        gap_factor_k: Gap detection threshold multiplier
        spike_z: Spike detection Z-score threshold
        min_points: Minimum points required for robust checks
        max_examples: Maximum examples per issue

    Returns:
        DataQualityReport with all findings
    """
    report = DataQualityReport(
        status='ok',
        config={
            'gap_factor_k': gap_factor_k,
            'spike_z': spike_z,
            'min_points': min_points,
            'max_examples': max_examples
        }
    )

    # Get X series
    if x_col is not None and x_col in df.columns:
        x_series = df[x_col]
    else:
        x_series = pd.Series(df.index, index=df.index, name='index')
        if x_col is None:
            report.notes.append("Using DataFrame index as X-axis")

    report.metrics['total_rows'] = len(df)

    # Check if dataset is too small
    if len(df) < min_points:
        report.notes.append(
            f"Dataset has only {len(df)} points (< {min_points}). "
            "Quality checks may be less reliable."
        )

    # Check X monotonicity
    issue = check_x_monotonic(x_series, max_examples)
    if issue:
        report.issues.append(issue)

    # Check X gaps
    issue = check_x_gaps(x_series, gap_factor_k, max_examples)
    if issue:
        report.issues.append(issue)
        # Add soft warning about irregular sampling
        gap_pct = issue.percentage
        if gap_pct > 5.0:  # More than 5% gaps
            report.notes.append(
                f"Irregular sampling detected ({gap_pct:.1f}% gaps). "
                "FFT/filter results may be less reliable."
            )

    # Check Y spikes for each column
    for y_col in y_cols:
        if y_col not in df.columns:
            continue

        y_series = df[y_col]
        issue = check_y_spikes(y_series, y_col, spike_z, max_examples)
        if issue:
            # y_constant is informational only, not a warning
            if issue.issue_type == 'y_constant':
                report.notes.append(
                    f"Column '{y_col}' has no variability (constant signal)"
                )
            else:
                report.issues.append(issue)

    # Set final status
    if report.has_issues():
        report.status = 'warning'

    return report
