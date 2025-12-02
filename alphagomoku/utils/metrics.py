"""Utilities for logging training metrics to CSV files."""

import csv
import os
from typing import Dict, Any, List, Optional


def append_metrics_to_csv(
    csv_path: str,
    headers: List[str],
    values: Dict[str, Any],
    formatters: Optional[Dict[str, str]] = None
) -> None:
    """Append training metrics to a CSV file.

    Creates the file with headers if it doesn't exist.
    This is a reusable function for both single-machine and distributed training.

    Args:
        csv_path: Path to CSV file
        headers: List of column names (in order)
        values: Dictionary mapping column names to values
        formatters: Optional dict mapping column names to format strings (e.g., '.6f')
                   If not provided, values are written as-is

    Example:
        >>> append_metrics_to_csv(
        ...     'metrics.csv',
        ...     ['epoch', 'loss', 'accuracy'],
        ...     {'epoch': 1, 'loss': 0.5234, 'accuracy': 0.892},
        ...     {'loss': '.4f', 'accuracy': '.3f'}
        ... )
    """
    formatters = formatters or {}

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

    # Check if file exists
    file_exists = os.path.exists(csv_path)

    # Build row with proper formatting
    row = []
    for header in headers:
        value = values.get(header, '')

        # Apply formatter if provided and value is not empty
        if value != '' and header in formatters:
            try:
                format_str = formatters[header]
                row.append(f"{value:{format_str}}")
            except (ValueError, TypeError):
                row.append(value)
        else:
            row.append(value)

    # Write to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
