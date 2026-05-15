"""Tests for preprocessing and modeling modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.preprocessing import (
    extract_numeric_value,
    clean_column_names,
    handle_missing_values,
    remove_duplicates
)


class TestPreprocessing:
    """Test preprocessing functions."""
    
    def test_extract_numeric_value_simple(self):
        """Test extraction of simple numeric values."""
        assert extract_numeric_value("963 hp") == 963.0
        assert extract_numeric_value("340 km/h") == 340.0
        assert extract_numeric_value("$1,100,000") == 1100000.0
    
    def test_extract_numeric_value_range(self):
        """Test extraction of range values (should return average)."""
        result = extract_numeric_value("70-85")
        assert 77.0 <= result <= 78.0  # Average of 70 and 85
    
    def test_extract_numeric_value_nan(self):
        """Test handling of NaN values."""
        assert np.isnan(extract_numeric_value(np.nan))
        assert np.isnan(extract_numeric_value(None))
    
    def test_clean_column_names(self):
        """Test column name cleaning."""
        df = pd.DataFrame({
            'Company Names': ['Ferrari'],
            'Cars Prices': [100000]
        })
        cleaned = clean_column_names(df)
        assert 'company' in cleaned.columns
        assert 'price' in cleaned.columns
        assert 'Company Names' not in cleaned.columns
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'A': [1, 1, 2],
            'B': [3, 3, 4]
        })
        cleaned = remove_duplicates(df)
        assert len(cleaned) == 2
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, np.nan, 4.0],
            'categorical': ['A', 'B', None, 'A']
        })
        cleaned = handle_missing_values(df)
        assert not cleaned['numeric'].isnull().any()
        assert not cleaned['categorical'].isnull().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
