#!/usr/bin/env python3
"""
Test script to verify the correctness of _is_value_empty_vectorized implementation
This ensures the vectorized version produces identical results to the original
"""

import pandas as pd
import numpy as np
from typing import List


def _is_value_empty(value) -> bool:
    """Original implementation (non-vectorized)"""
    if pd.isna(value): 
        return True
    if isinstance(value, str) and not value.strip(): 
        return True
    return False


def _is_value_empty_vectorized(series: pd.Series) -> pd.Series:
    """Vectorized implementation (from the PR)"""
    # First layer: Check for NA values (NaN, None, pd.NA)
    is_na = series.isna()
    
    # Second layer: Check for blank strings (only for object dtype)
    if series.dtype == 'object':
        # Pandas .str accessor behavior:
        # - String values: performs strip() operation
        # - Non-string values (int, float, etc.): returns NaN
        # Therefore, == '' only matches truly blank strings
        is_blank_str = series.str.strip() == ''
        return is_na | is_blank_str
    else:
        # For numeric dtypes, only NA check is needed
        return is_na


def test_vectorized_implementation():
    """Test vectorized implementation against original for various test cases"""
    
    print("Testing _is_value_empty_vectorized implementation...")
    print("=" * 80)
    
    # Test Case 1: Object dtype with various string types
    print("\nTest Case 1: Object dtype with strings")
    test_data_1 = pd.Series([
        "valid string",      # non-empty
        "",                  # empty string
        "  ",                # whitespace only
        "\t\n",              # tabs and newlines
        None,                # None value
        np.nan,              # NaN value
        pd.NA,               # pandas NA
        "   text   ",        # string with surrounding whitespace (not empty)
    ], dtype='object')
    
    # Apply original method element-wise
    original_result_1 = test_data_1.apply(_is_value_empty)
    # Apply vectorized method
    vectorized_result_1 = _is_value_empty_vectorized(test_data_1)
    
    print(f"Original result:    {original_result_1.tolist()}")
    print(f"Vectorized result:  {vectorized_result_1.tolist()}")
    assert original_result_1.equals(vectorized_result_1), "Test Case 1 FAILED: Results don't match!"
    print("✓ Test Case 1 PASSED")
    
    # Test Case 2: Numeric dtype
    print("\nTest Case 2: Numeric dtype (int)")
    test_data_2 = pd.Series([1, 2, 0, -5, np.nan], dtype='float64')
    
    original_result_2 = test_data_2.apply(_is_value_empty)
    vectorized_result_2 = _is_value_empty_vectorized(test_data_2)
    
    print(f"Original result:    {original_result_2.tolist()}")
    print(f"Vectorized result:  {vectorized_result_2.tolist()}")
    assert original_result_2.equals(vectorized_result_2), "Test Case 2 FAILED: Results don't match!"
    print("✓ Test Case 2 PASSED")
    
    # Test Case 3: Mixed content in object dtype
    print("\nTest Case 3: Mixed content in object dtype")
    test_data_3 = pd.Series([
        "text",
        123,       # integer in object dtype
        45.67,     # float in object dtype
        "",
        "  ",
        None,
        True,      # boolean in object dtype
        False,     # boolean in object dtype
    ], dtype='object')
    
    original_result_3 = test_data_3.apply(_is_value_empty)
    vectorized_result_3 = _is_value_empty_vectorized(test_data_3)
    
    print(f"Original result:    {original_result_3.tolist()}")
    print(f"Vectorized result:  {vectorized_result_3.tolist()}")
    assert original_result_3.equals(vectorized_result_3), "Test Case 3 FAILED: Results don't match!"
    print("✓ Test Case 3 PASSED")
    
    # Test Case 4: Empty Series
    print("\nTest Case 4: Empty Series")
    test_data_4 = pd.Series([], dtype='object')
    
    original_result_4 = test_data_4.apply(_is_value_empty)
    vectorized_result_4 = _is_value_empty_vectorized(test_data_4)
    
    print(f"Original result:    {original_result_4.tolist()}")
    print(f"Vectorized result:  {vectorized_result_4.tolist()}")
    # For empty series, both should be empty and have the same dtype
    assert len(original_result_4) == len(vectorized_result_4) == 0, "Test Case 4 FAILED: Results don't match!"
    print("✓ Test Case 4 PASSED")
    
    # Test Case 5: All empty values
    print("\nTest Case 5: All empty values")
    test_data_5 = pd.Series(["", "  ", None, np.nan, "\t"], dtype='object')
    
    original_result_5 = test_data_5.apply(_is_value_empty)
    vectorized_result_5 = _is_value_empty_vectorized(test_data_5)
    
    print(f"Original result:    {original_result_5.tolist()}")
    print(f"Vectorized result:  {vectorized_result_5.tolist()}")
    assert original_result_5.equals(vectorized_result_5), "Test Case 5 FAILED: Results don't match!"
    print("✓ Test Case 5 PASSED")
    
    # Test Case 6: Boolean dtype
    print("\nTest Case 6: Boolean dtype")
    test_data_6 = pd.Series([True, False, True, False], dtype='bool')
    
    original_result_6 = test_data_6.apply(_is_value_empty)
    vectorized_result_6 = _is_value_empty_vectorized(test_data_6)
    
    print(f"Original result:    {original_result_6.tolist()}")
    print(f"Vectorized result:  {vectorized_result_6.tolist()}")
    # Note: For boolean dtype, both implementations should return False for all values
    # since booleans are never considered empty unless they are NA
    assert original_result_6.equals(vectorized_result_6), "Test Case 6 FAILED: Results don't match!"
    print("✓ Test Case 6 PASSED")
    
    # Test Case 7: Large dataset performance test
    print("\nTest Case 7: Large dataset correctness (100,000 rows)")
    large_data = pd.Series(
        ["valid"] * 30000 + 
        [""] * 20000 + 
        ["  "] * 20000 + 
        [None] * 15000 + 
        [np.nan] * 15000,
        dtype='object'
    )
    
    original_result_7 = large_data.apply(_is_value_empty)
    vectorized_result_7 = _is_value_empty_vectorized(large_data)
    
    matches = (original_result_7 == vectorized_result_7).sum()
    total = len(large_data)
    print(f"Matching results: {matches}/{total}")
    assert original_result_7.equals(vectorized_result_7), "Test Case 7 FAILED: Results don't match!"
    print("✓ Test Case 7 PASSED")
    
    print("\n" + "=" * 80)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("The vectorized implementation is semantically equivalent to the original!")
    print("=" * 80)


if __name__ == "__main__":
    test_vectorized_implementation()
