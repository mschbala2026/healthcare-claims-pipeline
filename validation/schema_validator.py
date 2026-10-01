"""
Schema and Data Quality Validation

Validates that input data conforms to defined contracts and catches
schema changes before they cause production failures.

Key validations:
  1. Required columns present and correct types
  2. No unexpected NULL values
  3. Values within valid ranges
  4. No impossible date combinations
  5. No duplicate keys
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates data against predefined schema contracts"""
    
    def __init__(self, raise_on_error: bool = False):
        """
        Args:
            raise_on_error: If True, raise exception on validation failure.
                          If False, log and return False.
        """
        self.raise_on_error = raise_on_error
        self.validation_results = {}
    
    def _report_error(self, message: str, is_critical: bool = False):
        """Log error and optionally raise"""
        level = "CRITICAL" if is_critical else "WARNING"
        logger.error(f"[{level}] {message}")
        if self.raise_on_error and is_critical:
            raise ValueError(message)
    
    def _report_success(self, message: str):
        """Log successful validation"""
        logger.info(f"✓ {message}")
    
    # ========================================================================
    # COLUMN & TYPE VALIDATION
    # ========================================================================
    
    def validate_columns_present(self, df: pd.DataFrame, required_columns: List[str],
                                 table_name: str = "Table") -> bool:
        """Check that all required columns exist"""
        missing = set(required_columns) - set(df.columns)
        
        if missing:
            msg = f"{table_name}: Missing required columns: {missing}"
            self._report_error(msg, is_critical=True)
            return False
        
        self._report_success(f"{table_name}: All {len(required_columns)} columns present")
        return True
    
    def validate_column_types(self, df: pd.DataFrame, 
                             expected_types: Dict[str, str],
                             table_name: str = "Table") -> bool:
        """
        Validate column data types.
        
        Args:
            df: DataFrame to validate
            expected_types: Dict mapping column_name -> 'int', 'float', 'string', 'date'
            table_name: Name of table for error messages
        
        Returns:
            True if all types match
        """
        errors = []
        
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue
            
            actual_type = df[col].dtype
            
            # Map pandas dtypes to expected types
            type_map = {
                'int': ['int64', 'int32', 'int16', 'int8'],
                'float': ['float64', 'float32'],
                'string': ['object', 'string'],
                'date': ['datetime64[ns]'],
            }
            
            expected_types_list = type_map.get(expected_type, [expected_type])
            
            if str(actual_type) not in expected_types_list:
                errors.append(
                    f"  {col}: expected {expected_type}, got {actual_type}"
                )
        
        if errors:
            msg = f"{table_name} type mismatches:\n" + "\n".join(errors)
            self._report_error(msg, is_critical=True)
            return False
        
        self._report_success(f"{table_name}: All column types correct")
        return True
    
    # ========================================================================
    # NULL & MISSING VALUE VALIDATION
    # ========================================================================
    
    def validate_not_null(self, df: pd.DataFrame, columns: List[str],
                         table_name: str = "Table") -> bool:
        """Check that required columns have no NULL values"""
        errors = []
        
        for col in columns:
            if col not in df.columns:
                errors.append(f"  {col}: column not found")
                continue
            
            null_count = df[col].isnull().sum()
            if null_count > 0:
                pct = 100 * null_count / len(df)
                errors.append(
                    f"  {col}: {null_count:,} NULL values ({pct:.2f}%)"
                )
        
        if errors:
            msg = f"{table_name} NULL validation failed:\n" + "\n".join(errors)
            self._report_error(msg, is_critical=True)
            return False
        
        self._report_success(f"{table_name}: No NULLs in {len(columns)} required columns")
        return True
    
    # ========================================================================
    # VALUE RANGE VALIDATION
    # ========================================================================
    
    def validate_value_range(self, df: pd.DataFrame, column: str,
                            min_value: Optional[float] = None,
                            max_value: Optional[float] = None,
                            table_name: str = "Table") -> bool:
        """Check that numeric values are within expected range"""
        if column not in df.columns:
            self._report_error(f"{table_name}.{column}: column not found", is_critical=True)
            return False
        
        errors = []
        
        # Remove NULLs for range check
        non_null = df[column].dropna()
        
        if len(non_null) == 0:
            return True
        
        if min_value is not None:
            out_of_range = (non_null < min_value).sum()
            if out_of_range > 0:
                pct = 100 * out_of_range / len(non_null)
                min_actual = non_null.min()
                errors.append(
                    f"  {column}: {out_of_range:,} values < {min_value} "
                    f"(min: {min_actual}, {pct:.2f}%)"
                )
        
        if max_value is not None:
            out_of_range = (non_null > max_value).sum()
            if out_of_range > 0:
                pct = 100 * out_of_range / len(non_null)
                max_actual = non_null.max()
                errors.append(
                    f"  {column}: {out_of_range:,} values > {max_value} "
                    f"(max: {max_actual}, {pct:.2f}%)"
                )
        
        if errors:
            msg = f"{table_name} range validation failed:\n" + "\n".join(errors)
            self._report_error(msg, is_critical=True)
            return False
        
        range_desc = f"[{min_value}, {max_value}]" if min_value and max_value else ""
        self._report_success(f"{table_name}.{column}: values in range {range_desc}")
        return True
    
    # ========================================================================
    # CATEGORICAL VALUE VALIDATION
    # ========================================================================
    
    def validate_enum_values(self, df: pd.DataFrame, column: str,
                            valid_values: List,
                            table_name: str = "Table") -> bool:
        """Check that categorical column only has expected values"""
        if column not in df.columns:
            self._report_error(f"{table_name}.{column}: column not found", is_critical=True)
            return False
        
        # Get unique non-null values
        actual_values = set(df[column].dropna().unique())
        valid_set = set(valid_values)
        
        unexpected = actual_values - valid_set
        
        if unexpected:
            msg = (f"{table_name}.{column}: unexpected values: {unexpected}\n"
                   f"  Expected one of: {valid_values}")
            self._report_error(msg, is_critical=True)
            return False
        
        self._report_success(f"{table_name}.{column}: all values in {valid_values}")
        return True
    
    # ========================================================================
    # KEY UNIQUENESS VALIDATION
    # ========================================================================
    
    def validate_primary_key(self, df: pd.DataFrame, key_columns: List[str],
                            table_name: str = "Table") -> bool:
        """Check that primary key is unique"""
        if not all(col in df.columns for col in key_columns):
            self._report_error(f"{table_name}: key columns not found", is_critical=True)
            return False
        
        duplicates = df.duplicated(subset=key_columns, keep=False).sum()
        
        if duplicates > 0:
            msg = f"{table_name}: {duplicates} duplicate rows on key {key_columns}"
            self._report_error(msg, is_critical=True)
            return False
        
        self._report_success(f"{table_name}: {len(df):,} rows, primary key unique")
        return True
    
    # ========================================================================
    # DATE CONSISTENCY VALIDATION
    # ========================================================================
    
    def validate_date_ordering(self, df: pd.DataFrame, 
                              start_col: str, end_col: str,
                              table_name: str = "Table") -> bool:
        """Check that start_date <= end_date"""
        if not all(col in df.columns for col in [start_col, end_col]):
            self._report_error(f"{table_name}: date columns not found", is_critical=True)
            return False
        
        # Parse dates
        try:
            start_dates = pd.to_datetime(df[start_col], format='%Y%m%d', errors='coerce')
            end_dates = pd.to_datetime(df[end_col], format='%Y%m%d', errors='coerce')
        except Exception as e:
            self._report_error(f"{table_name}: date parsing failed: {e}", is_critical=True)
            return False
        
        # Check ordering
        mask = (start_dates <= end_dates) | start_dates.isnull() | end_dates.isnull()
        violations = (~mask).sum()
        
        if violations > 0:
            pct = 100 * violations / len(df)
            msg = (f"{table_name}: {violations:,} rows with "
                   f"{start_col} > {end_col} ({pct:.2f}%)")
            self._report_error(msg, is_critical=True)
            return False
        
        self._report_success(f"{table_name}: Date ordering {start_col} <= {end_col} OK")
        return True
    
    # ========================================================================
    # LEAKAGE DETECTION
    # ========================================================================
    
    def validate_no_future_dates(self, df: pd.DataFrame, date_column: str,
                                feature_date: str, 
                                table_name: str = "Table") -> bool:
        """
        CRITICAL: Check that no data after feature_date exists.
        This is the primary leakage detection mechanism.
        """
        try:
            dates = pd.to_datetime(df[date_column], format='%Y%m%d', errors='coerce')
            feature_dt = pd.to_datetime(feature_date)
        except Exception as e:
            self._report_error(f"{table_name}: date parsing failed: {e}", is_critical=True)
            return False
        
        # CRITICAL: Check for dates >= feature_date (should be none)
        future_mask = dates >= feature_dt
        future_count = future_mask.sum()
        
        if future_count > 0:
            pct = 100 * future_count / len(df)
            max_future = dates[future_mask].max()
            msg = (f"[LEAKAGE DETECTED] {table_name}: {future_count:,} rows with "
                   f"{date_column} >= feature_date {feature_date} "
                   f"(max: {max_future}, {pct:.2f}%)")
            self._report_error(msg, is_critical=True)
            return False
        
        self._report_success(f"{table_name}: No future dates after {feature_date}")
        return True
    
    # ========================================================================
    # BENEFICIARY DATA VALIDATION
    # ========================================================================
    
    def validate_beneficiary_data(self, df: pd.DataFrame) -> bool:
        """Run all beneficiary-specific validations"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATING BENEFICIARY DATA")
        logger.info("=" * 70 + "\n")
        
        all_pass = True
        
        # Required columns
        required = ['DESYNPUF_ID', 'BENE_BIRTH_DT', 'BENE_SEX_IDENT_CD', 
                   'BENE_RACE_CD', 'BENE_HI_CVRAGE_TOT_MONS']
        all_pass &= self.validate_columns_present(df, required, "Beneficiary")
        
        # Not null
        all_pass &= self.validate_not_null(
            df,
            ['DESYNPUF_ID', 'BENE_BIRTH_DT', 'BENE_SEX_IDENT_CD', 'BENE_RACE_CD'],
            "Beneficiary"
        )
        
        # Value ranges
        all_pass &= self.validate_value_range(
            df, 'BENE_HI_CVRAGE_TOT_MONS', min_value=0, max_value=12, table_name="Beneficiary"
        )
        all_pass &= self.validate_value_range(
            df, 'BENE_SMI_CVRAGE_TOT_MONS', min_value=0, max_value=12, table_name="Beneficiary"
        )
        
        # Categorical
        all_pass &= self.validate_enum_values(
            df, 'BENE_SEX_IDENT_CD', [1, 2], "Beneficiary"
        )
        all_pass &= self.validate_enum_values(
            df, 'BENE_RACE_CD', [1, 2, 3, 5], "Beneficiary"
        )
        
        # Primary key
        all_pass &= self.validate_primary_key(df, ['DESYNPUF_ID'], "Beneficiary")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"BENEFICIARY VALIDATION: {'PASS ✓' if all_pass else 'FAIL ✗'}")
        logger.info("=" * 70 + "\n")
        
        return all_pass
    
    # ========================================================================
    # INPATIENT CLAIMS DATA VALIDATION
    # ========================================================================
    
    def validate_inpatient_data(self, df: pd.DataFrame, feature_date: Optional[str] = None) -> bool:
        """Run all inpatient-specific validations"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATING INPATIENT CLAIMS DATA")
        logger.info("=" * 70 + "\n")
        
        all_pass = True
        
        # Required columns
        required = ['DESYNPUF_ID', 'CLM_ID', 'CLM_ADMSN_DT', 'NCH_BENE_DSCHRG_DT',
                   'CLM_PMT_AMT', 'CLM_UTLZTN_DAY_CNT']
        all_pass &= self.validate_columns_present(df, required, "Inpatient")
        
        # Not null
        all_pass &= self.validate_not_null(
            df,
            ['DESYNPUF_ID', 'CLM_ID', 'CLM_ADMSN_DT', 'NCH_BENE_DSCHRG_DT'],
            "Inpatient"
        )
        
        # Date ordering (admission <= discharge)
        all_pass &= self.validate_date_ordering(
            df, 'CLM_ADMSN_DT', 'NCH_BENE_DSCHRG_DT', "Inpatient"
        )
        
        # Positive LOS
        all_pass &= self.validate_value_range(
            df, 'CLM_UTLZTN_DAY_CNT', min_value=1, table_name="Inpatient"
        )
        
        # Leakage detection (if feature_date provided)
        if feature_date:
            all_pass &= self.validate_no_future_dates(
                df, 'CLM_ADMSN_DT', feature_date, "Inpatient"
            )
        
        logger.info("\n" + "=" * 70)
        logger.info(f"INPATIENT VALIDATION: {'PASS ✓' if all_pass else 'FAIL ✗'}")
        logger.info("=" * 70 + "\n")
        
        return all_pass
    
    # ========================================================================
    # FEATURE OUTPUT VALIDATION
    # ========================================================================
    
    def validate_features(self, df: pd.DataFrame, feature_date: str) -> bool:
        """Validate output feature dataframe"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATING FEATURE OUTPUT")
        logger.info("=" * 70 + "\n")
        
        all_pass = True
        
        # Required columns
        required = ['DESYNPUF_ID', 'feature_date', 'computed_at']
        all_pass &= self.validate_columns_present(df, required, "Features")
        
        # Non-null ID
        all_pass &= self.validate_not_null(df, ['DESYNPUF_ID'], "Features")
        
        # Age is positive
        if 'age_at_feature_date' in df.columns:
            all_pass &= self.validate_value_range(
                df, 'age_at_feature_date', min_value=0, max_value=150, table_name="Features"
            )
        
        # No negative costs (except possibly CLM_PMT_AMT)
        cost_cols = [c for c in df.columns if 'cost' in c.lower() or 'liability' in c.lower()]
        for col in cost_cols:
            all_pass &= self.validate_value_range(
                df, col, min_value=0, table_name="Features"
            )
        
        logger.info("\n" + "=" * 70)
        logger.info(f"FEATURE VALIDATION: {'PASS ✓' if all_pass else 'FAIL ✗'}")
        logger.info("=" * 70 + "\n")
        
        return all_pass


def main():
    """Example usage"""
    validator = SchemaValidator(raise_on_error=False)
    logger.info("Schema Validator initialized. Example:")
    logger.info("  validator = SchemaValidator()")
    logger.info("  validator.validate_beneficiary_data(bene_df)")
    logger.info("  validator.validate_inpatient_data(claims_df, feature_date='2009-06-30')")


if __name__ == "__main__":
    main()
