"""
Feature Generation Engine for CMS Inpatient Prediction

This module implements point-in-time correct feature generation
with explicit leakage prevention for production deployment.

Key principles:
  1. All features use data strictly BEFORE feature_date
  2. Date filtering uses feature_date as exclusive upper bound
  3. Tests verify no future data leakage
  4. Idempotent: same feature_date always produces same features
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    Generates point-in-time correct features from CMS data.
    
    Usage:
        gen = FeatureGenerator()
        features = gen.compute_features(
            beneficiary_df=bene_df,
            inpatient_df=claims_df,
            feature_date='2009-06-30',
            output_path='features.parquet'
        )
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = logger
    
    def _log(self, msg: str):
        if self.verbose:
            self.logger.info(msg)
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def _parse_date(self, date_str: str) -> pd.Timestamp:
        """Parse date in YYYYMMDD or YYYY-MM-DD format"""
        if isinstance(date_str, pd.Timestamp):
            return date_str
        
        date_str = str(date_str).strip()
        
        # Remove hyphens if present, then parse
        clean_date = date_str.replace('-', '')
        
        # If it's 8 digits, parse as YYYYMMDD
        if len(clean_date) == 8:
            try:
                return pd.to_datetime(clean_date, format='%Y%m%d')
            except ValueError:
                pass
        
        # Otherwise, let pandas infer format
        return pd.to_datetime(date_str)
    
    def _validate_feature_date(self, df: pd.DataFrame, feature_date: str, 
                               date_column: str) -> None:
        """Validate that feature_date is not after max date in data"""
        feature_dt = self._parse_date(feature_date)
        max_date = pd.to_datetime(df[date_column], format='%Y%m%d', errors='coerce').max()
        
        if feature_dt > max_date:
            self._log(f"WARNING: feature_date {feature_date} is after max date in data: {max_date}")
    
    def _apply_date_filter(self, df: pd.DataFrame, date_column: str, 
                          feature_date: str, max_days_back: Optional[int] = None) -> pd.DataFrame:
        """
        Filter dataframe to only include rows BEFORE feature_date.
        
        Args:
            df: Input dataframe
            date_column: Column name with dates in YYYYMMDD format
            feature_date: Cutoff date (exclusive)
            max_days_back: Optional window in days (e.g., 90 for last 90 days)
        
        Returns:
            Filtered dataframe with date_column < feature_date
        """
        feature_dt = self._parse_date(feature_date)
        dates = pd.to_datetime(df[date_column], format='%Y%m%d', errors='coerce')
        
        # CRITICAL: Use < (exclusive) not <= to prevent leaking feature_date itself
        mask = dates < feature_dt
        
        if max_days_back is not None:
            cutoff_dt = feature_dt - timedelta(days=max_days_back)
            mask = mask & (dates >= cutoff_dt)
        
        result = df[mask].copy()
        n_rows_before = len(df)
        n_rows_after = len(result)
        
        if self.verbose and n_rows_after < n_rows_before:
            pct = 100 * n_rows_after / n_rows_before if n_rows_before > 0 else 0
            self._log(f"  Date filter: {n_rows_before} → {n_rows_after} rows ({pct:.1f}%)")
        
        return result
    
    # ========================================================================
    # BENEFICIARY FEATURES (Static demographics)
    # ========================================================================
    
    def _compute_beneficiary_features(self, bene_df: pd.DataFrame, 
                                      feature_date: str) -> pd.DataFrame:
        """
        Compute all beneficiary-level features.
        
        SAFE: These use only immutable demographic data, no date filtering needed.
        """
        self._log("Computing beneficiary features...")
        
        # Handle empty beneficiary data
        if bene_df.empty or 'DESYNPUF_ID' not in bene_df.columns:
            self._log("  No beneficiary data")
            return pd.DataFrame(columns=['DESYNPUF_ID'])
        
        result = pd.DataFrame()
        result['DESYNPUF_ID'] = bene_df['DESYNPUF_ID']
        
        # Age
        bene_birth = pd.to_datetime(bene_df['BENE_BIRTH_DT'], format='%Y%m%d', errors='coerce')
        feature_dt = self._parse_date(feature_date)
        result['age_at_feature_date'] = (
            (feature_dt - bene_birth).dt.days / 365.25
        ).round(1)
        
        # Demographics
        result['sex_male'] = (bene_df['BENE_SEX_IDENT_CD'] == 1).astype(int) if 'BENE_SEX_IDENT_CD' in bene_df.columns else 0
        result['race_white'] = (bene_df['BENE_RACE_CD'] == 1).astype(int) if 'BENE_RACE_CD' in bene_df.columns else 0
        
        # Chronic conditions (1=Yes, 2=No in CMS data)
        # Handle missing columns gracefully - default to 0 (no condition) if missing
        result['has_diabetes'] = (bene_df['SP_DIABETES'] == 1).astype(int) if 'SP_DIABETES' in bene_df.columns else 0
        result['has_chf'] = (bene_df['SP_CHF'] == 1).astype(int) if 'SP_CHF' in bene_df.columns else 0
        result['has_ischemic_heart_disease'] = (bene_df['SP_ISCHMCHT'] == 1).astype(int) if 'SP_ISCHMCHT' in bene_df.columns else 0
        result['has_copd'] = (bene_df['SP_COPD'] == 1).astype(int) if 'SP_COPD' in bene_df.columns else 0
        result['has_chronic_kidney_disease'] = (bene_df['SP_CHRNKIDN'] == 1).astype(int) if 'SP_CHRNKIDN' in bene_df.columns else 0
        result['has_stroke_or_tia'] = (bene_df['SP_STRKETIA'] == 1).astype(int) if 'SP_STRKETIA' in bene_df.columns else 0
        result['has_depression'] = (bene_df['SP_DEPRESSN'] == 1).astype(int) if 'SP_DEPRESSN' in bene_df.columns else 0
        result['has_cancer'] = (bene_df['SP_CNCR'] == 1).astype(int) if 'SP_CNCR' in bene_df.columns else 0
        result['has_osteoporosis'] = (bene_df['SP_OSTEOPRS'] == 1).astype(int) if 'SP_OSTEOPRS' in bene_df.columns else 0
        result['has_ra_or_oa'] = (bene_df['SP_RA_OA'] == 1).astype(int) if 'SP_RA_OA' in bene_df.columns else 0
        result['has_alzheimer'] = (bene_df['SP_ALZHDMTA'] == 1).astype(int) if 'SP_ALZHDMTA' in bene_df.columns else 0
        
        # Count chronic conditions
        chronic_cols = ['has_diabetes', 'has_chf', 'has_ischemic_heart_disease', 
                       'has_copd', 'has_chronic_kidney_disease', 'has_stroke_or_tia',
                       'has_depression', 'has_cancer', 'has_osteoporosis', 
                       'has_ra_or_oa', 'has_alzheimer']
        result['num_chronic_conditions'] = result[chronic_cols].sum(axis=1)
        
        # Coverage months - handle missing columns
        result['part_a_coverage_months'] = bene_df['BENE_HI_CVRAGE_TOT_MONS'].astype(int) if 'BENE_HI_CVRAGE_TOT_MONS' in bene_df.columns else 12
        result['part_b_coverage_months'] = bene_df['BENE_SMI_CVRAGE_TOT_MONS'].astype(int) if 'BENE_SMI_CVRAGE_TOT_MONS' in bene_df.columns else 12
        
        self._log(f"  ✓ Computed {len(result)} beneficiary feature sets")
        return result
    
    # ========================================================================
    # INPATIENT CLAIMS FEATURES (Dynamic aggregations)
    # ========================================================================
    
    def _compute_inpatient_features(self, claims_df: pd.DataFrame, 
                                   feature_date: str) -> pd.DataFrame:
        """
        Compute all inpatient claims aggregations.
        
        CRITICAL: All aggregations use data strictly BEFORE feature_date.
        Window definitions:
          - prior_30d: [feature_date - 30 days, feature_date)
          - prior_90d: [feature_date - 90 days, feature_date)
          - prior_365d: [feature_date - 365 days, feature_date)
        """
        self._log("Computing inpatient claims features...")
        
        # Handle empty dataframe - return empty with proper columns for merge
        if claims_df.empty or 'CLM_ADMSN_DT' not in claims_df.columns:
            self._log("  No inpatient claims data")
            return pd.DataFrame(columns=['DESYNPUF_ID'])
        
        feature_dt = self._parse_date(feature_date)
        
        # Filter to only claims before feature_date
        claims_before_ft = self._apply_date_filter(
            claims_df, 
            date_column='CLM_ADMSN_DT',
            feature_date=feature_date
        )
        
        # Extract admission dates for windowing
        claims_before_ft = claims_before_ft.copy()
        
        # If no claims before feature_date, return empty result with proper columns
        if claims_before_ft.empty:
            self._log("  No claims before feature date")
            return pd.DataFrame(columns=['DESYNPUF_ID'])
        
        adm_dates = pd.to_datetime(
            claims_before_ft['CLM_ADMSN_DT'], 
            format='%Y%m%d', 
            errors='coerce'
        )
        
        # FEATURE 1: Claims in prior 90 days
        prior_90d_mask = adm_dates >= (feature_dt - timedelta(days=90))
        claims_90d = claims_before_ft[prior_90d_mask].copy()
        
        # FEATURE 2: Claims in prior 365 days
        prior_365d_mask = adm_dates >= (feature_dt - timedelta(days=365))
        claims_365d = claims_before_ft[prior_365d_mask].copy()
        
        # FEATURE 3: Claims in prior 30 days
        prior_30d_mask = adm_dates >= (feature_dt - timedelta(days=30))
        claims_30d = claims_before_ft[prior_30d_mask].copy()
        
        # Aggregate by beneficiary
        result = pd.DataFrame({'DESYNPUF_ID': claims_before_ft['DESYNPUF_ID'].unique()})
        
        # 90-day features
        grp_90d = claims_90d.groupby('DESYNPUF_ID').agg({
            'CLM_ID': 'count',
            'CLM_PMT_AMT': ['sum', 'mean'],
            'CLM_UTLZTN_DAY_CNT': ['mean', 'max']
        })
        grp_90d.columns = ['num_claims_90d', 'total_cost_90d', 'avg_cost_per_claim_90d',
                          'avg_los_90d', 'max_los_90d']
        grp_90d = grp_90d.reset_index()
        
        # 365-day features
        grp_365d = claims_365d.groupby('DESYNPUF_ID').agg({
            'CLM_ID': 'count',
            'CLM_PMT_AMT': 'sum'
        })
        grp_365d.columns = ['num_claims_365d', 'total_cost_365d']
        grp_365d = grp_365d.reset_index()
        
        # 30-day features
        has_30d = claims_30d[['DESYNPUF_ID']].drop_duplicates()
        has_30d['has_claim_30d'] = 1
        
        # Merge all
        result = result.merge(grp_90d, on='DESYNPUF_ID', how='left')
        result = result.merge(grp_365d, on='DESYNPUF_ID', how='left')
        result = result.merge(has_30d, on='DESYNPUF_ID', how='left')
        
        # Fill NaN with 0 for "no claims" scenarios
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)
        result['has_claim_30d'] = result['has_claim_30d'].fillna(0).astype(int)
        
        # Handle division by zero for averages
        result['avg_cost_per_claim_90d'] = result['avg_cost_per_claim_90d'].fillna(0)
        result['avg_los_90d'] = result['avg_los_90d'].fillna(0)
        
        # Beneficiary liability (deductible + coinsurance) - handle missing columns
        if 'NCH_BENE_IP_DDCTBL_AMT' in claims_90d.columns and 'NCH_BENE_PTA_COINSRNC_LBLTY_AM' in claims_90d.columns:
            liability = claims_90d.groupby('DESYNPUF_ID').agg({
                'NCH_BENE_IP_DDCTBL_AMT': 'sum',
                'NCH_BENE_PTA_COINSRNC_LBLTY_AM': 'sum'
            })
            liability['beneficiary_liability_90d'] = (
                liability['NCH_BENE_IP_DDCTBL_AMT'] + liability['NCH_BENE_PTA_COINSRNC_LBLTY_AM']
            )
            liability = liability[['beneficiary_liability_90d']].reset_index()
            result = result.merge(liability, on='DESYNPUF_ID', how='left')
        else:
            result['beneficiary_liability_90d'] = 0
        
        result['beneficiary_liability_90d'] = result['beneficiary_liability_90d'].fillna(0)
        
        self._log(f"  ✓ Computed {len(result)} inpatient feature sets")
        return result
    
    # ========================================================================
    # DIAGNOSIS FEATURES
    # ========================================================================
    
    def _extract_diagnosis_features(self, claims_df: pd.DataFrame, 
                                   feature_date: str) -> pd.DataFrame:
        """
        Extract diagnosis-based features from claims before feature_date.
        
        Diagnosis codes are in columns: ADMTNG_ICD9_DGNS_CD, ICD9_DGNS_CD_1-10
        """
        self._log("Computing diagnosis features...")
        
        # Handle empty dataframe or missing date column
        if claims_df.empty or 'CLM_ADMSN_DT' not in claims_df.columns:
            self._log("  No diagnosis data available")
            return pd.DataFrame(columns=['DESYNPUF_ID'])
        
        # Filter to claims before feature_date
        claims_before_ft = self._apply_date_filter(
            claims_df,
            date_column='CLM_ADMSN_DT',
            feature_date=feature_date,
            max_days_back=90
        )
        
        # If no claims, return empty with proper columns
        if claims_before_ft.empty:
            return pd.DataFrame(columns=['DESYNPUF_ID'])
        
        result = pd.DataFrame({'DESYNPUF_ID': claims_before_ft['DESYNPUF_ID'].unique()})
        
        # Diagnosis code columns
        diag_cols = ['ADMTNG_ICD9_DGNS_CD', 'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 
                     'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5',
                     'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 
                     'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10']
        
        # Helper function to check for diagnosis code pattern
        def has_diagnosis_pattern(df, pattern):
            """Check if any diagnosis code matches pattern (e.g., '250%' for diabetes)"""
            mask = pd.Series([False] * len(df), index=df.index)
            for col in diag_cols:
                if col in df.columns:
                    col_mask = df[col].astype(str).str.startswith(
                        pattern.rstrip('%'), na=False
                    )
                    mask = mask | col_mask
            return mask.astype(int)
        
        # Diabetes (ICD9: 250.xx)
        result['has_diabetes_90d'] = has_diagnosis_pattern(claims_before_ft, '250')
        
        # Sepsis (ICD9: 995.9x)
        result['has_sepsis_90d'] = has_diagnosis_pattern(claims_before_ft, '995.9')
        
        # Heart failure (ICD9: 428.xx)
        result['has_hf_90d'] = has_diagnosis_pattern(claims_before_ft, '428')
        
        # Pneumonia (ICD9: 480-486)
        result['has_pneumonia_90d'] = has_diagnosis_pattern(claims_before_ft, '48')
        
        # Aggregate to per-beneficiary
        result = result.groupby('DESYNPUF_ID')[
            ['has_diabetes_90d', 'has_sepsis_90d', 'has_hf_90d', 'has_pneumonia_90d']
        ].max().reset_index()  # max() converts boolean to 1/0
        
        self._log(f"  ✓ Computed {len(result)} diagnosis feature sets")
        return result
    
    # ========================================================================
    # MAIN ORCHESTRATION
    # ========================================================================
    
    def compute_features(self, 
                        beneficiary_df: pd.DataFrame,
                        inpatient_df: pd.DataFrame,
                        feature_date: str,
                        output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compute all features for a specific feature_date.
        
        Args:
            beneficiary_df: Beneficiary summary data (2.3M rows per year)
            inpatient_df: Inpatient claims data (500k+ rows per year)
            feature_date: Point-in-time date (YYYY-MM-DD or YYYYMMDD)
            output_path: Optional path to save features as parquet
        
        Returns:
            DataFrame with all features, one row per beneficiary
        
        Raises:
            ValueError: If data validation fails
        """
        feature_dt = self._parse_date(feature_date)
        self._log(f"\n{'=' * 70}")
        self._log(f"FEATURE COMPUTATION")
        self._log(f"Feature Date: {feature_date}")
        self._log(f"Input: {len(beneficiary_df):,} beneficiaries, "
                 f"{len(inpatient_df):,} claims")
        self._log(f"{'=' * 70}\n")
        
        # ====== COMPUTE BENEFICIARY FEATURES ======
        bene_features = self._compute_beneficiary_features(beneficiary_df, feature_date)
        
        # ====== COMPUTE INPATIENT FEATURES ======
        inp_features = self._compute_inpatient_features(inpatient_df, feature_date)
        
        # ====== COMPUTE DIAGNOSIS FEATURES ======
        diag_features = self._extract_diagnosis_features(inpatient_df, feature_date)
        
        # ====== MERGE ALL FEATURES ======
        self._log("Merging feature sets...")
        
        # Ensure all beneficiaries in output (using beneficiary file as source of truth)
        result = bene_features.copy()
        
        result = result.merge(inp_features, on='DESYNPUF_ID', how='left')
        result = result.merge(diag_features, on='DESYNPUF_ID', how='left')
        
        # Fill any remaining NaNs
        result = result.fillna(0)
        
        # Add metadata
        result['feature_date'] = feature_date
        result['computed_at'] = datetime.utcnow().isoformat()
        
        self._log(f"✓ Final feature set: {len(result):,} rows × {len(result.columns)} columns\n")
        
        # ====== SAVE OUTPUT ======
        if output_path:
            self._log(f"Saving features to: {output_path}")
            result.to_parquet(output_path, index=False)
            self._log(f"✓ Saved {len(result):,} rows")
        
        return result


def load_cms_data(beneficiary_path: str, inpatient_path: str, 
                 sample_n: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CMS data from CSV files.
    
    Args:
        beneficiary_path: Path to beneficiary summary CSV
        inpatient_path: Path to inpatient claims CSV
        sample_n: If set, load only first N rows from each file (for testing)
    
    Returns:
        (beneficiary_df, inpatient_df)
    """
    logger.info(f"Loading beneficiary data: {beneficiary_path}")
    bene_df = pd.read_csv(beneficiary_path, dtype={'DESYNPUF_ID': str})
    
    logger.info(f"Loading inpatient data: {inpatient_path}")
    inp_df = pd.read_csv(inpatient_path, dtype={'DESYNPUF_ID': str, 'CLM_ID': str})
    
    if sample_n:
        logger.info(f"Sampling to {sample_n} rows for testing")
        bene_df = bene_df.head(sample_n)
        inp_df = inp_df.head(sample_n)
    
    logger.info(f"Loaded {len(bene_df):,} beneficiaries, {len(inp_df):,} claims")
    
    return bene_df, inp_df


if __name__ == "__main__":
    # Example usage
    gen = FeatureGenerator(verbose=True)
    print("\nFeature generation engine ready. Import and use:")
    print("  from feature_generator import FeatureGenerator, load_cms_data")
    print("  gen = FeatureGenerator()")
    print("  bene_df, inp_df = load_cms_data('bene.csv', 'claims.csv')")
    print("  features = gen.compute_features(bene_df, inp_df, '2009-06-30')")
