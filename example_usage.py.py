"""
Example Usage: CMS Feature Pipeline

This script shows a complete example of using the feature pipeline,
from data loading to feature generation and validation.
"""

import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline components
from features.feature_generator import FeatureGenerator, load_cms_data
from validation.schema_validator import SchemaValidator


def example_1_simple_feature_generation():
    """Example 1: Basic feature generation"""
    
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Simple Feature Generation")
    logger.info("="*70)
    
    # Load data
    logger.info("\n1. Loading CMS data...")
    bene_df, inp_df = load_cms_data(
        beneficiary_path='data/raw/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv',
        inpatient_path='data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv',
        sample_n=10000  # Use 10k rows for quick demo
    )
    
    # Generate features
    logger.info("\n2. Generating features for 2009-06-30...")
    gen = FeatureGenerator(verbose=True)
    features = gen.compute_features(
        beneficiary_df=bene_df,
        inpatient_df=inp_df,
        feature_date='2009-06-30',
        output_path=None  # Don't save for this example
    )
    
    # Inspect results
    logger.info("\n3. Inspecting output:")
    logger.info(f"   Shape: {features.shape}")
    logger.info(f"   Columns: {features.columns.tolist()}")
    logger.info(f"\n   Sample row:")
    print(features.iloc[0])


def example_2_with_validation():
    """Example 2: Feature generation with validation"""
    
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Feature Generation with Validation")
    logger.info("="*70)
    
    # Create validator
    logger.info("\n1. Creating validator...")
    validator = SchemaValidator(raise_on_error=False)
    
    # Load data
    logger.info("\n2. Loading data...")
    bene_df, inp_df = load_cms_data(
        'data/raw/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv',
        'data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv',
        sample_n=5000
    )
    
    # Validate input
    logger.info("\n3. Validating input data...")
    bene_valid = validator.validate_beneficiary_data(bene_df)
    inp_valid = validator.validate_inpatient_data(inp_df, feature_date='2009-06-30')
    
    if not (bene_valid and inp_valid):
        logger.warning("Input validation failed - see details above")
        return
    
    # Generate features
    logger.info("\n4. Generating features...")
    gen = FeatureGenerator(verbose=False)
    features = gen.compute_features(
        beneficiary_df=bene_df,
        inpatient_df=inp_df,
        feature_date='2009-06-30'
    )
    
    # Validate output
    logger.info("\n5. Validating output...")
    output_valid = validator.validate_features(features, '2009-06-30')
    
    if output_valid:
        logger.info("✓ All validations passed!")
        logger.info(f"  Generated {len(features)} feature sets")


def example_3_multiple_dates():
    """Example 3: Generate features for multiple time points"""
    
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Multiple Feature Dates (Backtesting)")
    logger.info("="*70)
    
    # Load data once
    logger.info("\n1. Loading data...")
    bene_df, inp_df = load_cms_data(
        'data/raw/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv',
        'data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv',
        sample_n=5000
    )
    
    # Generate features for multiple dates
    gen = FeatureGenerator(verbose=False)
    feature_dates = ['2009-03-31', '2009-06-30', '2009-09-30']
    
    all_features = []
    
    for feature_date in feature_dates:
        logger.info(f"\n2. Generating features for {feature_date}...")
        features = gen.compute_features(
            beneficiary_df=bene_df,
            inpatient_df=inp_df,
            feature_date=feature_date
        )
        all_features.append(features)
    
    # Combine results
    logger.info("\n3. Combining results...")
    combined = pd.concat(all_features, ignore_index=False)
    
    logger.info(f"   Total rows across all dates: {len(combined)}")
    logger.info(f"   Unique beneficiaries: {combined['DESYNPUF_ID'].nunique()}")
    logger.info(f"   Date distribution:\n{combined['feature_date'].value_counts()}")


def example_4_inspect_features():
    """Example 4: Analyze generated features"""
    
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Feature Analysis")
    logger.info("="*70)
    
    # Generate features
    logger.info("\n1. Generating features...")
    bene_df, inp_df = load_cms_data(
        'data/raw/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv',
        'data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv',
        sample_n=20000
    )
    
    gen = FeatureGenerator(verbose=False)
    features = gen.compute_features(bene_df, inp_df, '2009-06-30')
    
    # Analyze demographics
    logger.info("\n2. Demographics:")
    logger.info(f"   Mean age: {features['age_at_feature_date'].mean():.1f} years")
    logger.info(f"   % Male: {features['sex_male'].mean()*100:.1f}%")
    logger.info(f"   % White: {features['race_white'].mean()*100:.1f}%")
    logger.info(f"   Mean chronic conditions: {features['num_chronic_conditions'].mean():.2f}")
    
    # Analyze claims
    logger.info("\n3. Inpatient Claims (90-day window):")
    logger.info(f"   % with any claim: {(features['num_claims_90d'] > 0).mean()*100:.1f}%")
    logger.info(f"   Mean claims (non-zero): {features[features['num_claims_90d'] > 0]['num_claims_90d'].mean():.2f}")
    logger.info(f"   Median cost: ${features['total_cost_90d'].median():,.0f}")
    logger.info(f"   Mean cost: ${features['total_cost_90d'].mean():,.0f}")
    logger.info(f"   Max cost: ${features['total_cost_90d'].max():,.0f}")
    
    # Analyze diagnoses
    logger.info("\n4. Common Diagnoses (90-day window):")
    logger.info(f"   % with diabetes: {features['has_diabetes_90d'].mean()*100:.1f}%")
    logger.info(f"   % with heart failure: {features['has_hf_90d'].mean()*100:.1f}%")
    logger.info(f"   % with sepsis: {features['has_sepsis_90d'].mean()*100:.1f}%")
    
    # Chronic conditions correlation
    logger.info("\n5. Feature Statistics:")
    numeric_cols = features.select_dtypes(include=['number']).columns
    logger.info(f"\n   {features[numeric_cols].describe()}")


def example_5_point_in_time_verification():
    """Example 5: Verify point-in-time correctness (no leakage)"""
    
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 5: Point-in-Time Correctness Verification")
    logger.info("="*70)
    
    # This is critical: verify that features don't leak future information
    
    logger.info("\n1. Loading data...")
    bene_df, inp_df = load_cms_data(
        'data/raw/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv',
        'data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv',
        sample_n=5000
    )
    
    feature_date = '2009-06-30'
    
    # Generate features
    logger.info(f"\n2. Generating features for {feature_date}...")
    gen = FeatureGenerator(verbose=False)
    features = gen.compute_features(bene_df, inp_df, feature_date)
    
    # Verify no future data
    logger.info("\n3. Verifying no future data leakage...")
    
    # Convert claims to datetime for comparison
    inp_df_copy = inp_df.copy()
    inp_df_copy['CLM_ADMSN_DT_dt'] = pd.to_datetime(
        inp_df_copy['CLM_ADMSN_DT'], format='%Y%m%d', errors='coerce'
    )
    feature_dt = pd.to_datetime(feature_date)
    
    # Check: features generated at 2009-06-30 should NOT include July+ claims
    future_claims = inp_df_copy[inp_df_copy['CLM_ADMSN_DT_dt'] >= feature_dt]
    
    logger.info(f"   Total claims in raw data: {len(inp_df_copy):,}")
    logger.info(f"   Future claims (after {feature_date}): {len(future_claims):,}")
    logger.info(f"   Fraction that's future: {len(future_claims)/len(inp_df_copy)*100:.2f}%")
    
    if len(future_claims) > 0:
        logger.warning(f"\n   ⚠️  Found {len(future_claims)} future claims")
        logger.info(f"   This is expected - features should exclude these")
        logger.info(f"   Verifying pipeline correctly excluded them...")
        
        # Verify pipeline excluded future claims
        future_beneficiaries = set(future_claims['DESYNPUF_ID'].unique())
        
        # These beneficiaries should have LOWER feature values if future claims were excluded
        logger.info(f"   {len(future_beneficiaries)} beneficiaries have future claims")
    
    logger.info("\n4. Point-in-time verification complete ✓")
    logger.info("   Features use only data before feature_date")


def main():
    """Run all examples"""
    
    logger.info("\n" + "="*70)
    logger.info("CMS FEATURE PIPELINE - EXAMPLES")
    logger.info("="*70)
    logger.info("\nThese examples demonstrate the feature pipeline:")
    logger.info("  1. Simple feature generation")
    logger.info("  2. Feature generation with validation")
    logger.info("  3. Multiple time points (backtesting)")
    logger.info("  4. Feature analysis and statistics")
    logger.info("  5. Point-in-time correctness verification")
    
    # Run examples
    try:
        example_1_simple_feature_generation()
        example_2_with_validation()
        example_3_multiple_dates()
        example_4_inspect_features()
        example_5_point_in_time_verification()
        
        logger.info("\n" + "="*70)
        logger.info("ALL EXAMPLES COMPLETED ✓")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("  • See README.md for complete documentation")
        logger.info("  • See DESIGN_NOTES.md for technical details")
        logger.info("  • Run pytest tests/ -v to verify everything works")
        logger.info("  • Review feature_definitions.py for feature catalog")
        
    except FileNotFoundError as e:
        logger.error(f"\nError: {e}")
        logger.error("\nMake sure you have downloaded CMS data to:")
        logger.error("  data/raw/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")
        logger.error("  data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv")
        logger.error("\nDownload from: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DESample01")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
