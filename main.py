#!/usr/bin/env python3
"""
Main Entry Point: CMS Feature Pipeline

This script demonstrates the complete feature generation workflow:
  1. Load raw data from CSV
  2. Validate schema compliance
  3. Generate features for a specific feature_date
  4. Validate output and save to parquet

Usage:
    python main.py --beneficiary data/raw/bene.csv \\
                   --inpatient data/raw/claims.csv \\
                   --feature-date 2009-06-30 \\
                   --output data/processed/features_2009-06-30.parquet

    # Or with sample data (for testing):
    python main.py --beneficiary data/raw/bene.csv \\
                   --inpatient data/raw/claims.csv \\
                   --feature-date 2009-06-30 \\
                   --sample 10000
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from features.feature_generator import FeatureGenerator, load_cms_data
from validation.schema_validator import SchemaValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline orchestration"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="CMS Inpatient Feature Pipeline",
        epilog="For more info, see README.md"
    )
    parser.add_argument(
        '--beneficiary',
        type=str,
        required=True,
        help='Path to beneficiary summary CSV'
    )
    parser.add_argument(
        '--inpatient',
        type=str,
        required=True,
        help='Path to inpatient claims CSV'
    )
    parser.add_argument(
        '--feature-date',
        type=str,
        required=True,
        help='Feature date (YYYY-MM-DD or YYYYMMDD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for features parquet (optional)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample N rows from each input (for testing)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip schema validation (not recommended)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose logging (default: True)'
    )
    
    args = parser.parse_args()
    
    # ==== STEP 1: LOAD DATA ====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOAD DATA")
    logger.info("=" * 70)
    
    try:
        bene_df, inp_df = load_cms_data(
            beneficiary_path=args.beneficiary,
            inpatient_path=args.inpatient,
            sample_n=args.sample
        )
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # ==== STEP 2: VALIDATE SCHEMA ====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: VALIDATE INPUT SCHEMA")
    logger.info("=" * 70)
    
    validator = SchemaValidator(raise_on_error=True)
    
    try:
        if not args.skip_validation:
            assert validator.validate_beneficiary_data(bene_df), \
                "Beneficiary data validation failed"
            assert validator.validate_inpatient_data(inp_df, args.feature_date), \
                "Inpatient data validation failed"
            logger.info("\n✓ All input validations passed")
        else:
            logger.warning("Skipping schema validation (not recommended)")
    except AssertionError as e:
        logger.error(f"\nValidation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    
    # ==== STEP 3: GENERATE FEATURES ====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: GENERATE FEATURES")
    logger.info("=" * 70)
    
    gen = FeatureGenerator(verbose=args.verbose)
    
    try:
        features = gen.compute_features(
            beneficiary_df=bene_df,
            inpatient_df=inp_df,
            feature_date=args.feature_date,
            output_path=args.output if args.output else None
        )
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        sys.exit(1)
    
    # ==== STEP 4: VALIDATE OUTPUT ====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: VALIDATE OUTPUT")
    logger.info("=" * 70)
    
    try:
        assert validator.validate_features(features, args.feature_date), \
            "Output feature validation failed"
        logger.info("\n✓ Output validation passed")
    except AssertionError as e:
        logger.error(f"\nOutput validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    
    # ==== STEP 5: SUMMARY & NEXT STEPS ====
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE ✓")
    logger.info("=" * 70)
    
    logger.info(f"\nSummary:")
    logger.info(f"  Input beneficiaries: {len(bene_df):,}")
    logger.info(f"  Input inpatient claims: {len(inp_df):,}")
    logger.info(f"  Output feature sets: {len(features):,}")
    logger.info(f"  Output columns: {len(features.columns)}")
    logger.info(f"  Feature date: {args.feature_date}")
    
    # Show sample features
    logger.info(f"\nSample features (first row):")
    if len(features) > 0:
        for col in features.columns[:10]:
            val = features.iloc[0][col]
            logger.info(f"  {col:30s} = {val}")
    
    if args.output:
        logger.info(f"\n✓ Features saved to: {args.output}")
        logger.info(f"  File size: {Path(args.output).stat().st_size / 1e6:.1f} MB")
    
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review features with pandas: pd.read_parquet('{args.output}')")
    logger.info(f"  2. Create labels for supervised learning")
    logger.info(f"  3. Run test suite: pytest tests/ -v")
    logger.info(f"  4. See README.md for complete documentation")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
