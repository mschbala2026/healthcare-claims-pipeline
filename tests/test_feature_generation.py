"""
Test Suite for Feature Pipeline

Tests verify:
  1. Point-in-time correctness (no future data leakage)
  2. Data quality and schema compliance
  3. Feature correctness (expected aggregation logic)
  4. Edge cases and boundary conditions
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import feature pipeline modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_generator import FeatureGenerator
from validation.schema_validator import SchemaValidator


# ============================================================================
# FIXTURES: Create test data
# ============================================================================

@pytest.fixture
def simple_beneficiary_data():
    """Create minimal test beneficiary data"""
    return pd.DataFrame({
        'DESYNPUF_ID': ['BENE001', 'BENE002', 'BENE003'],
        'BENE_BIRTH_DT': ['19350501', '19420715', '19550310'],
        'BENE_DEATH_DT': [None, None, None],
        'BENE_SEX_IDENT_CD': [1, 2, 1],  # 1=male, 2=female
        'BENE_RACE_CD': [1, 2, 1],  # 1=white, 2=black
        'BENE_ESRD_IND': ['0', '0', 'Y'],
        'SP_STATE_CODE': ['26', '39', '06'],
        'BENE_COUNTY_CD': ['950', '230', '850'],
        'BENE_HI_CVRAGE_TOT_MONS': [12, 12, 6],
        'BENE_SMI_CVRAGE_TOT_MONS': [12, 12, 0],
        'BENE_HMO_CVRAGE_TOT_MONS': [0, 0, 0],
        'PLAN_CVRG_MOS_NUM': ['12', '12', '00'],
        'SP_DIABETES': [1, 2, 1],
        'SP_CHF': [2, 1, 2],
        'SP_ISCHMCHT': [1, 1, 2],
        'SP_COPD': [2, 2, 1],
        'SP_CHRNKIDN': [2, 2, 2],
        'SP_STRKETIA': [2, 2, 2],
        'SP_DEPRESSN': [1, 2, 2],
        'SP_CNCR': [2, 2, 2],
        'SP_OSTEOPRS': [2, 2, 2],
        'SP_RA_OA': [2, 2, 2],
        'SP_ALZHDMTA': [2, 2, 2],
        'MEDREIMB_IP': [2500.0, 0.0, 5000.0],
        'BENRES_IP': [250.0, 0.0, 500.0],
        'PPPYMT_IP': [0.0, 0.0, 0.0],
        'MEDREIMB_OP': [625.0, 500.0, 1000.0],
        'BENRES_OP': [125.0, 100.0, 200.0],
        'PPPYMT_OP': [0.0, 0.0, 0.0],
        'MEDREIMB_CAR': [1000.0, 800.0, 1200.0],
        'BENRES_CAR': [200.0, 160.0, 240.0],
        'PPPYMT_CAR': [0.0, 0.0, 0.0],
    })


@pytest.fixture
def simple_inpatient_data():
    """Create minimal test inpatient claims data"""
    return pd.DataFrame({
        'DESYNPUF_ID': ['BENE001', 'BENE001', 'BENE002', 'BENE003'],
        'CLM_ID': ['CLM001', 'CLM002', 'CLM003', 'CLM004'],
        'SEGMENT': [1, 1, 1, 1],
        'PRVDR_NUM': ['260001', '260001', '260002', '260003'],
        'CLM_FROM_DT': ['20090601', '20090615', '20090620', '20090605'],
        'CLM_THRU_DT': ['20090605', '20090620', '20090625', '20090610'],
        'CLM_ADMSN_DT': ['20090601', '20090615', '20090620', '20090605'],
        'NCH_BENE_DSCHRG_DT': ['20090605', '20090620', '20090625', '20090610'],
        'CLM_PMT_AMT': [3000.0, 5500.0, 8000.0, 2500.0],
        'NCH_PRMRY_PYR_CLM_PD_AMT': [0.0, 0.0, 0.0, 0.0],
        'CLM_PASS_THRU_PER_DIEM_AMT': [0.0, 0.0, 0.0, 0.0],
        'NCH_BENE_IP_DDCTBL_AMT': [1000.0, 0.0, 1000.0, 500.0],
        'NCH_BENE_PTA_COINSRNC_LBLTY_AM': [200.0, 200.0, 300.0, 100.0],
        'NCH_BENE_BLOOD_DDCTBL_LBLTY_AM': [0.0, 0.0, 0.0, 0.0],
        'CLM_UTLZTN_DAY_CNT': [5, 6, 6, 6],
        'CLM_DRG_CD': ['291', '291', '292', '280'],
        'ADMTNG_ICD9_DGNS_CD': ['42789', '42789', '25000', '41400'],
        'ICD9_DGNS_CD_1': ['42789', '42789', '25000', '41400'],
        'ICD9_DGNS_CD_2': ['', '25000', '', 'E11'],
        'ICD9_DGNS_CD_3': ['', '', '', ''],
        'ICD9_DGNS_CD_4': ['', '', '', ''],
        'ICD9_DGNS_CD_5': ['', '', '', ''],
        'ICD9_DGNS_CD_6': ['', '', '', ''],
        'ICD9_DGNIS_CD_7': ['', '', '', ''],
        'ICD9_DGNS_CD_8': ['', '', '', ''],
        'ICD9_DGNS_CD_9': ['', '', '', ''],
        'ICD9_DGNS_CD_10': ['', '', '', ''],
        'ICD9_PRCDR_CD_1': ['3721', '3721', '', '3722'],
        'ICD9_PRCDR_CD_2': ['', '', '', ''],
        'ICD9_PRCDR_CD_3': ['', '', '', ''],
        'ICD9_PRCDR_CD_4': ['', '', '', ''],
        'ICD9_PRCDR_CD_5': ['', '', '', ''],
        'ICD9_PRCDR_CD_6': ['', '', '', ''],
        'HCPCS_CD_1': ['', '', '', ''],
        'AT_PHYSN_NPI': ['1234567890', '1234567890', '1111111111', '2222222222'],
        'OP_PHYSN_NPI': ['', '', '', ''],
        'OT_PHYSN_NPI': ['', '', '', ''],
    })


@pytest.fixture
def feature_generator():
    """Create feature generator instance"""
    return FeatureGenerator(verbose=False)


@pytest.fixture
def schema_validator():
    """Create schema validator instance"""
    return SchemaValidator(raise_on_error=False)


# ============================================================================
# TEST SUITE 1: SCHEMA VALIDATION
# ============================================================================

class TestSchemaValidation:
    """Test data contract validation"""
    
    def test_beneficiary_required_columns(self, simple_beneficiary_data, schema_validator):
        """Verify required columns are present"""
        required = ['DESYNPUF_ID', 'BENE_BIRTH_DT', 'BENE_SEX_IDENT_CD']
        assert schema_validator.validate_columns_present(
            simple_beneficiary_data, required, "Beneficiary"
        )
    
    def test_beneficiary_missing_column_fails(self, simple_beneficiary_data, schema_validator):
        """Verify validation fails when column missing"""
        df = simple_beneficiary_data.drop('BENE_BIRTH_DT', axis=1)
        assert not schema_validator.validate_columns_present(
            df, ['BENE_BIRTH_DT'], "Beneficiary"
        )
    
    def test_no_nulls_in_required(self, simple_beneficiary_data, schema_validator):
        """Verify no NULLs in required columns"""
        assert schema_validator.validate_not_null(
            simple_beneficiary_data,
            ['DESYNPUF_ID', 'BENE_BIRTH_DT'],
            "Beneficiary"
        )
    
    def test_nulls_detection_fails(self, simple_beneficiary_data, schema_validator):
        """Verify NULL detection catches NULLs"""
        df = simple_beneficiary_data.copy()
        df.loc[0, 'DESYNPUF_ID'] = None
        assert not schema_validator.validate_not_null(
            df, ['DESYNPUF_ID'], "Beneficiary"
        )
    
    def test_value_range_validation(self, simple_beneficiary_data, schema_validator):
        """Verify value range checking"""
        assert schema_validator.validate_value_range(
            simple_beneficiary_data,
            'BENE_HI_CVRAGE_TOT_MONS',
            min_value=0,
            max_value=12,
            table_name="Beneficiary"
        )
    
    def test_enum_validation(self, simple_beneficiary_data, schema_validator):
        """Verify categorical value checking"""
        assert schema_validator.validate_enum_values(
            simple_beneficiary_data,
            'BENE_SEX_IDENT_CD',
            [1, 2],
            "Beneficiary"
        )
    
    def test_enum_validation_fails(self, simple_beneficiary_data, schema_validator):
        """Verify enum validation catches invalid values"""
        df = simple_beneficiary_data.copy()
        df.loc[0, 'BENE_SEX_IDENT_CD'] = 999
        assert not schema_validator.validate_enum_values(
            df, 'BENE_SEX_IDENT_CD', [1, 2], "Beneficiary"
        )
    
    def test_primary_key_uniqueness(self, simple_beneficiary_data, schema_validator):
        """Verify primary key uniqueness check"""
        assert schema_validator.validate_primary_key(
            simple_beneficiary_data,
            ['DESYNPUF_ID'],
            "Beneficiary"
        )
    
    def test_duplicate_key_fails(self, simple_beneficiary_data, schema_validator):
        """Verify duplicate key detection"""
        df = pd.concat([simple_beneficiary_data, simple_beneficiary_data.iloc[[0]]])
        assert not schema_validator.validate_primary_key(
            df, ['DESYNPUF_ID'], "Beneficiary"
        )


# ============================================================================
# TEST SUITE 2: POINT-IN-TIME CORRECTNESS & LEAKAGE DETECTION
# ============================================================================

class TestPointInTimeCorrectness:
    """Test that features use only historical data"""
    
    def test_no_future_dates_in_claims(self, simple_inpatient_data, schema_validator):
        """Verify leakage detection catches future dates"""
        # All test data is before 2009-06-30, so should pass
        assert schema_validator.validate_no_future_dates(
            simple_inpatient_data,
            'CLM_ADMSN_DT',
            '2009-06-30',
            "Inpatient"
        )
    
    def test_future_dates_detected(self, simple_inpatient_data, schema_validator):
        """Verify leakage is caught when dates are after feature_date"""
        df = simple_inpatient_data.copy()
        # Add a claim after feature_date
        future_claim = df.iloc[[0]].copy()
        future_claim['CLM_ADMSN_DT'] = '20090710'  # After 2009-06-30
        df = pd.concat([df, future_claim])
        
        # Should fail leakage detection
        assert not schema_validator.validate_no_future_dates(
            df, 'CLM_ADMSN_DT', '2009-06-30', "Inpatient"
        )
    
    def test_date_ordering_admission_before_discharge(self, simple_inpatient_data, schema_validator):
        """Verify admission date <= discharge date"""
        assert schema_validator.validate_date_ordering(
            simple_inpatient_data,
            'CLM_ADMSN_DT',
            'NCH_BENE_DSCHRG_DT',
            "Inpatient"
        )
    
    def test_date_ordering_fails_on_impossible_dates(self, simple_inpatient_data, schema_validator):
        """Verify impossible date combinations are caught"""
        df = simple_inpatient_data.copy()
        df.loc[0, 'CLM_ADMSN_DT'] = '20090610'
        df.loc[0, 'NCH_BENE_DSCHRG_DT'] = '20090605'  # Discharge before admission
        
        assert not schema_validator.validate_date_ordering(
            df, 'CLM_ADMSN_DT', 'NCH_BENE_DSCHRG_DT', "Inpatient"
        )


# ============================================================================
# TEST SUITE 3: FEATURE COMPUTATION CORRECTNESS
# ============================================================================

class TestFeatureComputation:
    """Test feature generation logic"""
    
    def test_beneficiary_features_created(self, simple_beneficiary_data, feature_generator):
        """Verify beneficiary features are computed"""
        features = feature_generator._compute_beneficiary_features(
            simple_beneficiary_data,
            feature_date='2009-06-30'
        )
        
        # Should have one row per beneficiary
        assert len(features) == 3
        assert 'DESYNPUF_ID' in features.columns
        assert 'age_at_feature_date' in features.columns
        assert 'sex_male' in features.columns
        assert 'has_diabetes' in features.columns
    
    def test_age_calculation(self, simple_beneficiary_data, feature_generator):
        """Verify age is calculated correctly"""
        features = feature_generator._compute_beneficiary_features(
            simple_beneficiary_data,
            feature_date='2009-06-30'
        )
        
        # BENE001: born 1935-05-01, feature date 2009-06-30 = ~74.1 years
        age_bene001 = features[features['DESYNPUF_ID'] == 'BENE001']['age_at_feature_date'].values[0]
        assert 74 < age_bene001 < 75  # Approximately 74 years old
    
    def test_inpatient_features_created(self, simple_beneficiary_data, simple_inpatient_data, 
                                        feature_generator):
        """Verify inpatient features are computed"""
        features = feature_generator._compute_inpatient_features(
            simple_inpatient_data,
            feature_date='2009-06-30'
        )
        
        # Should have one row per beneficiary
        assert len(features) >= 1
        assert 'DESYNPUF_ID' in features.columns
        assert 'num_claims_90d' in features.columns
        assert 'total_cost_90d' in features.columns
    
    def test_claim_aggregation_90d(self, simple_inpatient_data, feature_generator):
        """Verify claims are aggregated correctly for 90-day window"""
        features = feature_generator._compute_inpatient_features(
            simple_inpatient_data,
            feature_date='2009-06-30'
        )
        
        # BENE001: has 2 claims (06-01, 06-15) both within 90 days of 06-30
        bene001_features = features[features['DESYNPUF_ID'] == 'BENE001']
        assert bene001_features['num_claims_90d'].values[0] == 2
        # Total cost: 3000 + 5500 = 8500
        assert bene001_features['total_cost_90d'].values[0] == 8500.0
    
    def test_no_future_claims_in_features(self, feature_generator):
        """Verify future claims are NOT included (leakage prevention)"""
        # Create data with a claim after feature date
        future_bene = pd.DataFrame({
            'DESYNPUF_ID': ['BENE001', 'BENE001'],
            'CLM_ID': ['CLM001', 'CLM999'],
            'SEGMENT': [1, 1],
            'PRVDR_NUM': ['260001', '260001'],
            'CLM_FROM_DT': ['20090601', '20090701'],
            'CLM_THRU_DT': ['20090605', '20090705'],
            'CLM_ADMSN_DT': ['20090601', '20090701'],  # Second claim AFTER 06-30
            'NCH_BENE_DSCHRG_DT': ['20090605', '20090705'],
            'CLM_PMT_AMT': [3000.0, 999999.0],  # Huge future cost
            'NCH_PRMRY_PYR_CLM_PD_AMT': [0.0, 0.0],
            'CLM_PASS_THRU_PER_DIEM_AMT': [0.0, 0.0],
            'NCH_BENE_IP_DDCTBL_AMT': [0.0, 0.0],
            'NCH_BENE_PTA_COINSRNC_LBLTY_AM': [0.0, 0.0],
            'NCH_BENE_BLOOD_DDCTBL_LBLTY_AM': [0.0, 0.0],
            'CLM_UTLZTN_DAY_CNT': [5, 5],
            'CLM_DRG_CD': ['291', '291'],
            'ADMTNG_ICD9_DGNS_CD': ['42789', '25000'],
            'ICD9_DGNS_CD_1': ['42789', '25000'],
            'ICD9_DGNS_CD_2': ['', ''],
            'ICD9_DGNS_CD_3': ['', ''],
            'ICD9_DGNS_CD_4': ['', ''],
            'ICD9_DGNS_CD_5': ['', ''],
            'ICD9_DGNS_CD_6': ['', ''],
            'ICD9_DGNIS_CD_7': ['', ''],
            'ICD9_DGNS_CD_8': ['', ''],
            'ICD9_DGNS_CD_9': ['', ''],
            'ICD9_DGNS_CD_10': ['', ''],
            'ICD9_PRCDR_CD_1': ['', ''],
            'ICD9_PRCDR_CD_2': ['', ''],
            'ICD9_PRCDR_CD_3': ['', ''],
            'ICD9_PRCDR_CD_4': ['', ''],
            'ICD9_PRCDR_CD_5': ['', ''],
            'ICD9_PRCDR_CD_6': ['', ''],
            'HCPCS_CD_1': ['', ''],
            'AT_PHYSN_NPI': ['', ''],
            'OP_PHYSN_NPI': ['', ''],
            'OT_PHYSN_NPI': ['', ''],
        })
        
        features = feature_generator._compute_inpatient_features(
            future_bene,
            feature_date='2009-06-30'
        )
        
        # Should only see 1 claim (the 06-01 one), NOT the 07-01 claim
        bene001_features = features[features['DESYNPUF_ID'] == 'BENE001']
        assert bene001_features['num_claims_90d'].values[0] == 1
        assert bene001_features['total_cost_90d'].values[0] == 3000.0  # Not 999999


# ============================================================================
# TEST SUITE 4: INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test full feature pipeline"""
    
    def test_full_pipeline(self, simple_beneficiary_data, simple_inpatient_data, 
                          feature_generator):
        """Test complete feature generation"""
        features = feature_generator.compute_features(
            beneficiary_df=simple_beneficiary_data,
            inpatient_df=simple_inpatient_data,
            feature_date='2009-06-30'
        )
        
        # Should have beneficiary count rows
        assert len(features) == 3
        
        # Should have metadata columns
        assert 'feature_date' in features.columns
        assert 'computed_at' in features.columns
        
        # Should have demographic features
        assert 'age_at_feature_date' in features.columns
        assert 'sex_male' in features.columns
        
        # Should have claims aggregations
        assert 'num_claims_90d' in features.columns
        assert 'total_cost_90d' in features.columns
    
    def test_feature_reproducibility(self, simple_beneficiary_data, simple_inpatient_data, 
                                    feature_generator):
        """Test that same inputs produce same features (idempotency)"""
        features1 = feature_generator.compute_features(
            beneficiary_df=simple_beneficiary_data.copy(),
            inpatient_df=simple_inpatient_data.copy(),
            feature_date='2009-06-30'
        )
        
        features2 = feature_generator.compute_features(
            beneficiary_df=simple_beneficiary_data.copy(),
            inpatient_df=simple_inpatient_data.copy(),
            feature_date='2009-06-30'
        )
        
        # Sort by DESYNPUF_ID to handle any ordering differences
        features1_sorted = features1.sort_values('DESYNPUF_ID').reset_index(drop=True)
        features2_sorted = features2.sort_values('DESYNPUF_ID').reset_index(drop=True)
        
        # Compare (ignore computed_at which changes)
        pd.testing.assert_frame_equal(
            features1_sorted.drop('computed_at', axis=1),
            features2_sorted.drop('computed_at', axis=1)
        )


# ============================================================================
# TEST SUITE 5: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_beneficiary_data(self, feature_generator):
        """Test with empty beneficiary dataframe"""
        empty_bene = pd.DataFrame({
            'DESYNPUF_ID': [],
            'BENE_BIRTH_DT': [],
            'BENE_DEATH_DT': [],
            'BENE_SEX_IDENT_CD': [],
            'BENE_RACE_CD': [],
        })
        empty_claims = empty_bene.copy()
        
        features = feature_generator.compute_features(
            beneficiary_df=empty_bene,
            inpatient_df=empty_claims,
            feature_date='2009-06-30'
        )
        
        assert len(features) == 0
    
    def test_beneficiary_with_no_claims(self, simple_beneficiary_data, feature_generator):
        """Test beneficiary with zero claims"""
        # Only include claims for BENE001
        one_claim = pd.DataFrame({
            'DESYNPUF_ID': ['BENE001'],
            'CLM_ID': ['CLM001'],
            'CLM_ADMSN_DT': ['20090601'],
            'NCH_BENE_DSCHRG_DT': ['20090605'],
            'CLM_PMT_AMT': [3000.0],
            'CLM_UTLZTN_DAY_CNT': [5],
            'SEGMENT': [1],
            'PRVDR_NUM': ['260001'],
            'CLM_FROM_DT': ['20090601'],
            'CLM_THRU_DT': ['20090605'],
            'NCH_PRMRY_PYR_CLM_PD_AMT': [0.0],
            'CLM_PASS_THRU_PER_DIEM_AMT': [0.0],
            'NCH_BENE_IP_DDCTBL_AMT': [0.0],
            'NCH_BENE_PTA_COINSRNC_LBLTY_AM': [0.0],
            'NCH_BENE_BLOOD_DDCTBL_LBLTY_AM': [0.0],
            'CLM_DRG_CD': ['291'],
            'ADMTNG_ICD9_DGNS_CD': ['42789'],
            'ICD9_DGNS_CD_1': ['42789'],
            'ICD9_DGNS_CD_2': [''],
            'ICD9_DGNS_CD_3': [''],
            'ICD9_DGNS_CD_4': [''],
            'ICD9_DGNS_CD_5': [''],
            'ICD9_DGNS_CD_6': [''],
            'ICD9_DGNIS_CD_7': [''],
            'ICD9_DGNS_CD_8': [''],
            'ICD9_DGNS_CD_9': [''],
            'ICD9_DGNS_CD_10': [''],
            'ICD9_PRCDR_CD_1': [''],
            'ICD9_PRCDR_CD_2': [''],
            'ICD9_PRCDR_CD_3': [''],
            'ICD9_PRCDR_CD_4': [''],
            'ICD9_PRCDR_CD_5': [''],
            'ICD9_PRCDR_CD_6': [''],
            'HCPCS_CD_1': [''],
            'AT_PHYSN_NPI': [''],
            'OP_PHYSN_NPI': [''],
            'OT_PHYSN_NPI': [''],
        })
        
        features = feature_generator.compute_features(
            beneficiary_df=simple_beneficiary_data,
            inpatient_df=one_claim,
            feature_date='2009-06-30'
        )
        
        # All 3 beneficiaries should be in output
        assert len(features) == 3
        
        # BENE002 & BENE003 should have 0 claims
        bene002_claims = features[features['DESYNPUF_ID'] == 'BENE002']['num_claims_90d'].values[0]
        assert bene002_claims == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
