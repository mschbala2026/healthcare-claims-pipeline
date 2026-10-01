"""
Feature Definitions for CMS Inpatient Feature Pipeline

This module defines all features that the pipeline will compute.
Each feature has clear documentation about:
  - What it measures
  - How it's computed
  - Whether it can leak future information
  - Point-in-time considerations
  - Data sources
"""

from dataclasses import dataclass
from typing import Optional
from datetime import timedelta


@dataclass
class FeatureDefinition:
    """Metadata about a single feature"""
    
    name: str
    """Feature name (used in output dataframe)"""
    
    description: str
    """Human-readable description of what this feature measures"""
    
    data_source: str
    """Table this feature comes from (e.g., 'inpatient', 'beneficiary')"""
    
    aggregation_window_days: Optional[int]
    """How many days back to aggregate (None=point-in-time, all-time aggregate)"""
    
    is_point_in_time_safe: bool
    """True if feature uses only data available BEFORE feature_date"""
    
    example_value: Optional[str]
    """Example output value"""
    
    leakage_notes: str
    """Notes on potential data leakage risks and mitigations"""


# ============================================================================
# BENEFICIARY-LEVEL FEATURES (From Beneficiary Summary File)
# ============================================================================
# These are STATIC features available at any point in time
# No point-in-time considerations needed

BENEFICIARY_FEATURES = [
    FeatureDefinition(
        name="age_at_feature_date",
        description="Patient age (in years) at the feature date, computed from birth date",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="78.5",
        leakage_notes="""
        SAFE: Derived from BENE_BIRTH_DT which is fixed and known at all times.
        No future information used. Age is deterministic from birth date.
        """
    ),
    
    FeatureDefinition(
        name="sex_male",
        description="Binary indicator: 1 if male, 0 if female",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="1",
        leakage_notes="""
        SAFE: Immutable demographic attribute. No future information.
        """
    ),
    
    FeatureDefinition(
        name="race_white",
        description="Binary indicator: 1 if white race, 0 otherwise",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="1",
        leakage_notes="""
        SAFE: Immutable demographic attribute. No future information.
        """
    ),
    
    FeatureDefinition(
        name="has_diabetes",
        description="Binary indicator: 1 if patient has diabetes diagnosis",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="1",
        leakage_notes="""
        SAFE: Chronic condition indicator calculated from claims up to feature_date.
        The CMS data provides this pre-computed, but we verify no future claims included.
        """
    ),
    
    FeatureDefinition(
        name="has_chf",
        description="Binary indicator: 1 if patient has congestive heart failure",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="0",
        leakage_notes="""
        SAFE: Chronic condition flag. Calculated from claims preceding feature date.
        """
    ),
    
    FeatureDefinition(
        name="has_ischemic_heart_disease",
        description="Binary indicator: 1 if patient has ischemic heart disease",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="1",
        leakage_notes="""
        SAFE: Chronic condition flag from historical claims.
        """
    ),
    
    FeatureDefinition(
        name="has_copd",
        description="Binary indicator: 1 if patient has COPD",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="0",
        leakage_notes="""
        SAFE: Chronic condition flag.
        """
    ),
    
    FeatureDefinition(
        name="has_chronic_kidney_disease",
        description="Binary indicator: 1 if patient has chronic kidney disease",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="0",
        leakage_notes="""
        SAFE: Chronic condition flag.
        """
    ),
    
    FeatureDefinition(
        name="has_stroke_or_tia",
        description="Binary indicator: 1 if patient has stroke or TIA",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="0",
        leakage_notes="""
        SAFE: Chronic condition flag.
        """
    ),
    
    FeatureDefinition(
        name="num_chronic_conditions",
        description="Count of chronic conditions (0-11)",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="3",
        leakage_notes="""
        SAFE: Simple sum of chronic condition flags.
        """
    ),
    
    FeatureDefinition(
        name="part_a_coverage_months",
        description="Number of months beneficiary had Part A (Hospital Insurance) coverage",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="12",
        leakage_notes="""
        SAFE: Static field for the year. Known at all times.
        """
    ),
    
    FeatureDefinition(
        name="part_b_coverage_months",
        description="Number of months beneficiary had Part B (Supplementary Medical) coverage",
        data_source="beneficiary",
        aggregation_window_days=None,
        is_point_in_time_safe=True,
        example_value="12",
        leakage_notes="""
        SAFE: Static field for the year.
        """
    ),
]


# ============================================================================
# INPATIENT CLAIMS FEATURES (From Inpatient Claims File)
# ============================================================================
# These are DYNAMIC features that aggregate over a time window
# CRITICAL: Must use only data strictly BEFORE feature_date

INPATIENT_FEATURES = [
    FeatureDefinition(
        name="num_inpatient_claims_prior_90d",
        description="Number of inpatient claims in the 90 days before feature date",
        data_source="inpatient",
        aggregation_window_days=90,
        is_point_in_time_safe=True,
        example_value="2",
        leakage_notes="""
        SAFE: Uses CLM_ADMSN_DT strictly < feature_date. Only counts claims submitted before.
        Window: [feature_date - 90 days, feature_date)
        """
    ),
    
    FeatureDefinition(
        name="num_inpatient_claims_prior_365d",
        description="Number of inpatient claims in the 365 days (1 year) before feature date",
        data_source="inpatient",
        aggregation_window_days=365,
        is_point_in_time_safe=True,
        example_value="3",
        leakage_notes="""
        SAFE: All data strictly before feature_date.
        """
    ),
    
    FeatureDefinition(
        name="total_inpatient_cost_prior_90d",
        description="Total Medicare payment for inpatient claims in 90 days before feature date (dollars)",
        data_source="inpatient",
        aggregation_window_days=90,
        is_point_in_time_safe=True,
        example_value="5000.50",
        leakage_notes="""
        SAFE: Uses CLM_ADMSN_DT < feature_date and CLM_PMT_AMT aggregation.
        Only sums costs from claims that were submitted before feature_date.
        """
    ),
    
    FeatureDefinition(
        name="total_inpatient_cost_prior_365d",
        description="Total Medicare payment for inpatient claims in 365 days before feature date (dollars)",
        data_source="inpatient",
        aggregation_window_days=365,
        is_point_in_time_safe=True,
        example_value="12500.00",
        leakage_notes="""
        SAFE: Year-long window, all before feature date.
        """
    ),
    
    FeatureDefinition(
        name="avg_length_of_stay_prior_90d",
        description="Average length of stay (days) for inpatient admissions in prior 90 days",
        data_source="inpatient",
        aggregation_window_days=90,
        is_point_in_time_safe=True,
        example_value="4.5",
        leakage_notes="""
        SAFE: Computed from CLM_ADMSN_DT < feature_date using CLM_UTLZTN_DAY_CNT.
        LOS is fully known upon admission, not future information.
        """
    ),
    
    FeatureDefinition(
        name="max_length_of_stay_prior_90d",
        description="Maximum length of stay (days) for any inpatient admission in prior 90 days",
        data_source="inpatient",
        aggregation_window_days=90,
        is_point_in_time_safe=True,
        example_value="14",
        leakage_notes="""
        SAFE: Maximum of CLM_UTLZTN_DAY_CNT for claims before feature_date.
        """
    ),
    
    FeatureDefinition(
        name="has_inpatient_claim_last_30d",
        description="Binary indicator: 1 if patient had any inpatient admission in last 30 days",
        data_source="inpatient",
        aggregation_window_days=30,
        is_point_in_time_safe=True,
        example_value="1",
        leakage_notes="""
        SAFE: Boolean indicator of claims strictly before feature_date.
        Window: [feature_date - 30 days, feature_date)
        """
    ),
    
    FeatureDefinition(
        name="has_diagnosis_diabetes",
        description="Binary: 1 if any inpatient claim in prior 90d has diabetes diagnosis (ICD9 250.xx)",
        data_source="inpatient",
        aggregation_window_days=90,
        is_point_in_time_safe=True,
        example_value="1",
        leakage_notes="""
        SAFE: Searches diagnosis codes (ICD9_DGNS_CD_*) in claims before feature_date.
        Diagnoses must be from claims that occurred before feature_date.
        """
    ),
    
    FeatureDefinition(
        name="has_diagnosis_sepsis",
        description="Binary: 1 if any inpatient claim in prior 90d has sepsis diagnosis (ICD9 995.9x)",
        data_source="inpatient",
        aggregation_window_days=90,
        is_point_in_time_safe=True,
        example_value="0",
        leakage_notes="""
        SAFE: Diagnosis codes from prior claims.
        """
    ),
    
    FeatureDefinition(
        name="median_drg_severity",
        description="Median DRG (Diagnosis Related Group) code among inpatient admissions prior 90d",
        data_source="inpatient",
        aggregation_window_days=90,
        is_point_in_time_safe=True,
        example_value="291",
        leakage_notes="""
        SAFE: DRG codes assigned at admission, known before discharge.
        Uses CLM_DRG_CD from prior claims.
        NOTE: DRG is for billing classification, not patient severity.
        """
    ),
    
    FeatureDefinition(
        name="inpatient_cost_per_claim_prior_90d",
        description="Average cost per inpatient claim in prior 90 days (total_cost / num_claims)",
        data_source="inpatient",
        aggregation_window_days=90,
        is_point_in_time_safe=True,
        example_value="2500.00",
        leakage_notes="""
        SAFE: Simple ratio of prior costs and counts.
        Handles division by zero (returns 0 if no claims).
        """
    ),
    
    FeatureDefinition(
        name="beneficiary_liability_prior_90d",
        description="Total beneficiary responsibility (deductible + coinsurance) prior 90 days (dollars)",
        data_source="inpatient",
        aggregation_window_days=90,
        is_point_in_time_safe=True,
        example_value="250.00",
        leakage_notes="""
        SAFE: Sums NCH_BENE_IP_DDCTBL_AMT, NCH_BENE_PTA_COINSRNC_LBLTY_AM from prior claims.
        Amount is determined at claim submission, not future.
        """
    ),
]


# ============================================================================
# TARGET VARIABLE DEFINITIONS
# ============================================================================

@dataclass
class TargetDefinition:
    """Metadata about the target variable (what we're predicting)"""
    
    name: str
    """Target name"""
    
    description: str
    """What this target measures"""
    
    prediction_window_days: int
    """How many days after feature_date to look for the event"""
    
    label_creation_logic: str
    """SQL/pseudocode showing how to create the label"""
    
    leakage_prevention: str
    """How we prevent leaking future information"""
    
    example_positive_case: str
    """Example scenario where target = 1"""
    
    example_negative_case: str
    """Example scenario where target = 0"""


TARGET_VARIABLES = [
    TargetDefinition(
        name="high_inpatient_cost_next_90d",
        description="Binary target: 1 if patient has inpatient costs >= $10,000 in next 90 days",
        prediction_window_days=90,
        label_creation_logic="""
        feature_date = DATE('2009-06-30')
        label_date_start = feature_date + 1 day = '2009-07-01'
        label_date_end = feature_date + 90 days = '2009-09-28'
        
        target = 1 IF (
            SUM(CLM_PMT_AMT) >= 10000.00 
            WHERE CLM_ADMSN_DT >= label_date_start 
              AND CLM_ADMSN_DT <= label_date_end
        )
        target = 0 OTHERWISE
        """,
        leakage_prevention="""
        CRITICAL: Use CLM_ADMSN_DT, not CLM_FROM_DT, for date filtering.
        Ensure label window is STRICTLY AFTER feature_date:
        - feature_date: 2009-06-30
        - label_window: 2009-07-01 to 2009-09-28
        
        Never use data where CLM_ADMSN_DT <= feature_date for the label.
        Never look at costs after label_date_end.
        """,
        example_positive_case="""
        Patient admitted 2009-07-15 for pneumonia, 14 days stay, cost $12,000 -> TARGET = 1
        """,
        example_negative_case="""
        Patient has no admissions from 2009-07-01 to 2009-09-28 -> TARGET = 0
        OR patient admitted but total cost only $5,000 -> TARGET = 0
        """
    ),
    
    TargetDefinition(
        name="readmission_within_30d",
        description="Binary target: 1 if patient has inpatient admission within 30 days of prior discharge",
        prediction_window_days=30,
        label_creation_logic="""
        For each inpatient claim at feature_date:
            discharge_date = NCH_BENE_DSCHRG_DT
            readmission_window_start = discharge_date + 1 day
            readmission_window_end = discharge_date + 30 days
        
        target = 1 IF EXISTS (
            SELECT 1 FROM inpatient 
            WHERE CLM_ADMSN_DT >= readmission_window_start 
              AND CLM_ADMSN_DT <= readmission_window_end
        )
        target = 0 OTHERWISE
        """,
        leakage_prevention="""
        Key: Use actual discharge date (NCH_BENE_DSCHRG_DT) from prior admission.
        Only look for readmissions in the 30 days following discharge.
        Never use information after the 30-day window.
        
        For patients with no prior inpatient admission, set TARGET = NULL (or exclude).
        """,
        example_positive_case="""
        Discharge date: 2009-06-20
        Readmission date: 2009-06-25 (5 days after discharge) -> TARGET = 1
        """,
        example_negative_case="""
        Discharge date: 2009-06-20
        No readmission within 30 days -> TARGET = 0
        OR readmission on 2009-07-22 (32 days, outside window) -> TARGET = 0
        """
    ),
]


def get_all_features():
    """Return all feature definitions"""
    return BENEFICIARY_FEATURES + INPATIENT_FEATURES


def get_all_targets():
    """Return all target definitions"""
    return TARGET_VARIABLES


def print_feature_manifest():
    """Print summary of all features"""
    print("\n" + "=" * 80)
    print("FEATURE MANIFEST")
    print("=" * 80 + "\n")
    
    print(f"BENEFICIARY FEATURES ({len(BENEFICIARY_FEATURES)} features):")
    for feat in BENEFICIARY_FEATURES:
        print(f"  • {feat.name:40s} (aggregation: {feat.aggregation_window_days})")
    
    print(f"\nINPATIENT FEATURES ({len(INPATIENT_FEATURES)} features):")
    for feat in INPATIENT_FEATURES:
        print(f"  • {feat.name:40s} (window: {feat.aggregation_window_days} days)")
    
    print(f"\nTARGET VARIABLES ({len(TARGET_VARIABLES)} targets):")
    for tgt in TARGET_VARIABLES:
        print(f"  • {tgt.name:40s} (prediction: {tgt.prediction_window_days} days)")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {len(get_all_features())} features, {len(TARGET_VARIABLES)} targets")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_feature_manifest()
