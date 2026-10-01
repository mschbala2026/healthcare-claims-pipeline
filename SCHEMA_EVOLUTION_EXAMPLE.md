# Schema Evolution Handling

This document demonstrates how the feature pipeline handles schema changes and breaking changes in upstream data.

---

## Scenario 1: Column Rename

**Change:** BENE_SEX_IDENT_CD → BENE_GENDER_CD

**Problem:**
- Old code expects `BENE_SEX_IDENT_CD`
- New data provides `BENE_GENDER_CD`
- Pipeline crashes with KeyError

**Solution (in feature_generator.py):**
```python
def _compute_beneficiary_features(self, bene_df, feature_date):
    # Handle column rename: BENE_SEX_IDENT_CD → BENE_GENDER_CD
    sex_col = 'BENE_GENDER_CD' if 'BENE_GENDER_CD' in bene_df.columns else 'BENE_SEX_IDENT_CD'
    result['sex_male'] = (bene_df[sex_col] == 1).astype(int) if sex_col in bene_df.columns else 0
```

**Feature Versioning:**
- Old pipelines produce: `sex_male` from `BENE_SEX_IDENT_CD`
- New pipelines produce: `sex_male` from `BENE_GENDER_CD` (semantically identical)
- Output feature name unchanged → downstream consumers unaffected
- Schema evolution is transparent at feature level

---

## Scenario 2: New Nullable Column

**Change:** New optional column `BENE_DIABETES_SEVERITY` added (0=none, 1=controlled, 2=uncontrolled)

**Problem:**
- Historical data (2008-2010) has NULL for all rows
- New data (2011+) has valid values
- Must not break historical feature generation

**Solution (in feature_generator.py):**
```python
# Old approach: direct access
result['has_diabetes'] = (bene_df['SP_DIABETES'] == 1).astype(int) if 'SP_DIABETES' in bene_df.columns else 0

# New approach: handle optional column with sensible default
result['diabetes_severity'] = bene_df.get('BENE_DIABETES_SEVERITY', 0).fillna(0).astype(int)
```

**Data Contract Update (contracts/beneficiary_contract.yaml):**
```yaml
# Old contract
columns:
  SP_DIABETES: [int, 1=yes 2=no]

# New contract (backward compatible)
columns:
  SP_DIABETES: [int, 1=yes 2=no, required]
  BENE_DIABETES_SEVERITY: [int, 0=none 1=controlled 2=uncontrolled, optional, new_2011]
```

**Pipeline Behavior:**
- Historical data (NULL values) → default to 0 (no severity tracked)
- New data (valid values) → use actual severity
- Same feature name, semantically compatible
- Tests should verify both historical and current data work

---

## Scenario 3: Type Change (Breaking)

**Change:** CLM_PMT_AMT changes from integer (cents) to decimal (dollars)

**Problem:**
- Old format: 300000 cents = $3,000
- New format: 3000.00 dollars = $3,000
- Features would scale 100x if not detected

**Impact on Features:**
- `total_cost_90d`, `total_cost_365d` values scale by 100x
- Models trained on old scale fail on new scale
- **Requires model retraining** and feature versioning

**Solution (in feature_generator.py):**
```python
def _compute_inpatient_features(self, claims_df, feature_date):
    """Handle cost column type changes"""
    
    # Detect data format by checking if most values > 10000 (likely cents)
    sample_costs = claims_df['CLM_PMT_AMT'].dropna().head(100)
    is_cents_format = (sample_costs > 10000).mean() > 0.8
    
    if is_cents_format:
        # Convert cents to dollars for consistency
        cost_multiplier = 0.01
    else:
        # Already in dollars
        cost_multiplier = 1.0
    
    # Aggregate with multiplier
    costs_by_bene = claims_90d.groupby('DESYNPUF_ID')['CLM_PMT_AMT'].sum() * cost_multiplier
```

**Feature Versioning:**
```
Feature Version 1.0: CLM_PMT_AMT in cents
  - total_cost_90d: $0 - $100,000
  - Models trained on this scale

Feature Version 1.1: CLM_PMT_AMT in dollars (type change detected)
  - Same feature name but different scale
  - Triggers automatic schema migration
  - Models must be retrained on new scale
  - Old and new models cannot be mixed
```

**Data Contract:**
```yaml
# Version 1.0
CLM_PMT_AMT: [integer, cents, scale_factor=0.01]

# Version 1.1 (breaking change)
CLM_PMT_AMT: [decimal, dollars, scale_factor=1.0]
```

---

## General Pattern: Defensive Column Access

**All features use defensive column access:**

```python
# ❌ Risky - crashes if column missing
result['has_diabetes'] = (bene_df['SP_DIABETES'] == 1).astype(int)

# ✅ Safe - handles renamed/missing columns
result['has_diabetes'] = (
    bene_df['SP_DIABETES'] == 1
).astype(int) if 'SP_DIABETES' in bene_df.columns else 0

# ✅ Also safe - handles rename mapping
sql_col = 'BENE_GENDER_CD' if 'BENE_GENDER_CD' in bene_df.columns else 'BENE_SEX_IDENT_CD'
result['sex_male'] = (bene_df[sql_col] == 1).astype(int)
```

---

## Testing Schema Evolution

**Add to test_feature_generation.py:**

```python
def test_schema_evolution_column_rename():
    """Verify pipeline handles renamed columns"""
    bene_df = simple_beneficiary_data()
    # Rename column
    bene_df = bene_df.rename(columns={'BENE_SEX_IDENT_CD': 'BENE_GENDER_CD'})
    
    # Pipeline should still work
    features = feature_generator.compute_features(
        beneficiary_df=bene_df,
        inpatient_df=simple_inpatient_data(),
        feature_date='2009-06-30'
    )
    
    # Feature should be computed despite rename
    assert 'sex_male' in features.columns
    assert not features['sex_male'].isna().all()


def test_schema_evolution_new_nullable_column():
    """Verify new optional columns don't break pipeline"""
    bene_df = simple_beneficiary_data()
    # Add new column with NULL values
    bene_df['BENE_DIABETES_SEVERITY'] = None
    
    features = feature_generator.compute_features(
        beneficiary_df=bene_df,
        inpatient_df=simple_inpatient_data(),
        feature_date='2009-06-30'
    )
    
    # Pipeline should handle NULL columns
    assert len(features) == len(bene_df)
```

---

## Versioning Strategy

**Files should include version tags:**
```python
# features/feature_generator.py
__version__ = "1.0.0"
FEATURES_SCHEMA_VERSION = "1.0"  # Changes when feature computation changes
DATA_CONTRACTS_VERSION = "1.0"   # Changes when input schema changes
```

**When to increment versions:**
- FEATURES_SCHEMA_VERSION: Any change to feature computation logic (requires model retrain)
- DATA_CONTRACTS_VERSION: Any change to input data contracts (may require mapping)

**On deployment:**
- Pin feature version in metadata
- Validate all consumers use compatible versions
- Prevent silent schema mismatches

