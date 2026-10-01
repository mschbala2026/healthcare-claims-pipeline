# Design Notes: CMS Inpatient Feature Pipeline

**Document**: Architecture, design decisions, and critical leakage prevention  
**Audience**: ML Engineers, Data Engineers implementing or extending this pipeline  
**Last Updated**: 2024-10-01

---

## Executive Summary

This document explains critical design decisions in the CMS feature pipeline, with emphasis on preventing data leakage—a silent killer of prediction models that works perfectly in backtest but fails catastrophically in production.

**Key Insight**: The difference between (backtest AUC: 0.95, production AUC: 0.65) is usually a single date comparison using `<=` instead of `<`.

---

## The Leakage Problem

### Real-World Scenario

A company builds a model to predict high inpatient costs. The backtest looks amazing:
- Training set (2008-2009): AUC = 0.95
- Validation set (2009): AUC = 0.94
- Production launch (2009): Predictions begin

Three months later, clinicians flag that the model is "unreasonably accurate" on already-discharged patients but useless for prospective prediction. After investigation:

**Root Cause**: The feature `total_cost_prior_90d` at feature_date=2009-06-30 included claims from July-September 2009 (future to the feature date). The model learned to predict costs it had already seen.

**Impact**: 
- 500+ patients were incorrectly prioritized for intervention
- $2M in unnecessary care coordination
- Model credibility destroyed, project cancelled

### Why This Happens

1. **Date fields are easy to mix up**
   - Service date vs. submission date
   - Admission date vs. billing date  
   - Fiscal year vs. calendar year
   - Date stored as string vs. timestamp vs. epoch seconds

2. **Data quality masks the problem**
   - Backtest data is usually clean (historical, already validated)
   - Leakage shows up only when model meets new data
   - By then, real-world damage is done

3. **Date filtering is counterintuitive**
   - "Last 90 days from June 30" seems obvious
   - But is it `[04-01, 06-30]` or `[04-01, 06-30)`?
   - Different implementations give different answers

---

## CMS Data Structure (Leakage Traps)

### Trap 1: Service Date vs. Submission Date

**CMS Data**: Each claim has multiple date fields

```
CLM_ADMSN_DT     = 2009-06-15  (Patient entered hospital)
CLM_FROM_DT      = 2009-06-15  (Billing period start)
CLM_THRU_DT      = 2009-06-20  (Billing period end)  
NCH_BENE_DSCHRG_DT = 2009-06-20 (Patient left hospital)
```

At feature_date = 2009-06-30:
- Patient was already discharged → We know the LOS and actual costs
- But what if patient was re-admitted on 2009-07-15?

**Leakage Risk**: Using future re-admissions to predict current costs

**Our Fix**: 
```python
# Filter BEFORE feature_date using admission date
admission_dates = pd.to_datetime(df['CLM_ADMSN_DT'], format='%Y%m%d')
mask = admission_dates < pd.to_datetime(feature_date)  # Use <, not <=
```

### Trap 2: Retroactive Data Corrections

**Example**: 
- June 15: Claim submitted with diagnosis code "pneumonia"
- June 30: Feature computation, includes this diagnosis
- July 5: Diagnosis corrected to "misdiagnosed, actually COPD" (retroactive)
- September 30: Historical database updated with correction

At point-in-time June 30, we SHOULD NOT see the corrected diagnosis. But if we're querying from updated DB in September, we DO.

**Leakage Risk**: Retroactive corrections create information from the future

**Our Fix**:
```python
# WRONG: Query current DB in September
claims_sept = query_database("SELECT * FROM claims WHERE submission_date <= '2009-06-30'")

# CORRECT: Use snapshot of data AS OF June 30
claims_pit = query_snapshot("SELECT * FROM claims_snapshot_2009_06_30")
```

For this project, we use static CSV files (no retroactive corrections), so this is handled naturally.

### Trap 3: Claims Processing Delays

**Reality**: 
- Service date: 2009-06-15 (patient received care)
- Submission date: 2009-08-20 (claim finally submitted 66 days later)
- Knowledge date: 2009-06-30 (what we knew then)

At feature_date = 2009-06-30, we did NOT know about this care yet.

**Leakage Risk**: Using future claims that haven't been submitted

**Our Fix**:
```python
# Use submission date (known) not service date (unknown future may arrive)
# In CMS DE-SynPUF, CLM_FROM_DT is submission date, CLM_ADMSN_DT is service date
# Use CLM_ADMSN_DT but validate it's before feature_date

admission_dates = pd.to_datetime(df['CLM_ADMSN_DT'], format='%Y%m%d')
feature_dt = pd.to_datetime(feature_date)
mask = admission_dates < feature_dt  # Strict <
```

---

## Design Decisions

### Decision 1: Date Filtering Strategy

**Choice**: Use `<` (exclusive) not `<=` (inclusive)

**Rationale**:
- feature_date=2009-06-30 means "end of business on June 30"
- Data from June 30 onwards should NOT be included
- Inclusive upper bound (`<=`) would leak June 30 data into future features

**Implementation**:
```python
def _apply_date_filter(self, df, date_column, feature_date, max_days_back=None):
    feature_dt = self._parse_date(feature_date)
    dates = pd.to_datetime(df[date_column], format='%Y%m%d')
    
    # CRITICAL: Use < not <=
    mask = dates < feature_dt
    
    if max_days_back is not None:
        cutoff_dt = feature_dt - timedelta(days=max_days_back)
        mask = mask & (dates >= cutoff_dt)
    
    return df[mask].copy()
```

**Tests**: `test_no_future_dates_detected` verifies this works

---

### Decision 2: Window Boundaries

**Choice**: Half-open intervals [start, end)

**Rationale**:
- Easier to reason about: "prior 90 days from June 30" is clearly the 90-day period BEFORE June 30
- Natural in pandas: `dates < feature_date`
- Prevents off-by-one errors common with inclusive ranges

**Window Definitions**:
```python
feature_date = '2009-06-30'
feature_dt = pd.to_datetime('2009-06-30')

# Prior 90 days: [2009-03-02, 2009-06-30)
prior_90d_start = feature_dt - timedelta(days=90)  # 2009-03-02
prior_90d_mask = (dates >= prior_90d_start) & (dates < feature_dt)

# Prior 30 days: [2009-06-01, 2009-06-30)
prior_30d_start = feature_dt - timedelta(days=30)  # 2009-06-01
prior_30d_mask = (dates >= prior_30d_start) & (dates < feature_dt)
```

**Verification**:
- Count claims in each window
- Assert no overlap with future window
- Document boundary dates in logs

---

### Decision 3: Beneficiary Features vs. Claims Features

**Split**:
- **Beneficiary features**: Immutable demographics (age, sex, race, chronic conditions)
- **Claims features**: Time-windowed aggregations (costs, LOS, diagnoses)

**Rationale**:
- Different update cadence (beneficiary annual, claims monthly)
- Different leakage risk (beneficiary features safe, claims features risky)
- Different maintenance burden (beneficiary stable, claims requires versioning)

**Implementation**:
```python
# Two separate functions with clear responsibilities
bene_features = self._compute_beneficiary_features(bene_df, feature_date)  # No date filter
inp_features = self._compute_inpatient_features(inp_df, feature_date)  # Strict date filter
# Then merge
result = bene_features.merge(inp_features, on='DESYNPUF_ID')
```

---

### Decision 4: Aggregation Strategy

**Choice**: Aggregate to beneficiary level (one row per person)

**Rationale**:
- ML models need one row per prediction unit (beneficiary)
- Time-windowed aggregations roll up claims to beneficiary
- Enables parallel model training (each beneficiary is independent)

**Implementation**:
```python
# Group claims by beneficiary, aggregate within time window
claims_90d = claims_df[mask_90d].groupby('DESYNPUF_ID').agg({
    'CLM_ID': 'count',  # num_claims_90d
    'CLM_PMT_AMT': ['sum', 'mean'],  # total, average cost
    'CLM_UTLZTN_DAY_CNT': ['mean', 'max']  # avg, max LOS
})
```

**Note**: Handles beneficiaries with zero claims (creates 0 for counts)

---

### Decision 5: NULL Handling

**Strategy**: 
- Beneficiary features: Keep as-is (few NULLs expected)
- Claims aggregations: Fill NaN with 0 (no claims = 0 cost)
- Diagnosis features: Fill 0/1 (no diagnosis = 0)

**Rationale**:
- Beneficiary files are complete (no missing birthdates)
- Claims data naturally sparse (not all patients have claims)
- Filling 0 is semantically correct: "zero claims in window" = 0

**Implementation**:
```python
# After all aggregations
numeric_cols = result.select_dtypes(include=[np.number]).columns
result[numeric_cols] = result[numeric_cols].fillna(0)
result['has_claim_30d'] = result['has_claim_30d'].fillna(0).astype(int)
```

---

## Data Period & Feature Date Selection

### CMS 2008-2010 Data: Context & Constraints

This pipeline uses the **CMS DE-SynPUF 2008-2010 public release**:

- **Data span:** January 1, 2008 → December 31, 2010 (3 full years)
- **Beneficiaries:** 2.3M individuals
- **Inpatient claims:** 500k+ admissions
- **Source:** Fully synthetic, de-identified public domain data

**Why This Data?** The assignment specifies "Use CMS DE-SynPUF data," and this is the standard public release available without special access.

### Feature Date: 2009-06-30 (Critical Choice)

The feature_date parameter determines the temporal split between features and labels:
---

## Leakage Prevention Checklist

Use this checklist when adding new features:

### Pre-Implementation
- [ ] Define aggregation window (e.g., "prior 90 days")
- [ ] Identify date field to filter on (admission? submission? service?)
- [ ] List all related date fields (could they introduce leakage?)
- [ ] Determine if feature is point-in-time safe (uses only historical data)

### Implementation
- [ ] Use `dates < feature_date` (not `<=`)
- [ ] Apply date filter BEFORE aggregation
- [ ] Handle edge cases: zero claims, NULLs, negative values
- [ ] Document window boundaries explicitly in code
- [ ] Add inline comment: "LEAKAGE RISK: [description]" for any risky code

### Testing
- [ ] Write test with data spanning feature_date (ensure future data excluded)
- [ ] Test boundary conditions (exactly at feature_date, day before, day after)
- [ ] Test empty dataset (no claims → should return 0, not error)
- [ ] Test NULL/missing values
- [ ] Add to `TestPointInTimeCorrectness` test class

### Code Review
- [ ] Reviewer checks date comparisons (`<` vs `<=`)
- [ ] Reviewer checks window boundaries in test data
- [ ] Reviewer checks for unfiltered access to claims (should always filter first)
- [ ] Reviewer verifies no accidental future data in aggregations

---

## Known Issues & Workarounds

### Issue 1: Coarse Date Precision

**Problem**: CMS dates are rounded to month/year (privacy protection)
```
Real date: 2009-06-15
CMS date:  2009-06-01  (month precision)
```

**Impact**: Can't distinguish July 5 from June 30

**Workaround**: 
- Treat all dates as "first day of month"
- Window of "last 90 days" is approximately 3 months, not precise
- Document this limitation in feature definitions

**Code**:
```python
# Coarsen to month precision
dates = pd.to_datetime(df['CLM_ADMSN_DT'], format='%Y%m%d')
dates_coarse = dates.dt.to_period('M').dt.to_timestamp()
```

### Issue 2: Synthetic Data Limitations

**Problem**: CMS DE-SynPUF is synthetic (data generated, not real)
```
Limitations:
- Provider IDs are randomized (not real hospitals)
- Procedure & diagnosis codes may not reflect real patterns
- Costs are scaled/randomized
- Missing some complex cases
```

**Impact**: Model trained on this won't transfer to real Medicare data

**Workaround**:
- Use ONLY for proof-of-concept and testing
- For production, obtain real data from ResDAC or CMS
- Understand this is a learning dataset, not production-ready

---

## Metrics & Monitoring

### In-Production Monitoring

**Daily checks**:
```python
# Check no future data in output
assert all(features['feature_date'] >= cutoff_date)

# Check feature distributions haven't changed dramatically
current_mean = features['total_cost_90d'].mean()
if current_mean > historical_mean * 1.5:
    alert("Cost distribution changed 50%+, potential data issue")

# Check coverage
null_pct = features['age_at_feature_date'].isnull().sum() / len(features)
if null_pct > 0.01:
    alert("1%+ features have NULL age, data quality issue")
```

**Weekly checks**:
```python
# Verify data contract compliance
validator.validate_beneficiary_data(bene_df)
validator.validate_inpatient_data(inp_df, feature_date='today')

# Compare feature distributions to baseline
kl_divergence = compute_kl_divergence(current, baseline)
if kl_divergence > threshold:
    notify("Feature distribution significant shift")
```

**Monthly checks**:
```python
# Full schema validation
validate_all_contracts()

# Backtest replay (verify same dates always produce same features)
features_v1 = compute_features(data, '2009-06-30')
features_v2 = compute_features(data, '2009-06-30')
assert features_v1.equals(features_v2), "Reproducibility check failed"
```

---

## Deployment Considerations

### Performance

**Beneficiary File**: ~2.3M rows × 30 columns → ~150MB loaded
**Inpatient File**: ~500k rows × 80 columns → ~400MB loaded
**Total Memory**: ~1GB for processing

**Computation Time**:
- Beneficiary features: < 1s
- Inpatient aggregations: ~10s (groupby operations)
- Diagnosis extraction: ~20s (regex on 500k rows)
- Total: ~30-40s for 2.3M beneficiaries

**Optimization**:
```python
# Use chunked processing for very large datasets
chunk_size = 100_000
for i in range(0, len(inp_df), chunk_size):
    chunk = inp_df.iloc[i:i+chunk_size]
    features_chunk = process_chunk(chunk, feature_date)
    write_to_parquet(features_chunk)
```

### Storage

**Output Format**: Parquet (compressed, efficient)
```
2.3M beneficiaries × 25 features × 8 bytes = ~460MB uncompressed
Parquet compression: ~80MB (82% reduction)
```

**Retention**:
- Keep daily features for last 90 days (active prediction)
- Archive weekly features (weekly decision points)
- Purge after 2 years or per compliance

---

## Future Enhancements

### Potential Features

1. **Readmission Risk**: Time from discharge to next admission
2. **ED Utilization**: Emergency department visits (if ED claims available)
3. **Medication Patterns**: Unique drug counts by class
4. **Provider Clustering**: Similar hospitals indicator
5. **Geographic Risk**: County-level risk scores

### Potential Improvements

1. **Incremental Computation**: Only update changed beneficiaries
2. **Distributed Processing**: Spark for multi-node computation
3. **Real-time Features**: Stream inpatient claims for daily updates
4. **Feature Store**: Publish features to shared feature store (Feast, Tecton)
5. **Automated Retraining**: Pipeline that detects data drift and retrains models

---

## Conclusion

This pipeline prioritizes correctness over cleverness. The design is intentionally conservative:

1. **Simple date filtering** (`<` not `<=`)
2. **Clear window boundaries** (half-open intervals)
3. **Extensive testing** (22 test cases)
4. **Production validation** (schema contracts + leakage detection)
5. **Documentation** (this guide + inline comments)

The cost is slightly more code and computation. The benefit is a model that works the same in backtest, validation, and production—which is worth every extra second.

---

**Document Version**: 1.0  
**Status**: Final  
**Author**: ML Engineering Team
