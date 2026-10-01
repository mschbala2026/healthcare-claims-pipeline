# Design Rationale, Assumptions & Improvements

## Design Rationale

### Why CMS DE-SynPUF 2008-2010?
- Public synthetic data (no privacy concerns)
- Realistic claims structure (multiple tables)
- Sufficient history for features (18 months)
- Sufficient label window (7 months forward)

### Why feature_date = 2009-06-30?
- Middle of 3-year dataset
- Balances history depth vs label availability
- All 90-day label windows fit in available data
- Prevents accidental future data usage

### Why 25 Features?
- Demographics: stable, don't change (12 features)
- Utilization: time-windowed aggregations (12 features)
- Diagnoses: pattern-matched ICD9 codes (1 feature)
- Covers prediction task: high cost + readmission

---

## Key Assumptions

### Data Assumptions
1. **No data corrections after feature_date** - assumes administrative finality
2. **Claims processed within 90 days** - assumes service date ≈ admission date
3. **Beneficiary demographics stable** - doesn't capture mid-year enrollment changes
4. **No beneficiary migration** - DESYNPUF_ID unique per person

### Temporal Assumptions
1. **Entity time = CLM_ADMSN_DT** - not submission/processed date
2. **Feature date is hard boundary** - dates < feature_date ONLY
3. **Label window = 7 months** - limited by 2010-12-31 data end
4. **No data lag** - synthetic data; real data would have 90-180 day lag

### Feature Assumptions
1. **Beneficiary-level aggregation** - assumes comparable beneficiaries
2. **No feature leakage by design** - physical data boundary prevents accidents
3. **Missing values = 0** - chronic conditions default to "no"
4. **Costs non-negative** - assumes no credits/reversals in sample

---

## What I Would Improve With More Time

### 1. Feature Store Integration (Priority: HIGH)
- Currently: batch offline features only
- Improvement: Redis/DynamoDB cache for online inference
- Benefit: Sub-100ms serving latency for real-time predictions
- Time: 3-4 hours

### 2. Schema Versioning (Priority: HIGH)
- Currently: single schema version (1.0.0)
- Improvement: automated compatibility checks on schema changes
- Benefit: prevent breaking downstream consumers
- Time: 2 hours

### 3. Incremental Feature Generation (Priority: MEDIUM)
- Currently: regenerate all features on each run
- Improvement: compute only new/modified claims
- Benefit: 10x faster on large datasets
- Time: 4 hours

### 4. Monitoring & Alerting (Priority: MEDIUM)
- Currently: no data quality monitoring
- Improvement: anomaly detection on feature distributions
- Benefit: catch data issues before ML model sees them
- Time: 3 hours

### 5. Alternative Dates Analysis (Priority: LOW)
- Currently: only 2009-06-30 tested
- Improvement: sweep feature_date across 2008-2010, measure label availability/history tradeoff
- Benefit: data-driven feature_date selection
- Time: 2 hours

### 6. Advanced Diagnoses Features (Priority: LOW)
- Currently: simple binary flags (diabetes, CHF, etc.)
- Improvement: ICD9 hierarchy embeddings, comorbidity indices (Charlson, Elixhauser)
- Benefit: richer diagnosis representations
- Time: 4 hours

---

## Production Readiness Checklist

✅ **Implemented:**
- [x] Schema validation with contracts
- [x] Point-in-time correctness (leakage prevention)
- [x] Unit tests (22 passing)
- [x] Integration tests (end-to-end pipeline)
- [x] Error handling (validation failures stop pipeline)
- [x] Logging (detailed execution traces)
- [x] Documentation (design + code)
- [x] Reproducibility (fixed random seed, deterministic aggregations)

⚠️ **Not Implemented (scope/time):**
- [ ] Feature store (batch features only)
- [ ] Data freshness monitoring
- [ ] Automated retraining pipeline
- [ ] A/B testing framework
- [ ] Feature importance tracking
- [ ] Model serving layer (only data generation)