# Online/Nearline Feature Serving Strategy

This document describes how the same feature semantics from historical batch computation can be served in real-time and nearline scenarios while maintaining consistency.

---

## Three Serving Modes

### Mode 1: Offline (Historical Batch Generation)

**Purpose:** Generate training data by computing features for all beneficiaries at fixed point-in-time

**Use Case:**
- Historical feature engineering for model training
- Reproducible, auditable feature sets
- Reference implementation

**Implementation:**
```python
from features.feature_generator import FeatureGenerator
import pandas as pd

# Load 2009-2010 claims data
bene_df = pd.read_parquet('gs://claims-bucket/beneficiary/2008-2010.parquet')
inpatient_df = pd.read_parquet('gs://claims-bucket/inpatient/2008-2010.parquet')

# Generate features for feature_date = 2009-06-30
# Uses ALL history available before this date
generator = FeatureGenerator(verbose=True)
features = generator.compute_features(
    beneficiary_df=bene_df,
    inpatient_df=inpatient_df,
    feature_date='2009-06-30',
    output_path='gs://features-bucket/historical/2009-06-30/features.parquet'
)
```

**Characteristics:**
- Latency: 2-3 days acceptable
- Throughput: Batch process, optimized for volume
- Freshness: Stale by design (uses fixed feature_date)
- Storage: Parquet files, versioned by feature_date
- Scalability: Distributed Spark/Dask for large datasets

**Advantages:**
- Complete history available
- Can recompute old features (audit trail)
- Foundation for training/validation splits
- Quality control possible (full data review)

---

### Mode 2: Nearline (Materialized View / Hourly Refresh)

**Purpose:** Pre-aggregate and cache features for fast inference updates (not real-time, but fresh within 1-2 hours)

**Use Case:**
- Hourly batch model inference
- Daily model retraining
- Monitoring and dashboards
- Report generation

**Implementation:**

```sql
-- Create materialized view of claims aggregations (refreshed hourly)
CREATE MATERIALIZED VIEW claims_features_90d AS
SELECT
    DESYNPUF_ID,
    COUNT(*) as num_claims_90d,
    SUM(CLM_PMT_AMT) as total_cost_90d,
    AVG(CLM_UTLZTN_DAY_CNT) as avg_los_90d,
    MAX(CLM_UTLZTN_DAY_CNT) as max_los_90d,
    CURRENT_DATE as feature_date  -- Data available up to yesterday
FROM inpatient_claims
WHERE CLM_ADMSN_DT >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
  AND CLM_ADMSN_DT < CURRENT_DATE()  -- CRITICAL: < not <=
GROUP BY DESYNPUF_ID
WITH DATA
REFRESH COMPLETE ON DEMAND;  -- Refresh daily at 2 AM

-- Combine with static demographics (cached)
SELECT
    b.DESYNPUF_ID,
    b.age_at_feature_date,
    b.sex_male,
    b.has_diabetes,
    c.num_claims_90d,
    c.total_cost_90d,
    c.avg_los_90d,
    c.max_los_90d,
    CURRENT_TIMESTAMP as computed_at
FROM beneficiary_cache b
LEFT JOIN claims_features_90d c USING (DESYNPUF_ID)
```

**Characteristics:**
- Latency: < 100ms (query materialized view)
- Throughput: High (pre-computed)
- Freshness: 1-2 hours old
- Storage: In-memory database (Redis, Cassandra)
- Scalability: Vertical (single machine) + replication

**Advantages:**
- Much faster than batch recomputation
- Consistent semantics with offline features
- Can serve inference requests in real-time
- Monitoring: Compare nearline vs offline features

**Trade-offs:**
- Freshness lag (data available up to yesterday)
- Storage overhead (pre-computed for all beneficiaries)
- Infrastructure complexity (materialized views, refresh)

---

### Mode 3: Online (Real-Time Inference API)

**Purpose:** Compute features on-demand for single beneficiary at prediction time

**Use Case:**
- Real-time risk scoring API
- Production inference serving
- Mobile/web applications
- Real-time dashboards

**Implementation (Python API):**

```python
# online_feature_server.py
from datetime import datetime, timedelta
import pandas as pd
from features.feature_generator import FeatureGenerator

class OnlineFeatureServer:
    """Serve features for single beneficiary at request time"""
    
    def __init__(self, db_connections):
        self.bene_db = db_connections['beneficiary']
        self.claims_db = db_connections['claims']
        self.generator = FeatureGenerator(verbose=False)
    
    def get_features(self, beneficiary_id: str, feature_date: str = None) -> dict:
        """
        Compute features for single beneficiary at feature_date.
        
        Args:
            beneficiary_id: DESYNPUF_ID
            feature_date: YYYY-MM-DD (defaults to today)
        
        Returns:
            Dictionary with 25 features
        
        Latency: ~100-500ms (depending on claims volume)
        """
        if feature_date is None:
            feature_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 1. Fetch beneficiary demographics (cached KV store, ~1ms)
        bene_row = self.bene_db.get(f'bene:{beneficiary_id}')
        if not bene_row:
            raise ValueError(f"Beneficiary {beneficiary_id} not found")
        
        bene_df = pd.DataFrame([bene_row])
        
        # 2. Fetch inpatient claims in prior 365 days (~50-200ms)
        feature_dt = pd.to_datetime(feature_date)
        cutoff_dt = feature_dt - timedelta(days=365)
        
        claims = self.claims_db.query(
            f"SELECT * FROM inpatient WHERE "
            f"DESYNPUF_ID = '{beneficiary_id}' AND "
            f"CLM_ADMSN_DT >= '{cutoff_dt.strftime('%Y%m%d')}' AND "
            f"CLM_ADMSN_DT < '{feature_dt.strftime('%Y%m%d')}'"
        )
        
        if len(claims) == 0:
            claims = pd.DataFrame()  # Empty dataframe
        else:
            claims = pd.DataFrame(claims)
        
        # 3. Compute features using exact same logic as offline (~50ms)
        features = self.generator.compute_features(
            beneficiary_df=bene_df,
            inpatient_df=claims,
            feature_date=feature_date
        )
        
        # 4. Return as dictionary (single row)
        return features.iloc[0].to_dict()


# API endpoint (FastAPI)
from fastapi import FastAPI

app = FastAPI()
server = OnlineFeatureServer(db_connections)

@app.get("/features/{beneficiary_id}")
def get_features(beneficiary_id: str, feature_date: str = None):
    """
    Real-time feature API.
    
    Example:
        GET /features/BENE001?feature_date=2009-06-30
        Response: {
            "age_at_feature_date": 74.1,
            "sex_male": 1,
            "has_diabetes": 1,
            "num_claims_90d": 3,
            "total_cost_90d": 8500.0,
            ...
        }
    """
    try:
        features = server.get_features(beneficiary_id, feature_date)
        return {"status": "success", "features": features}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500
```

**Characteristics:**
- Latency: 100-500ms (varies by data volume)
- Throughput: Medium (1-10 requests/sec per instance)
- Freshness: Real-time (queried current state)
- Storage: No pre-computation
- Scalability: Horizontal (load balance across instances)

**Advantages:**
- True real-time features
- No storage overhead
- Flexible feature_date (can compute historical features)
- No cache invalidation

**Trade-offs:**
- Higher latency than nearline
- Database load (one query per request)
- Slower queries for beneficiaries with many claims
- Network overhead

---

## Feature Parity Strategy

**Critical:** All three modes use identical feature computation logic

### 1. Shared Feature Code

**All modes call same functions:**
```python
# features/feature_generator.py (used by offline, nearline, online)

def _apply_date_filter(self, df, date_column, feature_date, max_days_back=None):
    """Strict temporal boundary: dates < feature_date"""
    dates = pd.to_datetime(df[date_column], format='%Y%m%d', errors='coerce')
    mask = dates < pd.to_datetime(feature_date)  # CRITICAL: < not <=
    if max_days_back is not None:
        cutoff_dt = pd.to_datetime(feature_date) - timedelta(days=max_days_back)
        mask = mask & (dates >= cutoff_dt)
    return df[mask].copy()

def _compute_inpatient_features(self, claims_df, feature_date):
    """30d/90d/365d aggregations - used by all modes"""
    # Same code in offline, nearline, online
```

### 2. Validation & Monitoring

**Test that all modes produce identical features:**

```python
def test_offline_nearline_online_parity():
    """Verify offline, nearline, online modes produce identical features"""
    
    beneficiary_id = 'BENE001'
    feature_date = '2009-06-30'
    
    # Offline: compute from full history
    offline_features = offline_generator.compute_features(
        beneficiary_df=all_beneficiaries,
        inpatient_df=all_claims,
        feature_date=feature_date
    ).set_index('DESYNPUF_ID').loc[beneficiary_id]
    
    # Nearline: query materialized view
    nearline_features = nearline_server.query(beneficiary_id, feature_date)
    
    # Online: real-time computation
    online_features = online_server.get_features(beneficiary_id, feature_date)
    
    # All should be identical
    assert_series_equal(offline_features[['num_claims_90d', 'total_cost_90d', ...]],
                       pd.Series(nearline_features[['num_claims_90d', 'total_cost_90d', ...]]))
    assert_series_equal(offline_features[['num_claims_90d', 'total_cost_90d', ...]],
                       pd.Series(online_features[['num_claims_90d', 'total_cost_90d', ...]]))
```

### 3. Data Contract Enforcement

**All modes validate same contracts:**
```
beneficiary_contract.yaml
inpatient_contract.yaml
```

---

## Production Deployment

### Architecture Diagram

```
Training (Offline)
    └─> Batch feature generation (2009-06-30)
        └─> Store in S3/GCS (parquet)
            └─> Feed to training pipeline
                └─> Train model (knows features at T)

Inference (Nearline)
    └─> Hourly batch job
        └─> Compute features for TODAY as feature_date
            └─> Store in Redis (cached)
                └─> Serve to batch scoring API
                    └─> Update model predictions hourly

Inference (Online)
    └─> API request arrives
        └─> Query claims DB for prior 365 days
            └─> Compute features in-memory
                └─> Return to caller (100-500ms)
                    └─> Real-time model scoring
```

### Consistency Guarantees

| Guarantee | How Achieved |
|-----------|-------------|
| Same temporal boundaries | All use `dates < feature_date` |
| Same aggregation windows | Shared 30d/90d/365d window code |
| Same feature output | All call `_compute_inpatient_features()` |
| Same data contracts | YAML schemas enforced at input |
| Same quality checks | SchemaValidator used in all modes |
| Reproducibility | feature_date is immutable reference |

---

## Failure Modes & Recovery

### Nearline Failure: Materialized View Out of Date
- **Symptom:** Features are stale (>2 hours old)
- **Detection:** Check `last_refresh_time` in metadata
- **Recovery:** Fall back to online mode (slower but fresh)

### Online Failure: Database Unavailable
- **Symptom:** Feature computation times out
- **Detection:** Catch exception, log error
- **Recovery:** Fall back to cached nearline features (stale but available)

### Schema Mismatch: Offline vs Online
- **Symptom:** Offline feature = 3, Online feature = 5 (for same beneficiary)
- **Detection:** Parity test fails
- **Recovery:** Roll back online code, investigate schema drift

---

## SLA & Monitoring

```
Offline:  2-3 day latency, 99.9% availability
Nearline: <100ms latency, 99% availability, <2 hour staleness
Online:   <500ms latency, 95% availability, real-time freshness

Monitoring:
  - Offline: Feature generation duration, data quality metrics
  - Nearline: Cache hit rate, feature staleness, query latency
  - Online: API latency (p50, p95, p99), database query time, error rate
  - Parity: Offline vs Nearline vs Online feature correlation (should be >0.999)
```

