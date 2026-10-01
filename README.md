# Healthcare claims feature Pipeline

Production-ready feature engineering pipeline for Medicare inpatient claim prediction, emphasizing point-in-time correctness and leakage prevention.

**Status:** Ready for deployment  
**Last Updated:** 2024-10-01  
**Owner:** ML Engineering Team

---

## Overview

This pipeline generates historical features from CMS 2008-2010 DE-SynPUF synthetic Medicare data for predicting inpatient utilization and costs. The implementation prioritizes:

1. **Point-in-Time Correctness**: No future data leakage via strict date filtering
2. **Data Contracts**: Formal schema validation with change management
3. **Production Readiness**: Comprehensive testing and idempotent operations
4. **Maintainability**: Clear separation of concerns and documentation

### Key Features

- **Static Demographics**: Age, sex, race, chronic conditions (11 flags)
- **Dynamic Claims Aggregations**: 90-day and 365-day windows
- **Diagnosis-Based Features**: Common diagnoses from admission records
- **Rigorous Validation**: Schema compliance, leakage detection, data quality

### Dataset

- **Source**: CMS 2008-2010 DE-SynPUF (Sample 1, public synthetic data)
- **Beneficiaries**: ~2.3M per year
- **Inpatient Claims**: ~500k-548k per year
- **Time Range**: 2008-01-01 to 2010-12-31
- **Note**: Synthetic data for development; NOT suitable for real Medicare research

---

## Quick Start

### 1. Setup

```bash
# Clone repository  - in vscode open terminal and type below commands
git clone <https://github.com/mschbala2026/healthcare-claims-pipeline>
cd healthcare-claims-pipeline

# Note: Python should be installed on code running machine
# Create virtual environment 
python -m venv venv
#in windows type below command to activate virtual environment
venv\Scripts\activate
#in mac type below command to activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download from CMS public site:
- **Beneficiary Summary**: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DESample01
- **Inpatient Claims**: Same URL

Place files in `data/raw/`:
```
data/raw/
├── DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv
└── DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv
```

### 3. Generate Features

```python
from features.feature_generator import FeatureGenerator, load_cms_data

# Load data
bene_df, inp_df = load_cms_data(
    'data/raw/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv',
    'data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv'
)

# Generate features for point in time
gen = FeatureGenerator(verbose=True)
features = gen.compute_features(
    beneficiary_df=bene_df,
    inpatient_df=inp_df,
    feature_date='2009-06-30',
    output_path='data/processed/features_2009-06-30.parquet'
)

print(f"Generated {len(features)} feature vectors")
print(f"Columns: {features.columns.tolist()}")
```

### 4. Validate Data Quality

```python
from validation.schema_validator import SchemaValidator

validator = SchemaValidator(raise_on_error=False)

# Validate raw data
assert validator.validate_beneficiary_data(bene_df)
assert validator.validate_inpatient_data(inp_df, feature_date='2009-06-30')

# Validate output features
assert validator.validate_features(features, feature_date='2009-06-30')
```

### 5. Run Tests

```bash
# Run full test suite
pytest tests/ -v

# Run specific test class
pytest tests/test_feature_generation.py::TestPointInTimeCorrectness -v

# Run with coverage
pytest tests/ --cov=features --cov=validation --cov-report=html
```

---

## Architecture

### Directory Structure

```
feature_pipeline/
├── contracts/                    # Data contracts (YAML schemas)
│   ├── beneficiary_contract.yaml
│   └── inpatient_contract.yaml
│
├── features/                     # Feature generation logic
│   ├── feature_definitions.py   # Feature catalog & metadata
│   ├── feature_generator.py     # Main feature computation
│   └── __init__.py
│
├── validation/                   # Data quality & schema validation
│   ├── schema_validator.py      # Contract enforcement
│   └── __init__.py
│
├── tests/                        # Comprehensive test suite
│   ├── test_feature_generation.py
│   └── __init__.py
│
├── data/                         # Data directory
│   ├── raw/                     # Input CSV files (gitignored)
│   └── processed/               # Output features (gitignored)
│
├── DESIGN_NOTES.md              # Technical design & leakage analysis
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── .gitignore                    # Version control ignore rules
```

### Data Flow

```
Raw Data (CSV)
    ↓
Schema Validation (Contracts)
    ↓
Feature Generation
  ├─ Beneficiary Features (Static)
  ├─ Inpatient Features (90d, 365d windows)
  └─ Diagnosis Features (Pattern matching)
    ↓
Output Validation
    ↓
Features (Parquet)
```

---

## Data Contracts

### Beneficiary Summary Contract

**Location**: `contracts/beneficiary_contract.yaml`

**Coverage**:
- 32 columns including demographics, enrollment, costs
- 2.3M beneficiaries per year (2008, 2009, 2010)
- One row per beneficiary per year

**Key Validations**:
- Required: DESYNPUF_ID, BENE_BIRTH_DT, BENE_SEX_IDENT_CD, BENE_RACE_CD
- Sex: 1 (male) or 2 (female)
- Race: 1 (white), 2 (black), 3 (other), 5 (hispanic)
- Coverage months: 0-12 per month field
- Chronic conditions: 1 (yes) or 2 (no)
- Costs: non-negative dollars
- No duplicate beneficiary IDs

### Inpatient Claims Contract

**Location**: `contracts/inpatient_contract.yaml`

**Coverage**:
- 81 columns including dates, diagnoses, procedures, costs
- ~500k-548k claims per year
- 1-2 segments per claim

**Key Validations**:
- Required: DESYNPUF_ID, CLM_ID, CLM_ADMSN_DT, NCH_BENE_DSCHRG_DT
- Date ordering: CLM_ADMSN_DT ≤ NCH_BENE_DSCHRG_DT
- LOS (CLM_UTLZTN_DAY_CNT): ≥ 1 day
- No future dates relative to feature_date
- All dates in format YYYYMMDD
- Costs can be negative (for reversals/adjustments)

---

## Features Generated

### Beneficiary Features (12 total)

| Feature | Type | Description |
|---------|------|-------------|
| `age_at_feature_date` | Float | Age in years at feature date |
| `sex_male` | Int (0/1) | 1 if male, 0 if female |
| `race_white` | Int (0/1) | 1 if white, 0 otherwise |
| `has_diabetes` | Int (0/1) | Chronic condition flag |
| `has_chf` | Int (0/1) | Congestive heart failure flag |
| `has_ischemic_heart_disease` | Int (0/1) | Ischemic heart disease flag |
| `has_copd` | Int (0/1) | COPD flag |
| `has_chronic_kidney_disease` | Int (0/1) | CKD flag |
| `has_stroke_or_tia` | Int (0/1) | Stroke/TIA flag |
| `has_depression` | Int (0/1) | Depression flag |
| `has_cancer` | Int (0/1) | Cancer flag |
| `has_osteoporosis` | Int (0/1) | Osteoporosis flag |
| `num_chronic_conditions` | Int | Count of chronic conditions (0-11) |
| `part_a_coverage_months` | Int | Months of hospital insurance coverage |
| `part_b_coverage_months` | Int | Months of medical insurance coverage |

### Inpatient Aggregation Features (12 total)

| Feature | Window | Description |
|---------|--------|-------------|
| `num_claims_90d` | 90 days | Number of inpatient claims |
| `total_cost_90d` | 90 days | Total Medicare payment |
| `avg_cost_per_claim_90d` | 90 days | Average cost per claim |
| `avg_los_90d` | 90 days | Average length of stay |
| `max_los_90d` | 90 days | Maximum length of stay |
| `has_claim_30d` | 30 days | Binary: any claim in last 30 days |
| `num_claims_365d` | 365 days | Annual claim count |
| `total_cost_365d` | 365 days | Annual total cost |
| `beneficiary_liability_90d` | 90 days | Deductible + coinsurance |
| `has_diabetes_90d` | 90 days | Diagnosis code 250.xx present |
| `has_sepsis_90d` | 90 days | Diagnosis code 995.9x present |
| `has_hf_90d` | 90 days | Diagnosis code 428.xx present |

### Metadata Columns

- `feature_date`: Point-in-time date (YYYY-MM-DD)
- `computed_at`: UTC timestamp when features were generated

---

## Point-in-Time Correctness

### Critical Principle

**All features use only data strictly BEFORE feature_date.**

Date filtering:
```python
# CORRECT: feature_date is exclusive upper bound
mask = dates < feature_date  # Use < not <=

# WRONG: Includes feature_date itself
mask = dates <= feature_date
```

Window definitions:
```python
feature_date = 2009-06-30
prior_90d  = [2009-03-02, 2009-06-30)   # Exclusive upper bound
prior_30d  = [2009-06-01, 2009-06-30)
prior_365d = [2008-07-01, 2009-06-30)
```

### Validation

The pipeline includes leakage detection at three levels:

1. **Schema Validator**: `validate_no_future_dates()` catches data after feature_date
2. **Feature Generator**: Uses `dates < feature_date` consistently
3. **Test Suite**: `TestPointInTimeCorrectness` verifies no future data in features

---

## Production Deployment

### Data Ingestion

```python
# Load with validation
from features.feature_generator import load_cms_data
from validation.schema_validator import SchemaValidator

bene_df, inp_df = load_cms_data(
    beneficiary_path='s3://data-lake/cms-bene-2009.csv',
    inpatient_path='s3://data-lake/cms-inp-2009.csv',
    sample_n=None  # Load full dataset
)

validator = SchemaValidator(raise_on_error=True)
assert validator.validate_beneficiary_data(bene_df)
assert validator.validate_inpatient_data(inp_df, feature_date='2009-06-30')
```

### Feature Generation

```python
gen = FeatureGenerator(verbose=False)
features = gen.compute_features(
    beneficiary_df=bene_df,
    inpatient_df=inp_df,
    feature_date='2009-06-30',
    output_path='s3://features/2009-06-30/features.parquet'
)
```

### Quality Assurance

```python
# Post-generation validation
assert validator.validate_features(features, feature_date='2009-06-30')

# Check no leakage
assert not (features['computed_at'].dt.date < pd.to_datetime('2009-06-30').date()).any()

# Log metrics
print(f"Generated {len(features)} beneficiary feature sets")
print(f"Feature coverage: {(features['age_at_feature_date'] > 0).sum() / len(features):.1%}")
```

### Scheduled Execution

```bash
# Daily cron job (example)
0 2 * * * python -c "
from features.feature_generator import FeatureGenerator, load_cms_data
gen = FeatureGenerator()
bene, inp = load_cms_data(...)
features = gen.compute_features(bene, inp, '$(date -d yesterday +%Y-%m-%d)')
"
```

---

## Common Issues

### Issue: "Missing columns: {XXX}"

**Cause**: Input CSV doesn't match schema contract  
**Fix**: Check column names in CSV header match contract definition

```python
# Debug
print(bene_df.columns.tolist())
print("Expected:", ["DESYNPUF_ID", "BENE_BIRTH_DT", ...])
```

### Issue: "Leakage detected: XXX rows with date >= feature_date"

**Cause**: Data from feature_date or after included in input  
**Fix**: Apply strict date filter BEFORE calling pipeline

```python
# Correct
claims = claims_df[claims_df['CLM_ADMSN_DT'] < '20090630']
features = gen.compute_features(bene, claims, '2009-06-30')
```

### Issue: Unexpected NULL values in required column

**Cause**: Schema changed or data quality degraded  
**Action**: 
1. Notify data engineering
2. Check source system for issues
3. Review data contract version history
4. Consider rolling back to previous schema version

---

## Testing

### Run All Tests

```bash
pytest tests/ -v --tb=short
```

### Run Specific Test Class

```bash
# Point-in-time correctness
pytest tests/test_feature_generation.py::TestPointInTimeCorrectness -v

# Feature computation
pytest tests/test_feature_generation.py::TestFeatureComputation -v

# Integration tests
pytest tests/test_feature_generation.py::TestIntegration -v
```

### Test Coverage

```bash
pytest tests/ --cov=features --cov=validation --cov-report=html
open htmlcov/index.html
```

### Test Organization

| Test Class | Purpose | Count |
|-----------|---------|-------|
| `TestSchemaValidation` | Data contract enforcement | 9 tests |
| `TestPointInTimeCorrectness` | Leakage detection | 4 tests |
| `TestFeatureComputation` | Feature logic correctness | 5 tests |
| `TestIntegration` | End-to-end pipeline | 2 tests |
| `TestEdgeCases` | Boundary conditions | 2 tests |

Total: **22 comprehensive test cases**

---

## Development

### Code Style

```bash
# Format code
black features/ validation/ tests/

# Lint
flake8 features/ validation/ tests/

# Type checking
mypy features/ validation/
```

### Adding New Features

1. **Define** feature in `features/feature_definitions.py`
   - Include description, data source, leakage notes
   - Set `is_point_in_time_safe = True` ONLY if verified

2. **Implement** computation in `features/feature_generator.py`
   - Use `self._apply_date_filter()` for time windows
   - Handle NULL/missing values explicitly
   - Document edge cases

3. **Test** thoroughly
   - Write unit tests in `tests/test_feature_generation.py`
   - Verify leakage with specific test case
   - Test edge cases (no data, all NULL, etc.)

4. **Validate** contract compliance
   - Update schema in `contracts/*.yaml` if needed
   - Document change in data contract version history
   - Notify data consumers of breaking changes

### Git Workflow

```bash
# Feature branch
git checkout -b feature/add-readmission-feature

# Commit with clear message
git commit -m "Add readmission-within-30d feature

- Compute time between discharge and next admission
- Filter to prior 90-day claims for baseline
- Add unit tests verifying leakage prevention
- Validate against inpatient_contract v1.0"

# Push and create PR
git push origin feature/add-readmission-feature
```

---

## References
# - See `SCHEMA_EVOLUTION_EXAMPLE.md` for handling schema changes
# - See `SERVING_STRATEGY.md` for online/nearline serving patterns

### CMS Data

- **Main Site**: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/
- **Codebook**: https://cms.gov/files/document/de-10-codebook.pdf
- **ResDAC**: https://resdac.org/ (additional documentation)
- **AWS Open Data**: https://registry.opendata.aws/cmsdesynpuf-omop/ (OMOP format subset)

### Key Documents

- **DESIGN_NOTES.md**: Technical deep dive on leakage traps and design decisions
- **contracts/beneficiary_contract.yaml**: Beneficiary schema definition
- **contracts/inpatient_contract.yaml**: Inpatient claims schema definition

### Learning Resources

- **Point-in-Time Correctness**: Kaggle blog on feature engineering timing
- **Data Contracts**: How to operationalize data contracts (datastack.tv)
- **CMS Data**: Medicare data basics from CMS Learning Network

---

## Support & Maintenance

### Regular Tasks

- **Weekly**: Monitor data quality metrics, check for schema changes
- **Monthly**: Review feature usage, run full validation suite
- **Quarterly**: Update documentation, audit leakage tests

### Escalation Path

1. **Data Quality Issue**: Email data-eng@example.com
2. **Schema Change Request**: Create GitHub issue with deprecation notice
3. **Production Incident**: Page oncall-ml@example.com

### Contact

- **Owner**: ML Engineering Team (ml-engineering@example.com)
- **Data Engineer**: data-eng@example.com
- **On-Call**: oncall-ml@example.com

---

**Version**: 1.0.0  
**Last Updated**: 2024-10-01  
**Status**: Production Ready
