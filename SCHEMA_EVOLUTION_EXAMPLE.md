# Schema Evolution Examples

This document shows how the pipeline handles schema changes when data structures evolve over time.

## Scenario 1: Renamed Column

**Old Schema (2008-2009):**
```yaml
columns:
  CLM_ID: integer  # Claim identifier
  CLM_ADMSN_DT: string  # Admission date
```

**New Schema (2010+):**
```yaml
columns:
  CLAIM_ID: integer  # Renamed from CLM_ID
  CLM_ADMSN_DT: string  # Unchanged
```

**How to Handle:**
```python
# In schema_validator.py, add migration logic
def _migrate_column_rename(df):
    if 'CLM_ID' in df.columns and 'CLAIM_ID' not in df.columns:
        df.rename(columns={'CLM_ID': 'CLAIM_ID'}, inplace=True)
        logger.warning("Migrating CLM_ID → CLAIM_ID (schema version 2.0)")
    return df
```

**Contract Update:**
```yaml
metadata:
  version: 2.0
  breaking_changes:
    - "CLM_ID renamed to CLAIM_ID in version 2.0"
  migration:
    old_column: CLM_ID
    new_column: CLAIM_ID
```

---

## Scenario 2: New Nullable Column

**Old Schema (2008-2009):**
```yaml
columns:
  DIAGNOSIS_PRIMARY: string  # ICD9 diagnosis code
```

**New Schema (2010+):**
```yaml
columns:
  DIAGNOSIS_PRIMARY: string  # Unchanged
  DIAGNOSIS_SECONDARY: string  # New column (nullable)
```

**How to Handle:**
```python
# In feature_generator.py, handle missing column gracefully
def _extract_diagnosis_features(self, inp_df):
    result = {}
    
    if 'DIAGNOSIS_PRIMARY' in inp_df.columns:
        result['has_primary_diagnosis'] = (inp_df['DIAGNOSIS_PRIMARY'] != '').astype(int)
    
    # New column - provide default if missing
    if 'DIAGNOSIS_SECONDARY' in inp_df.columns:
        result['has_secondary_diagnosis'] = (inp_df['DIAGNOSIS_SECONDARY'] != '').astype(int)
    else:
        result['has_secondary_diagnosis'] = 0  # Default: no secondary diagnosis
    
    return result
```

**Contract Update:**
```yaml
inpatient_schema:
  DIAGNOSIS_PRIMARY:
    type: string
    nullable: false
    version_added: "1.0"
  DIAGNOSIS_SECONDARY:
    type: string
    nullable: true
    version_added: "2.0"
    default: null
```

---

## Scenario 3: Type Change

**Old Schema (2008-2009):**
```yaml
columns:
  NCH_BENE_IP_DDCTBL_AMT: integer  # Deductible amount in cents
```

**New Schema (2010+):**
```yaml
columns:
  NCH_BENE_IP_DDCTBL_AMT: decimal  # Now precise decimal dollars
```

**How to Handle:**
```python
# In _parse_amounts(), handle type conversion
def _parse_amounts(self, df, column_name):
    """Convert amounts to decimal dollars, handling old int format"""
    
    if df[column_name].dtype == 'int64':
        # Old format: amounts in cents
        return (df[column_name] / 100).astype('float64')
    elif df[column_name].dtype == 'float64':
        # New format: amounts in dollars
        return df[column_name].astype('decimal')
    else:
        raise ValueError(f"Unexpected type for {column_name}: {df[column_name].dtype}")
```

**Contract Update:**
```yaml
NCH_BENE_IP_DDCTBL_AMT:
  type: decimal
  precision: 10
  scale: 2
  unit: "USD"
  version_added: "1.0"
  version_changed: "2.0"
  change_note: "Changed from integer (cents) to decimal (dollars) in version 2.0"
```

---

## Scenario 4: New Categorical Value

**Old Schema (2008-2009):**
```yaml
BENE_SEX_IDENT_CD:
  type: enum
  values:
    - 1  # Male
    - 2  # Female
```

**New Schema (2010+):**
```yaml
BENE_SEX_IDENT_CD:
  type: enum
  values:
    - 1  # Male
    - 2  # Female
    - 3  # Prefer not to say (NEW)
```

**How to Handle:**
```python
# In _validate_enum_values(), allow new values gracefully
def validate_enum_values(self, df, column_name, allowed_values):
    """Validate enum, log warning for new values"""
    
    unique_values = df[column_name].unique()
    unexpected = set(unique_values) - set(allowed_values)
    
    if unexpected:
        # New values in data - log but don't fail
        logger.warning(f"New values in {column_name}: {unexpected}")
        return True  # Accept with warning
    
    return True
```

**Contract Update:**
```yaml
BENE_SEX_IDENT_CD:
  type: enum
  values:
    - 1  # Male
    - 2  # Female
    - 3  # Prefer not to say (added in 2.0)
  allowed_values_version:
    "1.0": [1, 2]
    "2.0": [1, 2, 3]
```

---

## Schema Version Management

All schema changes should be tracked in contracts:

```yaml
metadata:
  schema_version: "2.0"
  previous_version: "1.0"
  
version_history:
  "1.0":
    date: 2024-01-01
    changes:
      - "Initial schema"
  
  "2.0":
    date: 2024-06-01
    breaking_changes:
      - "CLM_ID renamed to CLAIM_ID"
    non_breaking_changes:
      - "Added DIAGNOSIS_SECONDARY (nullable)"
      - "Changed NCH_BENE_IP_DDCTBL_AMT type to decimal"
      - "Added value 3 to BENE_SEX_IDENT_CD"
    migration_required: true
    migration_script: "scripts/migrate_v1_to_v2.py"
```

---

## Testing Schema Changes

```python
# In test_feature_generation.py, add tests for schema evolution

def test_renamed_column_migration():
    """Test that renamed columns are handled correctly"""
    # Old schema with CLM_ID
    old_data = pd.DataFrame({'CLM_ID': [1, 2, 3]})
    migrated = migrate_column_rename(old_data)
    assert 'CLAIM_ID' in migrated.columns
    assert migrated['CLAIM_ID'].tolist() == [1, 2, 3]

def test_new_nullable_column():
    """Test that new nullable columns default correctly"""
    # Data without DIAGNOSIS_SECONDARY
    data = pd.DataFrame({'DIAGNOSIS_PRIMARY': ['001', '002']})
    features = extract_diagnosis_features(data)
    assert features['has_secondary_diagnosis'] == 0

def test_type_conversion():
    """Test that type changes are handled correctly"""
    # Old format: cents
    old_data = pd.DataFrame({'NCH_BENE_IP_DDCTBL_AMT': [500, 1000]})  # 5.00, 10.00 dollars
    amounts = parse_amounts(old_data, 'NCH_BENE_IP_DDCTBL_AMT')
    assert amounts.tolist() == [5.0, 10.0]

def test_new_enum_value():
    """Test that new enum values are accepted with warning"""
    # New value not in old schema
    data = pd.DataFrame({'BENE_SEX_IDENT_CD': [1, 2, 3]})  # 3 is new
    result = validate_enum_values(data, 'BENE_SEX_IDENT_CD', [1, 2])
    assert result == True  # Accepted (with warning logged)
```

---

## Summary

Schema evolution is handled through:
1. **Defensive coding** - Check for column existence before use
2. **Versioned contracts** - Track changes in YAML schemas
3. **Migration functions** - Explicit transformation logic
4. **Logging** - Warn about unexpected values/types
5. **Testing** - Verify each scenario works correctly

This ensures the pipeline adapts gracefully as data structures change over time.