# Data Period Justification

## Why 2008-2010?

The assignment specifies:
> "Use the CMS DE-SynPUF data. You may work with the official CMS release..."

The **public CMS DE-SynPUF dataset covers 2008-2010 only**. This is not a limitation—it's the resource available.

---

## What We Have

- **Beneficiaries:** 2.3M individuals
- **Inpatient Claims:** 500k+ admissions
- **Coverage:** Full years 2008, 2009, 2010
- **Format:** Public domain, fully synthetic (no real patient data)

---

## Feature Date Selection: 2009-06-30

### Pros of This Choice

| Aspect | Value | Why It Matters |
|--------|-------|---|
| **Historical Data** | 18 months | Enough for features |
| **Label Window** | 210 days | Full 90d prediction |
| **Position** | Middle of data | Balanced choice |
| **Data Lag** | Realistic | Like production systems |

### Alternative Dates (Why We Didn't Choose Them)