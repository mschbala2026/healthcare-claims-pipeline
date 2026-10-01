# Project Documentation

## contracts/ folder
YAML files defining valid data structure:
- beneficiary_contract.yaml: 32 columns, demographics and conditions
- inpatient_contract.yaml: 81 columns, claims dates and costs

## tests/ folder
22 test cases verifying pipeline works correctly:
- Checks: data validation, leakage prevention, feature calculation, end-to-end flow

## features/ folder
Code that creates 25 features from raw data:
- Generates: 12 beneficiary features + 12 claims features + diagnosis flags

## data/ folder
Folder for input and output data files:
- Contains: raw/ (input CSVs) and processed/ (output features)

## requirements.txt
Python packages needed to run the pipeline:
- Lists: pandas, numpy, pyyaml, pytest for development

## main.py
Entry point to run the entire pipeline:
- Loads data → validates → generates features → saves output

## example_usage.py
Shows how to use the pipeline:
- Example code: loading data, generating features, validating output