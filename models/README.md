# Ticket Classification Model

This folder contains the first production-ready baseline model for Tickets Copilot.

## Goal

Predict the priority or ticket category from a ticket subject and body.

## Recommended first model

- Task: ticket priority classification
- Language: English only for the first baseline
- Split: 80/10/10 with stratification
- Model: TF-IDF + Logistic Regression

## Run

```bash
python3 scripts/train_priority_model.py
```

This script will:

- load the dataset from the local CSV file
- filter to English tickets
- split into train/validation/test sets
- train a baseline classifier
- print accuracy and F1 metrics
- save the trained model to `models/baseline_priority_model.joblib`
