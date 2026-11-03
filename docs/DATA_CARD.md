# Data card

This project uses a support ticket dataset for queue classification and priority triage. The raw dataset is stored in `data/raw` and should be tracked with DVC.

## Intended use

Support routing, prioritization, and retrieval grounding for customer support workflows.

## Risks

- Class imbalance in queue and priority labels
- Data drift across time and customer cohorts
- Language and quality variation in ticket text
