# MLOps plan

This project uses DVC for data versioning, MLflow for experiment tracking, GitHub Actions for CI validation, and regular model quality gates before promotion.

## Governance

- Support agent retains final decision authority.
- Model files and metrics are versioned through MLflow and DVC.
- Data and model quality gates run before promotion to production.
