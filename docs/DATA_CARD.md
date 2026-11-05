# Data card

## Source

`data/raw/customer-support-tickets.csv` is the real support-ticket corpus used by the pipeline. DVC tracks the processed artifacts and deterministic splits.

## Pipeline

1. Read the raw CSV.
2. Filter to English tickets.
3. Normalize subject/body whitespace.
4. Normalize queue and priority labels.
5. Drop null or blank required fields.
6. Remove duplicate subject/body pairs.
7. Merge queues with fewer than 5 training rows into `Other`.
8. Create a stratified, group-aware 70/15/15 split.

Run `make data` or `dvc repro ingest clean dedupe split`.

## Recorded audit

The current audit artifact is `reports/data_audit.json`:

- Raw rows: `61,765`
- English rows: `24,621`
- Duplicate rows removed: `3,702`
- Null rows removed: `5,301`
- Train / validation / test: `14,643 / 3,138 / 3,138`

## Label coverage

The English split contains ten real queue labels plus `Other`. Priority labels in the English data are `medium`, `high`, and `low`; `critical` and `very_low` are not present in the English corpus even though the raw all-language schema documents five priority levels.

## Intended use

Queue routing, priority triage, retrieval grounding, incident monitoring, and human-reviewed support drafting.

## Risks and limitations

- Class imbalance across queues and priorities.
- The English-only filter removes non-English customers.
- Missing `critical` and `very_low` English examples limit priority recall for those classes.
- Distribution drift across time, products, and customer cohorts.
- Human review remains mandatory for every outbound reply.
