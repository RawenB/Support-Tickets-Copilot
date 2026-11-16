# Model cards

Both tasks use the same architecture: a **ticket-memory ensemble**, a weighted soft vote over

| Member | What it captures |
|---|---|
| `tfidf_sgd` | word 1-3-gram TF-IDF + logistic SGD (the v0.1.0 champion recipe) |
| `knn_char` | 15 nearest past tickets by character 3-5-gram TF-IDF, similarity^10 voting |
| `knn_word` | 15 nearest past tickets by word 1-2-gram TF-IDF, similarity^4 voting |
| `knn_embedding` | 5 nearest past tickets by `all-MiniLM-L6-v2` embedding, similarity^10 voting |

Weights are tuned on the validation split (simplex grid, step 0.1). After selection, the
memory members are refit on train+val. The test split is used once, for the champion.
Configuration: `configs/train_<task>.yaml`. Input: `build_text(subject, body)`
(lower-cased, whitespace-collapsed; the same function at training and serving time).

## Why this architecture

The dataset is synthetic, and support tickets recur: 31 % of test tickets have no close
train/val neighbour ("novel"), while the other 69 % are paraphrases of a known ticket and
share its label 95 %+ of the time. Looking up similar past tickets is the strongest
signal, and three different similarity views make different mistakes.

| Experiment (queue, val macro F1) | Result |
|---|---|
| fine-tuned DistilBERT (old code) | 0.28 |
| fine-tuned `roberta-base`, fixed recipe, 2 of 4 epochs | 0.33 (stopped; MLflow run `b079faa8…`) |
| TF-IDF + SGD | 0.70 |
| best single kNN view (char) | 0.75 |
| **ensemble** | **0.78** |

## Queue model: `support-copilot-queue` v7 (`@champion`, tag `model-queue-v7`)

- Purpose: route a ticket to one of 10 queues.
- Ensemble weights: tfidf_sgd 0.4, knn_char 0.3, knn_embedding 0.2, knn_word 0.1.

| Test (n = 3,562, data-v2) | macro F1 | accuracy |
|---|---|---|
| all tickets | **0.823** | 0.826 |
| novel tickets (31 %) | 0.469 | 0.520 |
| seen-like tickets (69 %) | 0.959 | |
| v0.1.0 champion (data-v1 test set) | 0.711 (novel 0.424) | 0.711 |

## Priority model: `support-copilot-priority` v7 (`@champion`, tag `model-priority-v7`)

- Purpose: rank urgency as high / medium / low (the English data has no critical / very_low).
- Ensemble weights: knn_word 0.4, knn_char 0.3, knn_embedding 0.3 (TF-IDF SGD got weight 0).

| Test (n = 3,562, data-v2) | macro F1 | accuracy |
|---|---|---|
| all tickets | **0.847** | 0.850 |
| novel tickets (31 %) | 0.554 | |
| seen-like tickets (69 %) | 0.972 | |
| v0.1.0 champion (data-v1 test set) | 0.734 (novel 0.512) | 0.747 |

## Limitations

- On genuinely new tickets the models are much weaker (queue 0.47, priority 0.55 macro F1).
  Low-confidence predictions go to human review (`review_threshold` in `params.yaml`).
- Priority labels are noisy: semantically urgent text can be labelled "low" in the data,
  and the model follows the data.
- Memory models grow with the training set (≈250 MB per task) and need
  `all-MiniLM-L6-v2` at inference time (already used by the RAG module).
- English only; the 33k German tickets are not used.

## Release gate

`f1_gate` (0.70 macro F1) and `promotion.novel_f1_gate` (0.40 on novel tickets) in
`params.yaml`; a challenger may not regress more than `promotion.max_regression` against
`@champion`. See `docs/MLOPS.md`.
