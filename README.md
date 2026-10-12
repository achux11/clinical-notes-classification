# Clinical Notes Classification — Fine-Tuned Transformer

Classifies de-identified clinical transcriptions into medical specialties by fine-tuning DistilBERT in PyTorch, compared against a TF-IDF + logistic regression baseline.

**Data:** MTSamples medical transcriptions ([Kaggle: tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)) — 1,602 notes across the 6 largest clinical specialties after removing catch-all categories.

## Approach
- Stratified train / validation / test split (321 test notes, locked)
- Baseline: TF-IDF (1–2 grams) + logistic regression
- DistilBERT fine-tuned in PyTorch; best epoch selected on **validation** macro-F1, test evaluated once
- Per-class error analysis

## Results (test set, 321 notes)
| Model | Macro-F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.701 |
| **DistilBERT (fine-tuned)** | **0.736** |

| Specialty | F1 |
|---|---|
| Urology | 0.900 |
| Orthopedic | 0.855 |
| Gastroenterology | 0.804 |
| Cardiovascular / Pulmonary | 0.781 |
| Neurology | 0.562 |
| Radiology | 0.511 |

**Finding:** Radiology is the weakest class because imaging reports share vocabulary with the specialties they image (spine, heart, abdomen).

## Run (Google Colab, T4 GPU)
Run the setup commands at the top of the script, then the script. Part B (LangGraph agent) is optional and needs a `GROQ_API_KEY`.
