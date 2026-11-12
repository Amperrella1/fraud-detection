# Fraud Detection with SQL, Pandas, Numpy and PyTorch in Databricks

## Project Summary

This project develops and evaluates a financial transaction fraud
detection workflow in Databricks, combining SQL data preparation,
an interpretable rule-based baseline, and a supervised neural network
built with PyTorch.

The workflow covers duplicate handling, exploratory analysis, feature
preprocessing, model training, and evaluation. It also examines the
operational trade-off between catching fraud and generating false alerts.

## Approach

- Removed exact duplicate records and excluded transaction IDs with
  conflicting versions, retaining 49,881 unique transactions.
- Built a SQL baseline using transaction amount, recent transaction
  activity, and previous fraudulent transactions.
- Divided the data into training, validation, and test sets with no
  overlapping users.
- Fitted missing-value imputation, scaling, and categorical encoding
  exclusively on training data.
- Represented transaction hour using cyclical features.
- Trained a multilayer Variational Auto Encoder with 64 and 32-unit hidden layers,
  dropout, and early stopping based on validation average precision.
- Selected an alert threshold using validation data and applied the
  same threshold to the test set.
- Evaluated precision, recall, average precision, ROC-AUC, and lift,
  alongside the number of alerts requiring investigation.

The fraud label was used as the training target and for evaluation.
It was excluded from prediction inputs, along with transaction and
user identifiers.

## Results

The MLP achieved modest predictive lift on the test set:

| Metric | Result |
|---|---:|
| Test transactions | 9,287 |
| Actual fraudulent transactions | 459 |
| Transactions flagged | 494 |
| Fraud cases detected | 27 |
| False alerts | 467 |
| Precision | 5.47% |
| Recall | 5.88% |
| Average precision | 0.0533 |
| ROC-AUC | 0.5279 |
| Lift over random selection | 1.11× |

The model's alert precision exceeded the test fraud prevalence of
4.94%, although the improvement was small. At the same alert volume,
random selection would identify approximately 24 fraud cases on
average, compared with 27 detected by the model.

Threshold analysis showed that increasing the alert volume improved
recall while substantially increasing false positives.

## Limitations and Findings

The available features showed limited predictive signal in the
experiments performed. This does not establish that the fraud labels
are random, but it limits the conclusions that can be drawn from
model tuning alone.

The dataset contains transaction hour rather than a complete timestamp.
Consequently, the project does not construct reliable historical
features such as a customer's first use of a device or a new location.

Previous fraudulent transaction counts were included in the SQL
baseline but excluded from the MLP because their availability at
prediction time was not verified. The models also operate at different
alert volumes, so their raw detection counts are not directly comparable.

The test partition was examined during earlier project iterations.
The reported results are therefore a project benchmark rather than
a completely untouched final evaluation.

## Conclusion

The project demonstrates a complete fraud-modeling workflow using
SQL, Databricks, and PyTorch. The MLP produced a small improvement over
random selection, but its current performance does not support
operational deployment.

The main finding is that model complexity must be supported by
informative data. Further development would prioritize verified label
provenance, chronological transaction history, and customer-relative
behavioral features.

## Project Files and Execution

1. Data Cleaning — creates `fraud_transactions_for_detection`.
2. Exploratory Analysis — examines the source data and fraud distribution.
3. Fraud Detector (Basic) — creates and evaluates the SQL rule baseline.
4. Fraud Detector Final — prepares features, trains the MLP, and evaluates
   validation and test predictions.

Import the source CSV into Databricks as `fraud_detection_dataset`.
Run Data Cleaning before either detector. Run the basic detector before
the final notebook's SQL comparison cells.

## Technology

SQL, Databricks, Python, pandas, NumPy, scikit-learn, PyTorch, Matplotlib.

## Data Source

[Fraud Detection Dataset — Ranjit Mandal, Kaggle](https://www.kaggle.com/datasets/ranjitmandal/fraud-detection-dataset-csv)
