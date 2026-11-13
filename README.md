# Fraud Detection with SQL and Machine Learning in Databricks

## Project Summary

This project develops a financial transaction fraud detection workflow
using SQL, pandas, NumPy, scikit-learn, and PyTorch in Databricks.

It compares a rule-based SQL detector with three machine-learning
approaches: a variational autoencoder (VAE), a supervised multilayer
perceptron (MLP), and balanced logistic regression.

The project covers data cleaning, exploratory analysis, preprocessing,
model training, and evaluation. It examines how model choice and alert
thresholds affect fraud detection and investigation workload.

## Data Preparation

- Removed exact duplicates and excluded transaction IDs with conflicting
  records, retaining 49,881 unique transactions.
- Used SQL to explore transaction characteristics and fraud distribution.
- Imputed missing numerical values and encoded categorical features.
- Fitted learned preprocessing steps only on training data, or within
  individual training folds during cross-validation.
- Represented transaction hour using cyclical features.
- Excluded transaction IDs, user IDs, and the fraud label from model inputs.

The VAE and MLP experiments used user-separated training, validation,
and test sets. The later logistic regression experiment used stratified
random transaction splits and cross-validation. Its results therefore
do not provide a directly equivalent comparison with the earlier models.

## Modeling Approaches

### SQL Rule-Based Baseline

Built an interpretable detector using transaction amount, transaction
activity over the previous 24 hours, and previous fraudulent transactions.
Transactions triggering at least two rules were flagged for review.

### Variational Autoencoder — Anomaly Detection

Implemented a VAE in PyTorch and trained it on legitimate transactions
to learn a representation of normal transaction patterns.

The model used an encoder with 64- and 32-unit hidden layers, a
four-dimensional latent space, and a decoder with separate numerical
and categorical reconstruction outputs.

Training combined numerical reconstruction loss, categorical
reconstruction loss, and KL-divergence regularization. Reconstruction
masks excluded missing source values from reconstruction-loss terms.

Early stopping monitored loss on legitimate validation transactions.
Reconstruction-based anomaly scores were then used to flag transactions
that differed from the learned patterns.

Fraud labels identified legitimate training examples and supported
evaluation; they were not supplied as prediction inputs. The experiment
tested whether unusual transactions also corresponded to labeled fraud.

### Multilayer Perceptron — Supervised Classification

Implemented a PyTorch classifier with 64- and 32-unit hidden layers,
ReLU activations, and dropout.

The model learned from both legitimate and fraudulent training examples
using binary cross-entropy loss. Early stopping selected model weights
based on validation average precision.

An alert threshold was selected from validation scores and applied
unchanged to the test partition.

### Balanced Logistic Regression

Implemented logistic regression with balanced class weights to give
fraudulent examples greater influence during training.

A scikit-learn pipeline combined preprocessing and classification.
Five-fold cross-validation selected the regularization strength using
average precision, followed by evaluation on a separate validation set.

## Results

Across the experiments, the models showed limited ability to distinguish
fraudulent transactions from legitimate ones.

The VAE's anomaly scores did not effectively separate the fraud labels.
The supervised MLP and balanced logistic regression also produced
performance close to random ranking.

The following results belong specifically to the MLP test evaluation:

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

The MLP achieved a small observed lift over random selection. At the same
alert volume, random selection would identify approximately 24 fraud
cases on average, compared with 27 detected by the model. This difference
does not establish a reliable operational improvement.

Threshold analysis showed that increasing alert volume improved recall
while substantially increasing false positives. Balanced logistic
regression similarly generated more alerts without strong discrimination.

## Data Quality and Limitations

The experiments raised concerns about the relationship between the
source features and the `Fraudulent` label. Feature-level checks and
multiple modeling approaches found little predictive signal, consistent
with a possible label-generation or label-mapping issue.

The exact cause remains unverified. Without source lineage or the
label-generation process, these results do not prove that labels were
randomly assigned or that a specific ingestion error occurred.

Additional limitations include:

- Transaction hour is available, but a complete timestamp is not.
  Reliable chronological evaluation and customer-history features
  could therefore not be constructed.
- Previous fraud counts were used in the SQL baseline but excluded
  from the MLP and logistic regression because their availability
  at prediction time was not verified.
- The experiments used different split strategies and alert volumes,
  limiting direct comparisons.
- The test partition was examined during project development, so the
  results are a project benchmark rather than a completely untouched
  final evaluation.
- Anomalous behavior is not necessarily fraudulent behavior, which
  limits the interpretation of VAE reconstruction scores.

## Conclusion

The project demonstrates SQL data preparation, unsupervised anomaly
detection, supervised classification, cross-validation, and evaluation
on imbalanced transaction data.

Its main finding is that greater model complexity did not produce
reliable fraud detection with the available inputs and labels.
The VAE remains a documented experiment in probabilistic modeling,
while the supervised models provide complementary benchmarks.

Further development will prioritize validating label provenance and
using data with documented fraud behavior and chronological customer
history before undertaking additional model tuning.

## Project Files and Execution

- **Data Cleaning:** creates `fraud_transactions_for_detection`.
- **Exploratory Analysis:** examines the source data and fraud distribution.
- **Fraud Detector (Basic):** creates and evaluates the SQL rule baseline.
- **VAE notebook:** prepares inputs, trains the autoencoder, and evaluates
  reconstruction-based anomaly scores.
- **Fraud Detector Final:** prepares features, trains the MLP, and evaluates
  validation and test predictions.
- **Logistic regression notebook:** imports the cleaned data, constructs
  stratified splits, and trains a balanced classifier with cross-validation.

Import the source CSV into Databricks as `fraud_detection_dataset`.
Run Data Cleaning before the modeling notebooks. Run the basic detector
before the MLP notebook's SQL comparison cells.

## Technology

SQL, Databricks, Python, pandas, NumPy, scikit-learn, PyTorch, Matplotlib.

## Data Source

[Fraud Detection Dataset — Ranjit Mandal, Kaggle](https://www.kaggle.com/datasets/ranjitmandal/fraud-detection-dataset-csv)
