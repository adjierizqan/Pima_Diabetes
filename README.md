# Pima Diabetes ML Demo

A machine-learning exercise using the Pima Indians Diabetes dataset. The project compares several classifiers and includes a small Flask interface for trying a saved model.

The dataset has 768 records and eight input features. This is an educational dataset, not a basis for clinical decisions.

## Run locally

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To retrain the models and generate evaluation plots:

```bash
python model_training.py --verbose
```

To start the Flask demo:

```bash
python app.py
```

Open http://127.0.0.1:5000/. Existing model artifacts are also included in the repository.

## What's included

- Data exploration and preprocessing, including handling zero values in selected fields
- Logistic Regression, Random Forest, XGBoost, SVM and KNN comparisons
- Saved evaluation results, confusion matrices, ROC curves and optional SHAP plots
- A Flask form for entering features and viewing the model's classification
- Basic tests in `tests/`, runnable with `pytest`

Dataset: [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Limitations

This is a coursework-style demo, not a validated medical device. The output should not be used for diagnosis, treatment, or personal health decisions. The model has not been shown to generalize to other populations or clinical settings.

Do not deploy the Flask app as-is. Its configuration and security need review before any public hosting.
