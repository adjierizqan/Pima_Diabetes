# Pima Diabetes Prediction Web Application

An end-to-end, production-ready machine learning system that predicts diabetes risk using the Pima Indians Diabetes dataset. The project covers data exploration, preprocessing, model training, evaluation, and deployment via a Flask web application.

## Project Structure
```
.
├── app.py
├── data/
│   └── diabetes.csv
├── model_training.py
├── reports/
│   ├── confusion_matrices/
│   ├── feature_importance/
│   ├── roc_curves/
│   └── shap/
├── scaler.pkl              # generated after training
├── best_model.pkl          # generated after training
├── requirements.txt
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   └── result.html
└── tests/
    ├── test_prediction.py
    └── test_preprocessing.py
```

## Dataset

The [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) contains 768 samples with eight clinical measurements and a binary outcome indicating diabetes status.

## Getting Started

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train models and generate reports

```bash
python model_training.py --verbose
```

This command will:

- Perform exploratory data analysis and store visualizations in `reports/`.
- Preprocess the dataset (median imputation for zero values, scaling).
- Train Logistic Regression, Random Forest, XGBoost, SVM, and KNN models with grid search.
- Evaluate models on the test split and record metrics in `reports/model_performance.csv`.
- Select the best-performing model (by F1-score) and save artifacts:
  - `scaler.pkl` – preprocessing pipeline.
  - `best_model.pkl` – the tuned classifier.
- Generate SHAP explainability plots for tree-based best models.
- Export processed train/test splits to `data/processed_train.csv` and `data/processed_test.csv`.

### 4. Launch the web application

```bash
python app.py
```

Open your browser at [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the prediction UI.

## Features

- Clean separation between data science workflow and web interface.
- Automated EDA with saved plots (histograms, pairplot, correlation heatmap).
- Robust preprocessing (zero-value handling, median imputation, scaling).
- Hyperparameter tuning with cross-validation for five algorithms.
- Model performance tracking with accuracy, precision, recall, F1-score, and ROC-AUC.
- Confusion matrices and ROC curves for all trained models.
- SHAP explainability for tree-based models.
- Responsive Bootstrap 5 UI with sample data autofill.

## Testing

Basic unit tests validate preprocessing and prediction pipelines.

```bash
pytest
```

## Deployment

The Flask app is stateless and can be containerized or deployed to services such as Render or Heroku. Ensure environment variables allow access to the trained artifacts (`scaler.pkl` and `best_model.pkl`).

## Sample Input

| Feature | Example Value |
|---------|---------------|
| Pregnancies | 2 |
| Glucose | 130 |
| BloodPressure | 70 |
| SkinThickness | 30 |
| Insulin | 100 |
| BMI | 32.5 |
| DiabetesPedigreeFunction | 0.5 |
| Age | 45 |

## Screenshots

Screenshots can be generated after running the application locally. Use your browser’s screenshot tool to capture the form and prediction output pages.

## License

This project uses the Pima Indians Diabetes dataset under the [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) license provided on Kaggle.
