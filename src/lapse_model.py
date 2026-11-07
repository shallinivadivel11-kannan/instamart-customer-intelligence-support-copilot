from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parent.parent

# Load features
df = pd.read_csv(ROOT / "outputs/lapse_features.csv")

# Target
y = df["lapsed"]

# Exclude ID, target and non-feature columns
X = df.drop(columns=[
    "customer_id",
    "lapsed",
    "signup_date"
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Baseline model: always predict not lapsed
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

predictions = baseline.predict(X_test)

print("\nBaseline accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nBaseline classification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Numerical features
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "avg_delivery_minutes",
    "cancellation_rate",
    "tenure_days"
]

# Categorical features
categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data without fitting
X_test_processed = preprocessor.transform(X_test)

print("\nTraining data after preprocessing:", X_train_processed.shape)
print("Testing data after preprocessing:", X_test_processed.shape)

print("\nPreprocessing completed successfully!")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report

# Create pipeline
logistic_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Hyperparameter tuning
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    logistic_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Logistic Regression parameters:")
print(grid_search.best_params_)

print("Best cross-validation F1:", grid_search.best_score_)

# Test evaluation
best_logistic = grid_search.best_estimator_
logistic_predictions = best_logistic.predict(X_test)

print("\nLogistic Regression Test Results:")
print(classification_report(
    y_test,
    logistic_predictions,
    zero_division=0
))

from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_param_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

print("\nBest Random Forest parameters:")
print(rf_search.best_params_)

print("Best cross-validation F1:", rf_search.best_score_)

best_rf = rf_search.best_estimator_
rf_predictions = best_rf.predict(X_test)

print("\nRandom Forest Test Results:")
print(classification_report(
    y_test,
    rf_predictions,
    zero_division=0
))

from sklearn.metrics import confusion_matrix

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

# Oversample training data only
from sklearn.utils import resample

train_data = X_train.copy()
train_data["lapsed"] = y_train.values

majority = train_data[train_data["lapsed"] == 0]
minority = train_data[train_data["lapsed"] == 1]

minority_oversampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_train = pd.concat(
    [majority, minority_oversampled]
)

X_balanced = balanced_train.drop(columns=["lapsed"])
y_balanced = balanced_train["lapsed"]

print("\nBalanced training distribution:")
print(y_balanced.value_counts())

# Refit best Logistic Regression
balanced_logistic = grid_search.best_estimator_
balanced_logistic.fit(X_balanced, y_balanced)

balanced_predictions = balanced_logistic.predict(X_test)

print("\nOversampled Logistic Regression Results:")
print(classification_report(
    y_test,
    balanced_predictions,
    zero_division=0
))

# Random Forest feature importance
rf_preprocessor = best_rf.named_steps["preprocessor"]
rf_model = best_rf.named_steps["model"]

feature_names = rf_preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 5 Important Features:")
print(importance.head(5))

# Logistic Regression coefficients
logistic_preprocessor = best_logistic.named_steps["preprocessor"]
logistic_model = best_logistic.named_steps["model"]

feature_names = logistic_preprocessor.get_feature_names_out()

coefficients = pd.DataFrame({
    "feature": feature_names,
    "coefficient": logistic_model.coef_[0]
})

coefficients["absolute_value"] = coefficients["coefficient"].abs()

coefficients = coefficients.sort_values(
    by="absolute_value",
    ascending=False
)

print("\nTop 3 Logistic Regression Coefficients:")
print(coefficients.head(3))

import joblib

# Save selected model
model_path = ROOT / "outputs" / "lapse_model.joblib"

joblib.dump(balanced_logistic, model_path)

print("\nFinal model saved successfully!")
print("Model path:", model_path)

# Reload model
loaded_model = joblib.load(model_path)

# Verify predictions
reloaded_predictions = loaded_model.predict(X_test)

print(
    "\nReloaded model predictions match:",
    (reloaded_predictions == balanced_predictions).all()
)