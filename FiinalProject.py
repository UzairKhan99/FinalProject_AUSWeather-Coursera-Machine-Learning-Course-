import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import seaborn as sns


# Function to map date to season
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return "Summer"
    elif (month == 3) or (month == 4) or (month == 5):
        return "Autumn"
    elif (month == 6) or (month == 7) or (month == 8):
        return "Winter"
    elif (month == 9) or (month == 10) or (month == 11):
        return "Spring"


# Load data
df = pd.read_csv("Unconfirmed 141341.csv")

# Drop top 3 columns with most missing values
missing_counts = df.isna().sum()
cols_to_drop = missing_counts.sort_values(ascending=False).head(3).index
df = df.drop(columns=cols_to_drop)

# Rename columns
df = df.rename(columns={"RainToday": "RainYesterday", "RainTomorrow": "RainToday"})

## Exercise 1: Filter locations and convert Date to Season
df = df[df.Location.isin(["Melbourne", "MelbourneAirport", "Watsonia"])]
df["Date"] = pd.to_datetime(df["Date"])
df["Season"] = df["Date"].apply(date_to_season)
df = df.drop(columns=["Date"])

## Exercise 2. Define the feature and target dataframes
y = df["RainToday"]
X = df.drop("RainToday", axis=1)

# ✅ Drop rows with NaNs in target
X = X[y.notna()]
y = y[y.notna()]

## Exercise 3. How balanced are the classes?
print(y.value_counts())

## Exercise 5. Split data into training and test sets, ensuring target stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

## Exercise 6. Automatically detect numerical and categorical columns
numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
object_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Define preprocessing steps
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

## Exercise 7. Combine the transformers into a single preprocessing column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, object_features),
    ]
)

## Exercise 8. Create a pipeline by combining the preprocessing with a Random Forest classifier
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Define parameter grid for GridSearch
param_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
}

# Setup stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True)

## Exercise 9. Instantiate and fit GridSearchCV to the pipeline
grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid, cv=cv, scoring="accuracy", verbose=2
)

# Fit model
grid_search.fit(X_train, y_train)

# Results
print("\nBest parameters found:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
test_score = grid_search.score(X_test, y_test)
print("Test score:", test_score)

# Predictions
y_pred = grid_search.predict(X_test)

## Exercise 12. Print the classification report

print("classification_report:", classification_report(y_test, y_pred))

## Exercise 13. Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Rain Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
a
