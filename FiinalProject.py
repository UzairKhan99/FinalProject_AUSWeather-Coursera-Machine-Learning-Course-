import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import seaborn as sns


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


df = pd.read_csv("Unconfirmed 141341.csv")
# Count missing values per column
missing_counts = df.isna().sum()

# Get the top 3 columns with most NaNs
cols_to_drop = missing_counts.sort_values(ascending=False).head(3).index

# Drop them
df = df.drop(columns=cols_to_drop)


df = df.rename(columns={"RainToday": "RainYesterday", "RainTomorrow": "RainToday"})


## Exercise 1: Map the dates to seasons and drop the Date column
df = df[
    df.Location.isin(
        [
            "Melbourne",
            "MelbourneAirport",
            "Watsonia",
        ]
    )
]

df["Date"] = pd.to_datetime(df["Date"])
df["Season"] = df["Date"].apply(date_to_season)


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


df = pd.read_csv("Unconfirmed 141341.csv")
# Count missing values per column
missing_counts = df.isna().sum()

# Get the top 3 columns with most NaNs
cols_to_drop = missing_counts.sort_values(ascending=False).head(3).index

# Drop them
df = df.drop(columns=cols_to_drop)


df = df.rename(columns={"RainToday": "RainYesterday", "RainTomorrow": "RainToday"})


## Exercise 1: Map the dates to seasons and drop the Date column
df = df[
    df.Location.isin(
        [
            "Melbourne",
            "MelbourneAirport",
            "Watsonia",
        ]
    )
]
## Exercise 2. Define the feature and target dataframes
df["Date"] = pd.to_datetime(df["Date"])
df["Season"] = df["Date"].apply(date_to_season)
df = df.drop(columns=["Date"])
print(df.columns)
y = df["RainToday"]
X = df.drop("RainToday", axis=1, inplace=True)
## Exercise 3. How balanced are the classes?
print(y.value_counts())


## Exercise 5. Split data into training and test sets, ensuring target stratification
