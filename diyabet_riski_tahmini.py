# ---------- Import Libraries ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# ---------- Import Data and Perform EDA ----------

def load_and_describe_data(filepath):
    df = pd.read_csv(filepath)
    df.info()
    print(df.describe())
    return df

def plot_pairwise_relationships(df, target_column):
    sns.pairplot(df, hue=target_column)
    plt.show()

def plot_correlation_heatmap(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Correlation of Features")
    plt.show()

df = load_and_describe_data("diabetes.csv")
plot_pairwise_relationships(df, target_column="Outcome")
plot_correlation_heatmap(df)

# ---------- Outlier Detection ----------

def detect_outliers_iqr(df):
    outlier_indices = []
    outliers_df = pd.DataFrame()

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_indices.extend(outliers_in_col.index)
        outliers_df = pd.concat([outliers_df, outliers_in_col], axis=0)

    outlier_indices = list(set(outlier_indices))
    outliers_df = outliers_df.drop_duplicates()

    return outliers_df, outlier_indices

def remove_outliers(df, outlier_indices):
    return df.drop(outlier_indices).reset_index(drop=True)

outliers_df, outlier_indices = detect_outliers_iqr(df)
df_cleaned = remove_outliers(df, outlier_indices)

# ---------- Train-Test Split ----------

def split_data(df, target_column):
    x = df.drop([target_column], axis=1)
    y = df[target_column]
    return train_test_split(x, y, test_size=0.25, random_state=42)

x_train, x_test, y_train, y_test = split_data(df_cleaned, target_column="Outcome")

# ---------- Standardization ----------

def standardize_data(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

x_train_scaled, x_test_scaled = standardize_data(x_train, x_test)

# ---------- Model Training and Evaluation ----------

def get_based_models():
    return [
        ("LR", LogisticRegression()),
        ("DT", DecisionTreeClassifier()),
        ("KNN", KNeighborsClassifier()),
        ("NB", GaussianNB()),
        ("SVM", SVC()),
        ("AdaB", AdaBoostClassifier()),
        ("GBM", GradientBoostingClassifier()),
        ("RF", RandomForestClassifier()),
    ]

def train_base_models(x_train, y_train, models):
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name}: accuracy: {cv_results.mean()}, std: {cv_results.std()}")
    return names, results

def plot_model_accuracies(names, results):
    df = pd.DataFrame({names[i]: results[i] for i in range(len(names))})
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.show()

models = get_based_models()
names, results = train_base_models(x_train_scaled, y_train, models)
plot_model_accuracies(names, results)

# ---------- Hyperparameter Tuning ----------

def tune_hyperparameters(x_train, y_train):
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring="accuracy")
    grid_search.fit(x_train, y_train)

    print("Best Parameters: ", grid_search.best_params_)
    return grid_search.best_estimator_

best_dt_model = tune_hyperparameters(x_train_scaled, y_train)

# ---------- Model Evaluation ----------

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report")
    print(classification_report(y_test, y_pred))

evaluate_model(best_dt_model, x_test_scaled, y_test)

# ---------- Model Testing with Real Data ----------

def test_with_real_data(model, new_data):
    prediction = model.predict(new_data)
    print("New Prediction: ", prediction)

new_data = np.array([[5, 90, 72, 35, 0, 34.6, 0.627, 35]])
test_with_real_data(best_dt_model, new_data)
