# =====================================================
# STUDENT PERFORMANCE PREDICTION USING MACHINE LEARNING
# =====================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("./AI-Data.csv")

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
print("Dataset Shape:", df.shape)
print(df.head())

# =========================
# MISSING VALUE ANALYSIS
# =========================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": (missing / len(df)) * 100
})

print(
    missing_df.sort_values(
        by="Percentage",
        ascending=False
    )
)

# =========================
# HANDLE MISSING VALUES
# =========================

numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_cols = df.select_dtypes(
    include="object"
).columns

for col in numeric_cols:
    df[col] = df[col].fillna(
        df[col].median()
    )

for col in categorical_cols:
    df[col] = df[col].fillna(
        df[col].mode()[0]
    )

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

# =========================
# DATA TYPE VALIDATION
# =========================

numeric_columns = [
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("\nData Types:")
print(df.dtypes)

# =========================
# CREATE OUTPUT FOLDER
# =========================

import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================
# STUDENT PERFORMANCE
# DISTRIBUTION
# =========================

plt.figure(figsize=(7, 5))

sns.countplot(
    x="Class",
    data=df
)

plt.title("Distribution of Student Performance")
plt.xlabel("Performance Category")
plt.ylabel("Number of Students")

plt.savefig(
    "outputs/performance_distribution.png",
    bbox_inches="tight"
)

plt.show()

# =========================
# ATTENDANCE VS PERFORMANCE
# =========================

plt.figure(figsize=(8, 5))

sns.countplot(
    x="StudentAbsenceDays",
    hue="Class",
    data=df
)

plt.title("Attendance vs Performance")

plt.savefig(
    "outputs/attendance_vs_performance.png",
    bbox_inches="tight"
)

plt.show()

# =========================
# PARENT PARTICIPATION
# =========================

plt.figure(figsize=(8, 5))

sns.countplot(
    x="ParentAnsweringSurvey",
    hue="Class",
    data=df
)

plt.title("Parent Participation vs Performance")

plt.savefig(
    "outputs/parent_participation.png",
    bbox_inches="tight"
)

plt.show()

# =========================
# RAISED HANDS VS CLASS
# =========================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Class",
    y="raisedhands",
    data=df
)

plt.title("Raised Hands vs Performance")

plt.savefig(
    "outputs/raisedhands_vs_class.png",
    bbox_inches="tight"
)

plt.show()

# =========================
# DISCUSSION ACTIVITY
# =========================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Class",
    y="Discussion",
    data=df
)

plt.title("Discussion Activity vs Performance")

plt.savefig(
    "outputs/discussion_vs_class.png",
    bbox_inches="tight"
)

plt.show()

# =========================
# RESOURCE USAGE
# =========================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Class",
    y="VisITedResources",
    data=df
)

plt.title("Learning Resource Usage")

plt.savefig(
    "outputs/resource_usage.png",
    bbox_inches="tight"
)

plt.show()

# =========================
# CORRELATION HEATMAP
# =========================

df_corr = df.copy()

for col in df_corr.select_dtypes(
    include="object"
).columns:
    df_corr[col] = (
        df_corr[col]
        .astype("category")
        .cat.codes
    )

plt.figure(figsize=(12, 8))

sns.heatmap(
    df_corr.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "outputs/correlation_heatmap.png",
    bbox_inches="tight"
)

plt.show()

# =========================
# FEATURE ENGINEERING
# =========================

label_encoder = LabelEncoder()

df["Class"] = label_encoder.fit_transform(
    df["Class"]
)

X = df.drop(
    "Class",
    axis=1
)

X = pd.get_dummies(
    X,
    drop_first=True
)

y = df["Class"]

print("\nFeature Matrix Shape:", X.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# =========================
# LOGISTIC REGRESSION
# =========================

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train,
    y_train
)

lr_pred = lr_model.predict(
    X_test
)

# =========================
# DECISION TREE
# =========================

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

dt_pred = dt_model.predict(
    X_test
)

# =========================
# RANDOM FOREST
# =========================

rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

# =========================
# MODEL EVALUATION
# =========================

models = {
    "Logistic Regression": lr_pred,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred
}

results = []

for name, pred in models.items():

    accuracy = accuracy_score(
        y_test,
        pred
    )

    results.append(
        [name, accuracy]
    )

    print("\n")
    print("=" * 60)
    print(name.upper())
    print("=" * 60)

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print("\nConfusion Matrix")
    print(
        confusion_matrix(
            y_test,
            pred
        )
    )

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            pred
        )
    )

# =========================
# MODEL COMPARISON
# =========================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

# =========================
# ACCURACY BAR CHART
# =========================

plt.figure(figsize=(8, 5))

sns.barplot(
    x="Accuracy",
    y="Model",
    data=results_df
)

plt.title(
    "Model Accuracy Comparison"
)

plt.savefig(
    "outputs/model_accuracy_comparison.png",
    bbox_inches="tight"
)

plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

top_features = (
    importance
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))

top_features.plot(
    kind="barh"
)

plt.title(
    "Top 10 Most Important Features"
)

plt.savefig(
    "outputs/feature_importance.png",
    bbox_inches="tight"
)

plt.show()

# =========================
# SAVE TRAINED MODEL
# =========================

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

print("\n")
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("Saved Files:")
print("- models/random_forest.pkl")
print("- models/scaler.pkl")
print("- models/label_encoder.pkl")

print("\nCharts saved inside outputs/ folder")
print("Project completed successfully.")