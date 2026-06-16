import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import arff
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

import matplotlib.pyplot as plt

#load ARFF 
arff_file = "Training Dataset.arff"  

with open(arff_file, "r") as f:
    dataset = arff.load(f)

df = pd.DataFrame(dataset["data"], columns=[attr[0] for attr in dataset["attributes"]])

print("Data shape:", df.shape)
print("\nColumns:", list(df.columns))

#define x and y
target_col = "Result"
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

print("\nClass distribution:\n", y.value_counts())

#train split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

#scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#base logistic regression model
log_reg = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
)

log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
print("\nConfusion Matrix (rows=true, cols=pred, order=[-1,0,1]):\n", cm)

print("\nClassification report (base model):\n")
print(classification_report(y_test, y_pred, digits=4))

precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

print("Macro Precision:", precision_macro)
print("Macro Recall:   ", recall_macro)
print("Macro F1:       ", f1_macro)

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
}

log_reg_base = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
)

grid = GridSearchCV(
    estimator=log_reg_base,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
)

grid.fit(X_train_scaled, y_train)

print("\nBest params from GridSearchCV:", grid.best_params_)
print("Best CV f1_macro:", grid.best_score_)

best_model = grid.best_estimator_

cv_scores = cross_val_score(
    best_model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
)

print("\nCross-validation f1_macro scores:", cv_scores)
print("Mean CV f1_macro:", cv_scores.mean())

best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

print("\nClassification report (best tuned model):\n")
print(classification_report(y_test, y_pred_best, digits=4))

#feature importance
feature_names = X.columns
coef_matrix = best_model.coef_
importance = np.mean(np.abs(coef_matrix), axis=0)

importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": importance}
).sort_values(by="importance", ascending=False)

print("\nTop 15 features by importance:\n")
print(importance_df.head(15))

#feature importance bar chart
plt.figure(figsize=(10, 6))
top_n = 15
subset = importance_df.head(top_n)[::-1] 
plt.barh(subset["feature"], subset["importance"])
plt.xlabel("Average |Coefficient| (importance)")
plt.title("Top Features for Phishing Detection (Logistic Regression)")
plt.tight_layout()
plt.show()

#confusion matrix heatmap
cm_best = confusion_matrix(y_test, y_pred_best, labels=[-1, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_best,
    display_labels=["Legitimate (-1)", "Phishing (1)"],
)

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – Tuned Logistic Regression")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.show()

# roc curve

#convert labels to 0/1 for ROC (0 = legitimate -1, 1 = phishing 1)
y_test_binary = (y_test == 1).astype(int)

#get predicted probabilities for the positive class (phishing = 1)
proba = best_model.predict_proba(X_test_scaled)
#find index of class "1"
class_index_phishing = list(best_model.classes_).index(1)
phishing_proba = proba[:, class_index_phishing]

fpr, tpr, thresholds = roc_curve(y_test_binary, phishing_proba)
auc_score = roc_auc_score(y_test_binary, phishing_proba)

print("\nROC AUC score (phishing vs legitimate):", auc_score)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Tuned Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
