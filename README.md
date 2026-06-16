### Phishing Website Detection Through Logistic Regression

A machine learning project that classifies websites as **phishing** or **legitimate** using URL-based features and a Logistic Regression classifier.

This project was completed as undergraduate research at **Purdue University Indianapolis** and accompanies the research paper **Using URL Characteristics to Detect Phishing Websites Through Logistic Regression**.

---

## Project Overview

This project analyzes the **UCI Phishing Websites Dataset** to identify the URL characteristics most indicative of phishing attacks. The model was developed in Python using **scikit-learn** and demonstrates an end-to-end machine learning workflow, including:

* Data preprocessing
* Feature scaling
* Logistic Regression model training
* Hyperparameter tuning with GridSearchCV
* Cross-validation
* Model evaluation
* Feature importance analysis
* Data visualization

---

## Dataset

This project uses the **Phishing Websites Dataset** from the **UCI Machine Learning Repository**.

**Download the dataset here:**

https://archive.ics.uci.edu/dataset/327/phishing+websites

After downloading the dataset:

1. Extract the downloaded files.
2. Place `Training Dataset.arff` in the same directory as `phishing_detection.py`.
3. Run the Python script.

---

## Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* matplotlib
* liac-arff

---

## Results

The Logistic Regression model achieved:

* **Accuracy:** 92.8%
* **ROC-AUC Score:** 0.98

The model identified several of the strongest indicators of phishing websites, including:

* URL of Anchor
* SSL Final State
* Prefix Suffix
* Web Traffic
* SFH (Server Form Handler)

---

## Repository Contents

* `phishing_detection.py` – Python implementation of the machine learning model
* `phishing_detection_research_paper.pdf` – Undergraduate research paper
* `images/` – Generated visualizations, including the ROC curve, confusion matrix, and feature importance chart
**
