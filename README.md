# Customer Segmentation via PCA & k-Means Clustering
*Computational Linear Algebra for Large Scale Problems — Politecnico di Torino (M.Sc. Data Science and Engineering)*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

## 📌 Project Overview
This project simulates a **real-world customer profiling and market segmentation scenario** for business decision-making. Using high-dimensional survey data, the goal is to reduce dimensionality to key latent factors and identify interpretable customer personas using unsupervised learning techniques.

---

## 📊 Dataset: Young People Survey (YPS)
The analysis is based on the **Young People Survey dataset** (674 participants, 150 features). The questions in the survey concern eight different categories of questions:
- _Music preferences_ (19 items)
- _Movie preferences_ (12 items)
- _Hobbies & interests_ (32 items)
- _Phobias_ (10 items)
- _Health habits_ (3 items)
- _Personality traits, views on life, & opinions_ (57 items)
- _Spending habits_ (7 items)
- _Demographics_ (10 items)

Data consists primarily of 5-point Likert ratings (1–5) and ordinal categorical features.

Specifically, this analysis is conducted on a tailored subset of the dataset (according to the student ID):
- **Sample Size ($N$):** 506 respondents (75% stratified random sample)
- **Features ($D$):** 93 behavioral & psychological items.
- **Demographics & Profile Labels:** 10 target variables (Age, Gender, Education, Childhood Environment, etc.) used strictly for post-clustering profiling.

---

## 🔬 Methodology & Pipeline

### 1. Encoding & Scaling Strategy
- **Ordinal Encoding:** Applied to categorical features with intrinsic ordering. This avoids the curse of dimensionality and sparsity introduced by One-Hot Encoding while preserving distance metrics for PCA and k-Means.
- **Min-Max Scaling ($[0, 1]$):** Normalizes heterogeneous scales (1–5 Likert ratings vs. 0–3 encoded categories), preventing high-variance features from dominating principal component directions.

### 2. Dimensionality Reduction (PCA)
- SVD decomposition of the scaled feature matrix.
- Evaluation of the **Scree Plot** and cumulative explained variance.
- Selection of the leading Principal Components prioritizing **semantic interpretability** for managerial actionability over brute-force variance preservation.

### 3. Clustering & Customer Personas
- Application of **k-Means** on the reduced principal component space.
- Validation using **Silhouette Score** and **Inertia Elbow Curves**.
- Characterization of resulting clusters across music/movie tastes, spending behavior, and demographic traits.

---

## 🛠️ Tech Stack & Requirements
- **Language:** Python 3.9+
- **Core Libraries:** `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`

### Quick Start:
# Clone the repository
git clone https://github.com/giuliadesantis/Customer_segmentation_pca_kmeans.git

# Navigate into the project folder
cd Customer_segmentation_pca_kmeans

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook notebooks/HWpca2526_DeSantis.ipynb
