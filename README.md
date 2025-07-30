# 📊 Online Course Completion - Machine Learning Project

This project predicts whether a student will complete an online course (`completed_course`) using demographic, behavioral, and engagement data. The task is binary classification (0 = not completed, 1 = completed).

---

## 🎯 Problem Statement

With the surge in online learning, understanding learner behavior is crucial. This project aims to:

- 📌 Identify key features influencing course completion.
- 🤖 Predict course completion using machine learning models.
- 📊 Interpret patterns using visualizations and feature importance.

---

## 🛠️ Tech Stack

| Component       | Technology Used                                           |
|-----------------|-----------------------------------------------------------|
| Language        | Python 3.x                                                |
| Environment     | Ubuntu Linux                                              |
| Versioning      | `pyenv`, `poetry` for dependency management               |
| IDE             | Jupyter Notebook (Browser: Firefox)                       |
| Libraries       | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost` |

---

## 📁 Dataset Overview

- **Format:** CSV
- **Size:** ~500,000 rows
- **Target Column:** `completed_course` (binary: 0 = not completed, 1 = completed)

### 🔍 Sample Features

- **Demographics:** `age`, `continent`, `country`, `education_level`
- **Engagement:** `hours_per_week`, `num_logins_last_month`, `videos_watched_pct`
- **Course Activity:** `assignments_submitted`, `discussion_posts`
- **User Traits:** `is_working_professional`, `preferred_device`, `num_siblings`, `has_pet`
- **Dropped (Irrelevant):** `favorite_color`, `birth_month`, `height_cm`, `weight_kg`

---

## 🧼 Preprocessing Steps

### 1️⃣ Feature Selection

- ❌ Dropped non-informative columns: `favorite_color`, `birth_month`, `height_cm`, `weight_kg`
- 🌐 Removed `country` to avoid high cardinality in encoding

### 2️⃣ Handling Missing Data

- 🧩 Imputed or dropped features with high nulls:
  - `education_level`
  - `preferred_device`
  - `videos_watched_pct`

### 3️⃣ Encoding Categorical Variables

- 🎯 Applied **One-Hot Encoding** to:
  - `continent`, `education_level`, `preferred_device`

### 4️⃣ Feature Scaling

- ⚖️ Used `StandardScaler` for numerical features:
  - `age`, `hours_per_week`, `num_logins_last_month`, `assignments_submitted`, `discussion_posts`, `num_siblings`

---

## ⚖️ Class Imbalance Handling

- **Target Distribution:**
  - `0` (Non-completers): 375,089
  - `1` (Completers): 124,911

✅ Addressed via:

- `class_weight='balanced'` for Logistic Regression and Random Forest
- `scale_pos_weight` for XGBoost

---

## 🤖 Machine Learning Models

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.8330   | 0.6212    | 0.8496 | 0.7177   |
| Random Forest       | 0.9925   | 0.9964    | 0.9735 | 0.9848   |
| XGBoost             | 0.9866   | 0.9597    | 0.9880 | 0.9736   |

---

## 📊 Visualizations

- ✅ Confusion Matrix for each model
- 📉 Feature Importance bar plots (Random Forest, XGBoost)
- 🧮 Histograms, KDE plots for EDA
- 🔥 Correlation heatmap to detect multicollinearity

---

## 💡 Key Findings

- ✅ **Random Forest and XGBoost** outperformed Logistic Regression significantly.
- 🔝 Most predictive features:
  - `assignments_submitted`
  - `videos_watched_pct`
  - `hours_per_week`
- ⚖️ Imbalance handled successfully through model parameters.

---

## 🚀 How to Run the Project

### 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/Mohammedhasnaine/online-course-completion.git
cd online-course-completion
```

### ⚙️ Step 2: Install Dependencies

```bash
poetry install
poetry shell
```

### 📓 Step 3: Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open the notebooks in order from the `notebooks/` folder:

1. `1_data_loading_and_eda.ipynb`
2. `2_preprocessing.ipynb`
3. `3_course_completion_model.ipynb`

---

## 👤 Author

**Name:** Mohammed Hasnaine  
**Course:** B.E. Final Year  
**Project Type:** ML Classification Project  
**GitHub:** [@Mohammedhasnaine](https://github.com/Mohammedhasnaine)
