# Online Course Completion Prediction

This machine learning project predicts whether a student will complete an online course (`completed_course`) based on their demographics, activity, and preferences. The target is binary (0 or 1), making this a **classification problem**.



## Problem Statement

With increasing online course enrollments, it's essential to understand learner behavior. This project aims to:
- Identify factors influencing course completion.
- Predict future course completions using machine learning.
- Provide insights through exploratory data analysis (EDA) and model interpretation.



## Tech Stack

| Component       | Technology Used                                           |
|---------------- |-----------------------------------------------------------|
| Language        | Python 3.x                                                |
| Environment     | Ubuntu Linux                                              |
| Versioning      | `pyenv`, `Poetry` for dependency management               |
| IDE             | Jupyter Notebook (browser: Firefox)                       |
| Libraries       | pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost |

---

## Project Structure

```bash
online-course-completion/
├── notebooks/
│   ├── 1_data_loading_and_eda.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_course_completion_model.ipynb
│ 
├── data/
│   └── online_course_completion.csv
├── README.md
├── pyproject.toml
└── poetry.lock

---

##Data

- **Format:** CSV  
- **Size:** ~500,000 rows  
- **Target Column:** `completed_course` (0 = Not Completed, 1 = Completed)

### 🧾 Sample Features:
- Demographics: `age`, `continent`, `country`, `education_level`
- Engagement: `hours_per_week`, `num_logins_last_month`, `videos_watched_pct`
- Activity: `assignments_submitted`, `discussion_posts`
- Others: `is_working_professional`, `preferred_device`, `num_siblings`, `has_pet`, `height_cm`, `weight_kg`, `favorite_color`, `birth_month`

---

## 🧼 Preprocessing Steps

### 🔍 Feature Selection
- Dropped irrelevant features like:  
  `favorite_color`, `birth_month`, `height_cm`, `weight_kg`, etc.
- Removed `country` to avoid high-dimensional one-hot encoding.
  - (Optional: Target encoding could be considered if included)

### ❌ Handling Missing Data
- Columns with high missing values:  
  `education_level`, `preferred_device`, `videos_watched_pct`
- Applied appropriate imputation strategies or dropped if not salvageable.

### 🔤 Encoding Categorical Variables
- Applied **One-Hot Encoding** to:
  - `continent`, `education_level`, `preferred_device`, `is_working_professional`

### 📏 Feature Scaling
- Used **`StandardScaler`** on numerical columns:
  - `age`, `hours_per_week`, `num_logins_last_month`, `assignments_submitted`, `discussion_posts`, etc.

---



