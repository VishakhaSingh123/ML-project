# 📊 Student Performance Indicator

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey?style=flat-square&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Pipeline-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)
![Render](https://img.shields.io/badge/Deployed-Render-purple?style=flat-square)

> An end-to-end machine learning web application that predicts a student's math score based on demographic and academic factors — featuring automated multi-model benchmarking, hyperparameter tuning, and a live Flask deployment.

🔗 **Live Demo:** [student-performance-indicator-2sy9.onrender.com](https://student-performance-indicator-2sy9.onrender.com/predictdata)

---

## 📌 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Model Benchmarking Results](#model-benchmarking-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Deployment](#deployment)
- [Limitations & Future Work](#limitations--future-work)

---

## 🌐 Overview

This project builds a production-style ML pipeline to predict student math scores based on:
- Gender, race/ethnicity, parental education level
- Lunch type, test preparation course completion
- Reading and writing scores

Rather than training a single model, the system **automatically benchmarks 7 algorithms**, tunes hyperparameters for each, and selects the best performer — then serves predictions via a Flask web interface.

---

## 🏫 Problem Statement

Understanding which factors influence student academic performance can help educators and institutions make better interventions. This project answers:

> *"Given a student's background and other scores, what math score can we predict — and which ML model does it best?"*

---

## 🏗️ System Architecture

```
Raw Data (CSV)
      │
      ▼
Data Ingestion
(Train/Test Split)
      │
      ▼
Data Transformation Pipeline
├── Numerical: Median Imputation → Standard Scaling
└── Categorical: Mode Imputation → One-Hot Encoding → Standard Scaling
      │
      ▼
Model Training
(7 Algorithms + Hyperparameter Tuning)
      │
      ▼
Best Model Selection (R2 Score)
      │
      ▼
Serialized Model (.pkl)
      │
      ▼
Flask Web App → Live Prediction
```

---

## ✨ Key Features

- **🔄 Modular Pipeline** — separate ingestion, transformation, and training components
- **🤖 7 Model Benchmarking** — automatically compares and selects best algorithm
- **⚙️ Hyperparameter Tuning** — GridSearchCV across all models
- **💾 Model Serialization** — best model saved as `.pkl` for inference
- **🌐 Live Web App** — Flask interface with form input + real-time prediction
- **📝 Custom Logging** — structured logs throughout the pipeline
- **⚠️ Custom Exception Handling** — production-style error management
- **🌙 Dark Mode** — UI toggle for dark/light mode

---

## 📊 Model Benchmarking Results

All 7 models trained and evaluated on the same test set with hyperparameter tuning:

| Model | R2 Score |
|---|---|
| **Linear Regression** ← Selected | **0.880** |
| Gradient Boosting | 0.871 |
| CatBoost Regressor | 0.852 |
| Random Forest | 0.853 |
| XGBoost | 0.849 |
| AdaBoost | 0.850 |
| Decision Tree | 0.737 |

> **Best Model: Linear Regression — R2 Score: 0.88**
>
> The model explains **88% of the variance** in student math scores. Linear Regression outperformed all ensemble methods on this dataset, suggesting strong linear relationships between the features and target variable.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| ML Framework | Scikit-learn |
| Boosting Models | XGBoost, CatBoost, AdaBoost |
| Data Processing | Pandas, NumPy |
| Web Framework | Flask |
| Serialization | Pickle / Dill |
| Deployment | Render (Free Tier) |
| Dataset | [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) |

---

## 📁 Project Structure

```
Student-performance-indicator/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Train/test split pipeline
│   │   ├── data_transformation.py   # Preprocessing pipelines
│   │   └── model_trainer.py         # 7-model benchmarking + selection
│   ├── pipeline/
│   │   └── predict_pipeline.py      # Inference pipeline using saved model
│   ├── exception.py                 # Custom exception handling
│   ├── logger.py                    # Structured logging
│   └── utils.py                     # Helper functions
│
├── artifacts/
│   ├── model.pkl                    # Saved best model
│   └── preprocessor.pkl             # Saved preprocessing pipeline
│
├── templates/
│   ├── index.html                   # Landing page
│   └── home.html                    # Prediction form + results
│
├── notebook/data/                   # EDA notebooks
├── screenshots/                     # App screenshots
├── app.py                           # Flask application
├── Procfile                         # Render deployment config
├── requirements.txt
└── setup.py
```

---

## 🚀 How to Run Locally

### Prerequisites
```bash
Python 3.8+
```

### Setup
```bash
# Clone the repository
git clone https://github.com/VishakhaSingh123/Student-performance-indicator
cd Student-performance-indicator

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run
```bash
python app.py
```

Open browser at: `http://127.0.0.1:5000/predictdata`

---

## ☁️ Deployment

Deployed on **Render** (Free Tier) using Gunicorn as the WSGI server.

**Live URL:** https://student-performance-indicator-2sy9.onrender.com/predictdata

> ⚠️ Note: Free tier instances spin down after inactivity — first load may take 30-50 seconds to wake up.

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Dataset limited to one school district — may not generalise broadly
- Linear Regression selected as best model — more complex datasets may benefit from ensemble methods
- Free tier deployment spins down with inactivity

### Planned Improvements
- [ ] Add more features — attendance, extracurricular activities
- [ ] Implement cross-validation for more robust model evaluation
- [ ] Add prediction confidence intervals
- [ ] Upgrade to persistent deployment
- [ ] Add data visualisation dashboard

---

## 📸 Screenshots

### Prediction Form
![Prediction Form](screenshots/prediction_form.png)

---

## 🎓 Conclusion

This project demonstrates a complete ML engineering workflow — from raw data through transformation pipelines, automated model selection, and live deployment. The system achieves an **R2 score of 0.88**, correctly explaining 88% of variance in student math scores, with Linear Regression outperforming 6 other algorithms including XGBoost and CatBoost.

---

## 👩‍💻 Author

**Vishakha Singh**
B.Tech Computer Science, Manipal University Jaipur (2027)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/vishakha-singh-03309b24a)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/VishakhaSingh123)

---

*If you found this project useful, please consider giving it a ⭐*
