# 💻 Laptop Price Prediction System

End-to-end ML project: EDA → Feature Engineering → Outlier Removal → Model Training → GridSearchCV → Streamlit App.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [ML Pipeline Summary](#ml-pipeline-summary)
- [Final Model Performance](#final-model-performance)
- [Model Comparison](#model-comparison)
- [Top Feature Importances](#top-feature-importances)
- [Streamlit App Features](#streamlit-app-features)

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LapSense.git
   cd LapSense
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   - Place `laptop_price.csv` in the `data/` directory.
   - (Note: Dataset not included in repo due to size; obtain from original source)

---

## Usage

### Step 1: Train the Model
Run the ML pipeline to train and save the model:
```bash
python laptopPricePredictorAPI.py
```
This script performs:
- Data loading and exploratory data analysis (EDA)
- Feature engineering and preprocessing
- Outlier removal
- Model training and comparison
- Hyperparameter tuning with GridSearchCV
- Saves the best model to `model/laptop_price_model.pkl`

### Step 2: Launch the Web App
Start the Streamlit application:
```bash
streamlit run app.py
```
- Open your browser to `http://localhost:8501`
- Interact with the price prediction interface

### Additional Commands
- View EDA plots: Check the `eda_plots/` directory after training
- Retrain model: Re-run `python laptopPricePredictorAPI.py` to update the model

---

## Features

- **End-to-End ML Pipeline**: Complete workflow from data ingestion to model deployment
- **Comprehensive EDA**: 9 automated plots for data visualization and insights
- **Advanced Feature Engineering**: Custom transformations for laptop specifications
- **Outlier Detection**: Robust removal using IQR and Z-score methods
- **Model Comparison**: Evaluation of 10 regression algorithms
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Interactive Web App**: User-friendly Streamlit interface for price prediction
- **Multi-Currency Support**: Price predictions in EUR, INR, and USD
- **Model Interpretability**: Feature importance analysis and metrics dashboard

---

## Model Details

### Dataset
- **Source**: Laptop price dataset
- **Size**: 1,303 rows, 13 columns
- **Target Variable**: `Price_euros` (continuous regression)

### Preprocessing
- **Feature Engineering**: RAM/Weight parsing, PPI calculation, CPU/GPU brand extraction, OS simplification, HDD/SSD flags
- **Outlier Removal**: IQR + Z-score intersection (106 rows removed, 8.1%)
- **Train/Test Split**: 80/20 ratio (957 train, 240 test samples)
- **Encoding**: OneHotEncoder for categorical features, StandardScaler for numerical features

### Algorithms Compared
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)

### Best Model: Gradient Boosting (Tuned)
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Performance Metrics**:
  - R²: 0.7898
  - Adjusted R²: 0.7787
  - MAE: €172.52
  - RMSE: €250.81

---

## 📁 Project Structure

```
laptop_price_ml/
│
├── data/
│   └── laptop_price.csv          ← raw dataset (1,303 rows)
│
├── model/
│   └── laptop_price_model.pkl    ← trained model bundle (auto-generated)
│
├── eda_plots/                    ← all EDA & evaluation charts (auto-generated)
│   ├── 01_price_distribution.png
│   ├── 02_price_by_company.png
│   ├── 03_price_by_type.png
│   ├── 04_categorical_frequencies.png
│   ├── 05_numeric_vs_price.png
│   ├── 06_outlier_boxplots.png
│   ├── 07_model_comparison.png
│   ├── 08_final_model_evaluation.png
│   └── 09_feature_importances.png
│
├── laptopPricePredictorAPI.py    ← full ML pipeline (train & save model)
├── app.py                        ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## 📊 ML Pipeline Summary

| Stage | Details |
|---|---|
| **Dataset** | 1,303 laptops, 13 raw columns |
| **Target** | `Price_euros` (continuous regression) |
| **EDA** | Distributions, correlations, company/type breakdowns |
| **Feature Engineering** | RAM/Weight parsing, PPI calc, CPU/GPU brand, OS simplification, HDD/SSD flags |
| **Outlier Removal** | IQR + Z-score (both must flag) — 106 rows removed (8.1%) |
| **Train/Test Split** | 80/20 (957 train, 240 test) |
| **Preprocessing** | OneHotEncoder (categorical) + StandardScaler (numerical) via ColumnTransformer |
| **Models Compared** | Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, KNN, SVR |
| **Tuning** | GridSearchCV (5-fold) on RandomForest + GradientBoosting |
| **Best Model** | GradientBoosting (tuned) |

---

## 🏆 Final Model Performance

| Metric | Value |
|---|---|
| **R²** | 0.7898 |
| **Adjusted R²** | 0.7787 |
| **MAE** | € 172.52 |
| **RMSE** | € 250.81 |

---

## 📈 Model Comparison (all 10 models)

| Model | MAE | RMSE | R² | CV-R² |
|---|---|---|---|---|
| Random Forest | 182.48 | 259.60 | 0.7748 | 0.7855 |
| **Gradient Boosting** *(tuned winner)* | **172.52** | **250.81** | **0.7898** | **0.8033** |
| Extra Trees | 186.95 | 270.57 | 0.7554 | 0.7414 |
| Decision Tree | 207.85 | 294.63 | 0.7099 | 0.6788 |
| Ridge Regression | 222.99 | 297.35 | 0.7046 | 0.7097 |
| Lasso Regression | 224.58 | 301.08 | 0.6971 | 0.7047 |
| SVR | 217.02 | 326.52 | 0.6438 | 0.6703 |
| KNN | 229.62 | 336.00 | 0.6228 | 0.6791 |
| AdaBoost | 262.43 | 336.68 | 0.6212 | 0.6010 |
| ElasticNet | 257.12 | 349.79 | 0.5912 | 0.6187 |

---

## 🔑 Top Feature Importances

1. **RAM** (0.496) — single most important predictor
2. **Weight** (0.116) — proxy for build quality / category
3. **TypeName: Notebook** (0.111) — entry-level price signal
4. **PPI** (0.072) — display quality
5. **SSD** (0.053) — storage type matters
6. **TypeName: Workstation** (0.023)
7. **IPS display** (0.016)

---

## 🌐 Streamlit App Features

- Select brand, type, CPU/GPU, RAM, storage, display specs
- Live € / ₹ / $ price prediction
- Adjustable currency conversion rates in sidebar
- Model metrics dashboard in sidebar
- Spec summary after prediction
- Confidence interval hint based on MAE

---

*Built with Python · Scikit-learn · Streamlit*
