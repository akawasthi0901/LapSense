# 💻 Laptop Price Prediction System

End-to-end ML project: EDA → Feature Engineering → Outlier Removal → Model Training → GridSearchCV → Streamlit App.

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

## ⚙️ Setup

```bash
# 1. Clone / copy the project folder
cd laptop_price_ml

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the model
```bash
python laptopPricePredictorAPI.py
```
This will:
- Load and explore the dataset (EDA + 9 plots saved to `eda_plots/`)
- Apply feature engineering
- Remove outliers using IQR ∩ Z-score (conservative approach)
- Compare 10 regression models with 5-fold cross-validation
- Run GridSearchCV on the top 2 models
- Save the best model to `model/laptop_price_model.pkl`
- Print a final metrics summary in the terminal

### Step 2 — Launch the Streamlit app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

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
