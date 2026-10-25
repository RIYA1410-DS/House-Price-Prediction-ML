
# House Price Prediction using Machine Learning

## Project Overview
This project predicts residential property prices using machine learning.
It includes data validation, exploratory data analysis, model comparison,
hyperparameter tuning, automated tests, and an interactive Streamlit web app.

## Features
- Data validation and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Comparison of Linear Regression, Decision Tree, and Random Forest
- Random Forest hyperparameter tuning using GridSearchCV
- Model evaluation using MAE, RMSE, R², and MAPE
- Interactive house price prediction interface
- Automated tests using pytest

## Dataset
The dataset contains 300 property records and 8 columns.

Input features:
- Area
- Bedrooms
- Bathrooms
- Age
- Location
- Property_Type

Target variable: Price

## Model Performance
Selected model: Tuned Random Forest

Test-set results:
- MAE: ₹1,472,445.29
- RMSE: ₹1,993,538.12
- R²: 0.9721
- MAPE: 6.79%

These results are based on a held-out test set of 60 records.
Performance on new real-world properties may differ.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit
- Pytest

## Project Structure
```text
House_Price_Prediction_ML/
├── app.py
├── test_prediction.py
├── requirements.txt
├── README.md
├── data/
│   └── house_prices.csv
├── models/
│   ├── house_price_model_v1.joblib
│   └── model_metadata_v1.json
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── predict.py
├── scripts/
│   └── explore_data.py
└── docs/
    ├── eda_plots/
    └── model_comparison.csv
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd House_Price_Prediction_ML
```

### 2. Create and activate a virtual environment

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Run the web app
```bash
python -m streamlit run app.py
```

### 5. Run automated tests
```bash
python -m pytest test_prediction.py -v

## Application Screenshot

![House Price Prediction App](screenshots/house-price-app.png)
```

## Limitations
- The dataset contains only 300 records.
- Predictions depend on the patterns represented in the training data.
- Estimates are not guaranteed real estate market valuations.

