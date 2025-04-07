# End-to-End Machine Learning Process for Predicting Response Time

## 1. Understanding the Dataset

**Dataset Used:** 311 Service Requests

**Target Variable:** `ResponseTime`

### Features:
- **Numerical:** Latitude/Longitude transformations, Date/Time components
- **Categorical:** Agency, Complaint Type, Location details

---

## 2. Data Cleaning & Transformation

### Handling Missing Values:
- Dropped rows with missing critical numerical data

### Feature Engineering:
- Extracted sin/cos transformations for latitude and longitude
- Derived time-of-day and day-of-week features

### Encoding Categorical Variables:
- Used Label Encoding for categorical columns

### Scaling Numerical Data:
- Did not scale binary features (e.g., Holiday Created Date)
- did not scale other features

---

## 3. Splitting Data

### Train/Test Split:
- 80% Training / 20% Testing
- Used `stratify=['Open Data Channel Type']` to maintain class balance

---

## 4. Model Selection

### Compared Multiple Models:
- Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR, Gaussian Process

### Hyperparameter Tuning:
- Used `GridSearchCV` to find the best hyperparameters

---

## 5. Training the Model

### Selected Model:
- xgboost

### Hyperparameter Tuning:
- Grid-searched parameters like `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `eval_metric`

### Best Parameters:
- params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 9,
        'learning_rate': 0.3,
        'n_estimators': 300,
        'subsample': 1.0}

### Training on CPU

--- 

## 6. Evaluation & Fine-Tuning

### Metrics Used:
- Mean Squared Error (MSE)
- Roor Meean Squared Error (RMSE)
- R² Score (for explained variance)
- Mean Absolute Error (MAE) (for accuracy assessment)

### Results:
- **MSE Score:** `0.11`
- **RMSE Score validation:** `0.33`
- **R² Score:** `0.96`
- **MAE:** `0.17`

### Fine-Tuning:
- Applied feature importance analysis to refine features
- Increased training data for improved generalization

---

## 7. Final Model Deployment

### Saved Best Model:
- `joblib.dump(best_model, 'best_model.pkl')`


---

## Installation

To set up the environment and run the pipeline, you need the following dependencies:

```bash
pip install -r requirements.txt
