import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def nmec_metrics(y_test, y_pred, n_features=None):
    '''
    Calculate and display the evaluation metrics for the model's predictions
    '''
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Validation Metrics:')
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")

def modelData(df, test_size, random_state, stratify):
    '''
    Train an XGBoost regression model on the provided dataset, monitoring both training and validation RMSE.
    '''

    numerical_cols = ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'time_of_day Created Date', 
                      'dayOfWeek Created Date', 'month Created Date', 'weekday Created Date', 
                      'hour Created Date', 'Holiday Created Date', 'time_of_day Closed Date', 'dayOfWeek Closed Date', 
                      'month Closed Date', 'weekday Closed Date', 'hour Closed Date']

    categorical_cols = ['Agency', 'Complaint Type', 'Street Name', 'City', 'Borough', 'Open Data Channel Type']
    features = numerical_cols + categorical_cols

    X = df[features].copy()
    y = df['logResponseTime'].copy()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    joblib.dump(label_encoders, 'label_encoders_xgboost_raw_plot.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)

    # Track both training and validation RMSE values
    train_rmse_values = []
    val_rmse_values = []

    # Prepare DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 9,
        'learning_rate': 0.3,
        'n_estimators': 300,
        'subsample': 1.0,
    }
    
    # Watchlist to monitor training and validation RMSE
    watchlist = [(dtrain, 'train'), (dtest, 'validation_0')]
    
    # Train the model with custom callback to track RMSE
    evals_result = {}
    bst = xgb.train(params, dtrain, num_boost_round=300,
                    evals=watchlist,
                    evals_result=evals_result,
                    callbacks=[xgb.callback.EarlyStopping(rounds=10)])

    # Extract training and validation RMSE values from evals_result
    train_rmse_values = evals_result['train']['rmse']
    val_rmse_values = evals_result['validation_0']['rmse']
    
    # Plot RMSE for both training and validation sets per iteration
    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse_values, marker='o', linestyle='-', color='blue', label='Train RMSE')
    plt.plot(val_rmse_values, marker='o', linestyle='-', color='red', label='Validation RMSE')
    plt.title('Train vs Validation RMSE per Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Predict and evaluate
    y_pred = bst.predict(dtest)
    nmec_metrics(y_test, y_pred, len(features))

    # Save model
    joblib.dump(bst, 'best_model_xgboost_raw_plot.pkl')

    return X_train, X_test, y_train, y_test, y_pred, bst


def modelDataWithGridSearch(df, test_size, random_state, stratify):

    label_encoders = {}

    numerical_cols = ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'time_of_day Created Date', 
                    'dayOfWeek Created Date', 'month Created Date', 'weekday Created Date', 
                    'hour Created Date', 'Holiday Created Date', 'time_of_day Closed Date', 'dayOfWeek Closed Date', 'month Closed Date', 
                    'weekday Closed Date', 'hour Closed Date']

    categorical_cols = ['Agency', 'Complaint Type', 'Street Name', 'City', 'Borough', 'Open Data Channel Type']

    features = numerical_cols + categorical_cols
    
    X = df[features].copy()
    
    y = df['logResponseTime'].copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X[col])
        label_encoders[col] = le
        # Transform the categorical column in the batch
        X[col] = le.transform(X[col])
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)

    # Define models
    linear_model = LinearRegression()
    ridge_model = Ridge()
    lasso_model = Lasso()
    elasticnet_model = ElasticNet()
    decision_tree_model = DecisionTreeRegressor()
    random_forest_model = RandomForestRegressor()
    gradient_boosting_model = GradientBoostingRegressor()
    svr_model = SVR()
    kernel = 1.0 * RBF()
    gaussian_process_model = GaussianProcessRegressor()
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

    # Define hyperparameter grids for each model
    param_grids = {
        'Linear Regression': {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        },
        'Ridge Regression': {
            'alpha': [0.1, 0.5, 1.0],
            'fit_intercept': [True, False],
            'normalize': [True, False]
        },
        'Lasso Regression': {
            'alpha': [0.1, 0.5, 1.0],
            'fit_intercept': [True, False],
            'normalize': [True, False]
        },
        'ElasticNet Regression': {
            'alpha': [0.1, 0.5, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'fit_intercept': [True, False],
            'normalize': [True, False]
        },
        'Decision Tree Regression': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest Regression': {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting Regression': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'SVR': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.1, 0.2, 0.3]
        },
        'Gaussian Process Regression': {
            'kernel__length_scale': [0.1, 1.0, 10.0],
            'alpha': [0.1, 0.2, 0.3]
        },
        'XGBoost Regression': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
        }
    }

    # Define the models with grid search
    models = [
        ('Linear Regression', linear_model),
        ('Ridge Regression', ridge_model),
        ('Lasso Regression', lasso_model),
        ('ElasticNet Regression', elasticnet_model),
        ('Decision Tree Regression', decision_tree_model),
        ('Random Forest Regression', random_forest_model),
        ('Gradient Boosting Regression', gradient_boosting_model),
        ('SVR', svr_model)
        ('Gaussian Process Regression', gaussian_process_model)
        ('XGBoost Regression', xgboost_model) 
    ]

    # Initialize best model and best score
    bestModel = None
    bestScore = -float('inf')
    
    # Evaluate each model
    for name, model in models:
        # Perform grid search
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, n_jobs=-1, scoring='r2')
        grid_search.fit(X_train, y_train)

        # Make predictions
        y_pred = grid_search.best_estimator_.predict(X_test)

        # Evaluate the model
        nmec_metrics(y_test, y_pred, len(features))

        # Update best model and best score if necessary
        if grid_search.best_score_ > bestScore:  # highest mean cross-validated score
            bestModel = grid_search.best_estimator_
            bestScore = grid_search.best_score_

        print("Best parameters found by grid search:")
        print(grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        print()

    joblib.dump(bestModel, 'best_model.pkl')
    return bestModel