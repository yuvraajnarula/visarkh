import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = './KwhConsumptionBlower78_1.csv'
try:
    dataset = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    # Explore the dataset
    print(dataset.head())
    print(dataset.info())
    print(dataset.describe())

    # Handle missing values
    if dataset.isnull().sum().any():
        print("Missing values found. Filling with mean values.")
        dataset = dataset.fillna(dataset.mean())
    else:
        print("No missing values found.")

    # Drop unnecessary columns
    dataset = dataset.drop(['Unnamed: 0'], axis=1)

    # Convert date and time to datetime objects
    dataset['Datetime'] = pd.to_datetime(dataset['TxnDate'] + ' ' + dataset['TxnTime'])
    dataset = dataset.drop(['TxnDate', 'TxnTime'], axis=1)

    # Set datetime as index
    dataset.set_index('Datetime', inplace=True)

    # Extract additional features from datetime
    dataset['Hour'] = dataset.index.hour
    dataset['DayOfWeek'] = dataset.index.dayofweek
    dataset['Month'] = dataset.index.month

    # Calculate average consumption per hour
    avg_consumption_per_hour = dataset.groupby('Hour')['Consumption'].mean()

    # Determine the hour with the lowest average consumption
    best_hour = avg_consumption_per_hour.idxmin()
    lowest_avg_consumption = avg_consumption_per_hour.min()

    print(f'The best time of the day for feasible electricity consumption is {best_hour}:00 with an average consumption of {lowest_avg_consumption:.2f} kWh.')

    # Define features and target
    target_column = 'Consumption'  # The target variable in your dataset
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the model
    model = RandomForestRegressor(random_state=42)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print(f'Best Parameters: {best_params}')

    # Plot feature importances
    importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.tight_layout()
    plt.show()

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', lw=2, color='red')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Energy Consumption')
    plt.show()

    # Calculate hourly consumption per day
    hourly_consumption_per_day = dataset.groupby([dataset.index.date, 'Hour'])['Consumption'].mean().unstack()

    # Plot the hourly consumption per day
    plt.figure(figsize=(14, 8))
    sns.heatmap(hourly_consumption_per_day, cmap='viridis', cbar_kws={'label': 'Average Consumption (kWh)'})
    plt.title('Hourly Electricity Consumption Per Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Day')
    plt.show()

except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")

