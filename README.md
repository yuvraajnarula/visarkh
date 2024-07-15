# visarkh
This repository contains Python code for analyzing and predicting energy consumption using a RandomForestRegressor model. The dataset used (`KwhConsumptionBlower78_1.csv`) includes hourly energy consumption data.

## Overview

The code performs the following steps:

1. **Loading and Preprocessing Data:**
   - Loads the dataset (`KwhConsumptionBlower78_1.csv`).
   - Handles missing values by filling with mean values.
   - Converts date and time columns to datetime objects and sets datetime as index.
   - Extracts additional features like hour of day, day of week, and month.

2. **Exploratory Data Analysis:**
   - Calculates average consumption per hour (`avg_consumption_per_hour`).
   - Identifies the hour with the lowest average consumption.

3. **Model Training and Evaluation:**
   - Splits data into training and test sets.
   - Scales the features using `StandardScaler`.
   - Trains a `RandomForestRegressor` model.
   - Evaluates the model using Mean Squared Error (MSE) and R-squared metrics.
   - Performs hyperparameter tuning using `GridSearchCV` to optimize the model.

4. **Feature Importance:**
   - Plots feature importances derived from the trained model.

5. **Prediction Visualization:**
   - Visualizes actual vs predicted energy consumption using a scatter plot.

6. **Hourly Consumption Analysis:**
   - Computes hourly consumption per day and plots it using a heatmap.

## Dependencies

Ensure you have the following Python libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage
1. Clone the repository:
   ```
   git clone https://github.com/yuvraajnarula/visarkh.git
   cd visarkh
   ```
2. Install dependencies 
   ```
   pip install -r requirements.txt
   ```
3. Run the script
   ```
   python server.py
   ```