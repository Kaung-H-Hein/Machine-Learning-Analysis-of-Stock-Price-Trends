# Machine-Learning-Analysis-of-Stock-Price-Trends
This project analyses historical stock price data from five companies (AMZN, FB, GOOGL, IBM, and MSFT) to predict stock price trends using machine learning techniques. Conducted as part of my MSc AI studies, the analysis employs three distinct approaches:
- Regression: Predicts future closing prices through rigorous data preprocessing, feature engineering, hyperparameter tuning, and performance evaluation.
- Classification: Frames the problem as predicting stock price movement (up, down, or stable) using classification models, supported by comprehensive experimentation and analysis.
- Clustering: Utilises clustering techniques to group stock data and predict future prices based on cluster means, involving detailed preprocessing, feature engineering, and result interpretation.
The project demonstrates the application of machine learning techniques to financial data, showcasing their potential for stock price analysis and trend prediction.

## Regression Approach Overview:
**Data Inspection and EDA:** The first step involved exploring the dataset to understand its structure and identify key patterns and relationships.

**Data Preprocessing:** This step included cleaning the data, handling missing values, and transforming variables to prepare for analysis. Feature selection was done using correlation analysis and Lasso regression to identify the most influential predictors.

**Model Building and Selection:** Three models were considered:
- Linear Regression
- Lasso Regression
- Ridge Regression

Hyperparameter tuning was performed using GridSearchCV to optimise model performance.

**Evaluation Metrics:** The models were evaluated using several metrics:
- MAE (Mean Absolute Error): 11.5814
- MSE (Mean Squared Error): 484.3347
- RMSE (Root Mean Squared Error): 21.7648
- R² (R-squared): 0.9990

Performance was further validated using K-fold Cross-Validation and Learning Curve Analysis to assess model stability and consistency.
