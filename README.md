# Machine-Learning-Analysis-of-Stock-Price-Trends

This project analyses historical stock price data from five companies (AMZN, FB, GOOGL, IBM, and MSFT) to predict stock price trends using machine learning techniques. Conducted as part of my MSc AI studies, the analysis employs three distinct approaches:

- Regression: Predicts future closing prices through rigorous data preprocessing, feature engineering, hyperparameter tuning, and performance evaluation.
- Classification: Frames the problem as predicting stock price movement (up, down, or stable) using classification models, supported by comprehensive experimentation and analysis.
- Clustering: Utilises clustering techniques to group stock data and predict future prices based on cluster means, involving detailed preprocessing, feature engineering, and result interpretation.

The project demonstrates the application of machine learning techniques to financial data, showcasing their potential for stock price analysis and trend prediction.

## Tools and Libraries Used:

Data Analysis: pandas, numpy
Data Visualisation: matplotlib, seaborn, scipy
Machine Learning and Evaluation: scikit-learn (sklearn)

## 1. Regression Approach Overview:
**Data Inspection and EDA:**
Exploratory analysis was conducted to understand the dataset's structure and uncover key patterns and relationships.

**Data Preprocessing:**
The data was cleaned, missing values were handled, and variables were transformed for analysis. Feature selection was performed using correlation analysis and Lasso regression to identify the most significant predictors.

**Model Building and Selection:**
Three regression models were developed and evaluated:

- Linear Regression
- Lasso Regression
- Ridge Regression
- 
Hyperparameter tuning was conducted using GridSearchCV to optimise model performance.

**Evaluation:**
The models were assessed using the following metrics:

- Mean Absolute Error: 11.6
- Mean Squared Error: 484.3
- Root Mean Squared Error: 21.8
- R² (R-squared): 0.99

**Validation:**
Model performance was validated through K-fold Cross-Validation and Learning Curve Analysis to ensure stability and consistency.

The regression approach demonstrated strong predictive performance, with a high R² value indicating the model effectively captures variance in the data. The evaluation metrics underscore the model's reliability in forecasting stock prices. However, further refinement may enhance its applicability in real-world scenarios.

## 2. Classification Approach Overview:
**Data Inspection and EDA:**
Performed an initial exploration to analyse feature distributions and relationships.

**Data Preprocessing:**
Cleaned the data and added a new column to categorise price movements (up, down, or stable) based on previous closing prices.

**Feature Engineering and Selection:**
Applied SelectPercentile and tree-based feature importance to identify the most relevant predictors.

**Model Building and Selection:**
Developed and compared the performance of:

- Decision Tree Classifier
- Random Forest Classifier
- Hyperparameter Tuning:

Optimised model parameters using Random Search CV for improved performance.

**Evaluation:**
The models achieved a mean accuracy of 0.51, with the "Stable" category showing strong predictive performance. However, limitations were observed in accurately predicting "Up" and "Down" trends.

**Validation:**
Stratified K-Fold Cross-Validation and Learning Curve Analysis confirmed model consistency and robustness.

The classification approach provides a foundational framework for predicting stock price trends. While the current performance highlights areas for improvement, it demonstrates the potential for enhancing predictive accuracy through model refinement.

3. Clustering Approach Overview
Data Inspection and EDA:
Performed initial exploration to uncover patterns and relationships within the dataset.

Feature Selection:
Utilised the Silhouette Score and Davies-Bouldin Index to identify key features for clustering.

Dimensionality Reduction:
Applied PCA to reduce data complexity while retaining important variance.

Model Building:
Implemented K-Means Clustering, optimising the number of clusters (K) using the Silhouette Score.

Validation:
Evaluated clustering performance with:

Average Silhouette Score: 0.894977
Average Davies-Bouldin Index: 0.380072
Prediction:
Used Bootstrap Sampling to predict the mean nth-day price within each cluster.

The clustering approach successfully grouped similar data points, demonstrating high-quality clustering metrics. While this method simplifies stock price trend analysis, further refinement could improve its predictive capabilities.

## Project Outcome:

The project successfully applied three distinct machine learning approaches—regression, classification, and clustering—to predict stock price trends. The regression approach demonstrated strong predictive performance with high accuracy, particularly in forecasting future prices, though further refinement is needed for real-world applicability. The classification approach established a solid foundation for predicting stock price movements, with reasonable accuracy in identifying stable price trends but room for improvement in predicting up or down trends. Lastly, the clustering approach effectively grouped stock data and predicted mean prices within clusters, achieving high-quality clustering scores. Overall, the project highlights the potential of machine learning techniques in stock price prediction, offering valuable insights for refining predictive models in future work.
