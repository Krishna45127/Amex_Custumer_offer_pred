# Amex Customer Offer Recommendation System

## Overview

Built a Machine Learning pipeline to predict top-7 customer offers, achieving a MAP@7 score of 0.65. The project focuses on personalized offer recommendation using customer transactions, behavioral events, and offer metadata.

## Features

* End-to-end ML recommendation pipeline
* Feature engineering from transactional and event data
* Model-specific preprocessing workflows
* Hyperparameter tuning using Optuna
* Ensemble learning with stacking architecture
* MAP@7 evaluation for ranking performance

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Optuna

## Methodology

### Data Processing

* Cleaned and preprocessed customer, transaction, and offer datasets
* Handled missing values and categorical encoding
* Engineered features from user behavior and offer interactions

### Model Development

Implemented and tuned multiple boosting models:

* XGBoost
* LightGBM
* CatBoost

Used Optuna for automated hyperparameter optimization.

### Ensemble Learning

Designed a stacking ensemble using Logistic Regression as the meta-model to combine predictions from multiple base learners and improve ranking performance.

## Evaluation Metric

The project was evaluated using MAP@7 (Mean Average Precision at 7), a ranking metric commonly used in recommendation systems.

## Results

* Achieved MAP@7 score of 0.65
* Improved recommendation accuracy using ensemble learning
* Generated personalized top-7 customer offer predictions
