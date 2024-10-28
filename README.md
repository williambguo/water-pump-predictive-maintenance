# Predicting Water Pump Functionality in Tanzania

![image](https://github.com/user-attachments/assets/aadd8bca-9746-4dd1-ba98-336117986574)

## Overview

This project aims to build a predictive model to determine whether a water pump is functional or non-functional based on various features. The classification task involves using machine learning techniques to classify water pumps into two categories: 'functional' and 'non-functional'. The project is divided into several stages, as documented in the provided notebooks.

## Business & Data Understanding

As per the competition website on [DrivenData](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) where I obtained the dataset, a smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

The project begins with an in-depth **Exploratory Data Analysis (EDA)**, as documented in [eda.ipynb](./eda.ipynb). In this notebook, I dive deep into the features in the dataset, analyze distributions, and visualize relationships between variables. This step helped me identify any anomalies or trends that may affect model performance in addition to determining which features were redundant or unnecessary for classification modeling later on.

Here is a look at the target variable as a scatter plot over a map of Tanzania:
![image](https://github.com/user-attachments/assets/3987db1b-dc70-47f8-8006-4e9e0afa43b9)

## Data Preparation

Data preprocessing is a critical step in any ML classification analysis. This dataset contained a lot of missing values that had to be dealt with. 

*Missing data visualized*:
![image](https://github.com/user-attachments/assets/455ffdf3-5e2d-4a52-a1f1-57418fb5ea10)

In [training_data_preprocessing.ipynb](./training_data_preprocessing.ipynb), several data cleaning and transformation techniques are applied:
- Imputing missing values
- Encoding categorical features
- Solving target variable class imbalance problem
- Encoding target variable
- Feature scaling and transformation
- Creating new features by combining existing ones

This ensured the dataset is in good condition for modeling.

## Modeling

The **Modeling** phase, outlined in [modeling_and_evaluation.ipynb](./modeling_and_evaluation.ipynb), involved training five different classifier algorithms in their default forms using a pipeline of seven different feature encoding methods for every classifier:
- Logistic Regression
- Decision Tree
- Naive Bayes
- Random Forest
- XGBoost

The best scoring (highest accuracy score) of the five models was the Random Forest (RF) classifier. The train accuracy was 0.9663 while test accuracy was 0.8118 Hyperparameter tuning was then performed on the RF model to optimize its hyperparameters. The tuned RF model had a train accuracy of 0.9663 and a test accuracy of 0.8134, a minor improvement to the default RF model.

## Model Evaluation

Once the models were trained, they were evaluated with a full classification report based on key metrics such as accuracy, precision, recall, and F1-score. The main metric, however, was accuracy score mainly due to the fact that it is considered the most important evaluation metric by the competition host. 

*Tuned RF Confusion Matrix*
![image](https://github.com/user-attachments/assets/045fe61f-1171-4862-810c-d91fecdbba15)


*Tuned RF Feature Importance*
![image](https://github.com/user-attachments/assets/57f3088b-18c3-403b-be74-6521909f5640)

The feature importance plot shows `gps_height` and `quantity` as the most influential features.

## Summary of Findings

After evaluating multiple models, the best-performing model is the Random Forest algorithm with optimized hyperparameters. It is able to predict water pump functionality with an accuracy of 81.34%. The height at which a pump is located and the quantity of water at a water point are the most influential features with the local government authority and pump's age not far behind. 

## Future Work

These models can still be improved significantly with better data preprocessing. There are more sophisticated methods for imputing missing data and dealing with the high cardinality categorical variables in this dataset. One such case is that I simply dropped `funder` and `installer` before modeling due to their very high cardinality but they could be influential features for this classification analysis.
