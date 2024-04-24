# Dementia Prediction
## About

In this project, we'll explore a comprehensive analysis of a dementia patients dataset.  
Our objective is to clean this data, perform exploratory data analysis (EDA), and apply several machine learning models to predict whether a person has dementia based on it's attribute.

Please read through the notebook files in the following order
1. [Data Cleaning](https://github.com/Gideon2882/Dementia-Prediction/blob/cc34af83b3bcf7cb415e6734a5c768c649b30bc9/Data%20Cleaning.ipynb)
2. [Data Cleaning and Exploratory Analysis](https://github.com/Gideon2882/Dementia-Prediction/blob/cc34af83b3bcf7cb415e6734a5c768c649b30bc9/Exploratory%20Data%20Analysis.ipynb)
3. [Logistic Regression](https://github.com/Gideon2882/Dementia-Prediction/blob/cc34af83b3bcf7cb415e6734a5c768c649b30bc9/Logistic%20Regression.ipynb)
4. [Logistic Regression without 'Cognitive_Test_Score](https://github.com/Gideon2882/Dementia-Prediction/blob/cc34af83b3bcf7cb415e6734a5c768c649b30bc9/Logistic%20Regression%20without%20'Cognitive_Test_Scores'.ipynb)
5. [Multilayer Perceptron](https://github.com/Gideon2882/Dementia-Prediction/blob/cc34af83b3bcf7cb415e6734a5c768c649b30bc9/Multilayer%20Perceptron.ipynb)
6. [Multilayer Perceptron without 'Cognitive_Test_Score](https://github.com/Gideon2882/Dementia-Prediction/blob/7d08b6d3e1ab284cafeb1b0f8361d0269df6e20e/Multilayer%20Perceptron%20without%20'Cognitive_Test_Scores'.ipynb)

Note that there are 1 original & 3 cleaned version of the dataset, which would be separated for comparison purposes.

## Contributor
* @Gideon2882 - Gideon Leow - Exploratory Data Analysis, Multilayer Perception, Multilayer Perception without 'Cognitive_Test_Scores', Presenter.
* @MrLegumes - Ethan Yew - Data Cleaning, Logistic Regression, Logistic Regression without 'Cognitive_Test_Scores', Slides.
* @tyuxuan01 - Yu Xuan - Logistic Regression, Logistic Regression without 'Cognitive_Test_Scores', Slides.

## Problem
Dataset: [Dementia Patient Health,Prescriptions ML Dataset](https://www.kaggle.com/datasets/kaggler2412/dementia-patient-health-and-prescriptions-dataset/data)  
Context: Predict whether person has dementia by training classification models based on attributes mention in dataset.
Problem Definition: Are we able to predict dementia with based on its attributes and if so, can we find the most important attribute?

## Model Used
Logistic Regression
* Training a Logistic Regression model to predict dementia with our dataset and using coefficients to determine the most important variable.

Multilayer Perceptron (MLP)
* Training a Multilayer Perceptron(MLP) model to predict dementia with our dataset and confirming the effects of the presence/absence of the most important variable on the predictions.

## Conclusion
Through this effort to predict dementia by data analysis and statistical modelling, we've found that:
* Logistic Regresion and MLP performs very well (95+% accuracy and F1 Score) on our dataset, but MLP performs substantially less accurately(60% accuracy and 59% F1 Score) compared to Logistic Regresion model(74% accuracy and 75% F1 Score) when data is incomplete/ when most important variable is absent.
* By calculating Coefficients, we found out that Cognitive_Test_Score has the highest coefficient and thus is the most important attribute that determines/predicts whether a person has dementia.
* When we train our MLP model on our dataset for 100 epochs, our model is learning well without experiencing significant overfitting, as both losses hold true over multiple epochs.
* For real life situation, when data may not be fully complete, it may be better to use logistic regression to predict dementia.

## What we have learned
* Various machine learning model from regression to neural network.
* Different optimizer, overfitting vs underfitting.
* How to clean datasets to make the datatypes better suited for machine learning model.
* Handling imbalanced datasets through resampling/(fit_transforms).
* Different models(Logistic Regression, MLP)
* Neural Networks(Tensorflow)

## Reference
* [Dementia Patient Health. (2024, January 24). Kaggle.](https://www.kaggle.com/datasets/kaggler2412/dementia-patient-health-and-prescriptions-dataset/data)      
* [PyTorch 2.2 documentation. (n.d.).](https://pytorch.org/docs/stable/index.html)  
* [Logistic Regression in Python (n.d.). RealPython.](https://realpython.com/logistic-regression-python/)
* [Supervised learning. (n.d.). Scikit-learn.](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)     
