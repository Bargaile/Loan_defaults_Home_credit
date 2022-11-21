# Loan_defaults_Home_credit

# PURPOSE

The aim of this project was to create a Machine Learning tool for small lending firms, lending clubs (gathered from private individuals, like [this](https://www.paskoluklubas.lt/?gclid=Cj0KCQjwqoibBhDUARIsAH2OpWjB-YBaFz5v3KqnHxn0DvAD-6W9DL9NGsPszsVIIdwLaQ0HBFVkLacaAicqEALw_wcB)), which could enable them (after the loan application approval) to get the probability, if this new borrower could default in the future or not. This could help to decide on the interest rate to every certain new borrower. Having the probability of possible default, for sure interest rate can be higher. Also it could be a good possibility to decide the term of the future loan. 

# DATA

Home Credit data sets [taken from here](https://www.kaggle.com/competitions/home-credit-default-risk).


# STRUCTURE:

- **EDA_home_credit.ipynb**
Deep analysis of the given data sets, data cleaning, feature engineering to prepare the final, clean data set for modeling. Inferential statistical analysis was proceeded.

- **Modeling_home_credit.ipynb**

The success measure of the models - ROC AUC. Different models were explored to see, which could do best on this data. The two best models by ROC AUC were selected and developed further. Different dimensionality reduction techniques were tried (PCA, Linear Discriminant analysis, Recursive feature selection), the optimal probability threshold was calculated by RO curve, model calibration  was tried.

- **helpers[folder]**

Contain .py file with all classes and functions, used in the EDA and Modeling.

- **api[folder]**

Contains code of deploying model on localhost with fastapi, Dockerfile (app is 'dockerized')
and deployed to Google Cloud storage:

https://defaultpredictions-rv5fvp3ynq-lz.a.run.app/docs#/ 
