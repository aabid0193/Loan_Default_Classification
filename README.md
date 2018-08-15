This folder contains Jupyter notebooks for Project McNulty of the Metis Data Science Bootcamp - Chicago cohort 6.

The focus of this project was classification - I focused on Random Forests & Logistic Regression in an ensemble method of Voting Classifier. with specificity and Sensitivity as my target metrics. If I had more time, I would have also considered boosted trees.

# Objective

Using a kaggle dataset regarding loans from [LendingClub](https://www.lendingclub.com/), my project aims to predict: "Of those who were granted a loan by LC, will they pay off their loann on time."


# Folders and Notebooks.


EDA: This notebook contains the code used to clean and transform my data into a format compatiable with sklearn algorithms - both classification and generalized linear models (logisitic regression). This process is ciritical for compatibility with sklearn algorithms. This also contains inital EDA on the dataset as well as a SQL demonstration and imputation of missing data.


Baseline Model: This notebook contains an initial dummy model of different varieties


Test before final feature selection: This notebook contains a some miscellaneous testing and further EDA.


Feature Importance & Final feature selection: This notebook contains final feature selection and 


Model Tuning and Final Model: This notebook contains the final model used in prediction, as well as all of the hyper parameter tuning, I was havingproblems with GridSearchCV on AWS EC2 initially (job would hang for hours), however, this forced me to manually write the loops that GridSearchCV does automatically. I also took a sample from the data to reduce runtime and in the future will plan to run this on the whole file. 