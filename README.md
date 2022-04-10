# Data-Science-Class

This Data is given for one of my class's EE 460J- DATA SCIENCE LAB Mid term exam
![image](https://user-images.githubusercontent.com/93844635/162642410-ca0e1707-cbe3-483a-a2c1-8004bbc6d7ee.png)

This is a classification problem.

What worked for me and what didnt work for me in this dataset

Feature Engineering

1. So the columns were all not named. So it was difficult to make sense of each column.
So one thing I noticed was the columns were all highly skewed to the right side, and it
was difficult to do a good predicted if features are skewed
2. The first attempt I tried was transforming the columns which have skewness outside the
range of -1,1 with logarithmic transformations. Different logarithmic transformations
were done for different ranges. But my public score didn’t increase. I guess a lot of
information was lost during these transformations.
3. There were a lot of outliers for a few columns too. I have tried replacing the outliers with
the median of that column. But it didn’t work too.
Handling the Imbalance of classes1. So there was a huge imbalance of 1 and 0. Thus I used SMOTE Algorithm,
oversampling, and tried these methods. But the model score didn’t increase

Dropping Columns:

1. The columns we were given were about 25. It's usually a high number of columns. So I
have tried to remove those columns with PCA. But PCA didn’t work in this case as my
score didn’t improve
2. The Next attempt was dropping columns that had low variance and low correlation.
Unfortunately, this too didn’t work in my case which is the XGBOOST Model and in
fact, my score got decreased because of this approach

PreProcessing Steps

1. All the features were converted into a min-max scalar. I tried testing all the transforms
like quantile, power, and standard scaler transform. But the min-max scaler produced the
best results. Usually, a standard scaler performs better, but our feature didn’t follow a
normal distribution, thus it may be standard scaler didn’t work.

Model Selection

1. The first model I used was XGBOOST and without any hyperparameter tuning, it gave
me an AUC Score of 0.8792 in the public score. I tried then parameter tuning in the
XGBoost Original API version and my score didn’t increase. I guess XGBoost achieved
the saturation model for me and I used a randomized grid search, so it can be one of the
reasons too
2. Then the next model I tried was Random Forest. So Random forest almost gave me an
equal score as the XGBoost model and it actually gave me a better score when I tuned my
model's hyperparameters.
3. I tried other boosting algorithms like AdaBoost and Catboost too, but the model score
didn’t increase

Model Ensembling

1. In the final step, I trained 3 different models such as Random Forest, GBM, and XGB.
Then I got the prediction of these models and I took the weighted sum of it mostly in the
ratio of (0.3:0.3:0.4). So this model is what gave me the highest score on my public
leaderboard (0.90513)



FINAL SCORE WITH WEIGHTED PREDICTION MODEL FROM XGBOOST+ RANDOM FOREST+ GBM is 0.90513
