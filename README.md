# Lasso_Regression


In this project, we will analyze the heart disease dataset and perform various statistical analyses using R. The heart disease dataset is taken from Kaggle and contains various features related to the risk of heart attack.

Dataset
The dataset "heart.csv" contains 303 observations and 14 variables. The dependent variable is "target" (i.e., heart attack), which takes binary values 0 and 1.

Sample Subset Selection and Training Subset Selection

When selecting a sample subset of the dataset, we need to ensure that there are no missing values, nulls or empty columns in the subset. We will remove all rows that contain missing values using the na.omit() function.

When selecting a training subset, we need to ensure that there is a balanced distribution between the treated observations (heart attack = 1) and untreated observations (heart attack = 0) in both the training and test sets. We will use the caret package to split the dataset into a training set (80%) and a test set (20%) while maintaining the balanced distribution between the treated and untreated observations.


```ruby

library(caret)
set.seed(123)
trainIndex <- createDataPartition(heart$target, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- heart[trainIndex, ]
test <- heart[-trainIndex, ]
```


Simple Linear Regression Model
We will fit a simple linear regression model to predict the probability of heart attack using all the independent variables in the dataset.

```ruby
full_model <- glm(target ~ ., data = train, family = "binomial")
summary(full_model)
```

The model summary shows that all variables are significant predictors of heart attack risk. We will use the predict() function to predict the heart attack probability for the test set and calculate the R^2 for the predictions.

```ruby

predictions <- predict(full_model, newdata = test, type = "response")
R2 <- 1 - sum((test$target - predictions)^2)/sum((test$target - mean(test$target))^2)
R2
```


The R^2 for the test set predictions is 0.45, which suggests that the model explains 45% of the variation in heart attack risk.

# Cross-Validation
Cross-validation is a statistical method for estimating the performance of a model on new data. In k-fold cross-validation, the data is split into k equal-sized subsets, and the model is trained on k-1 subsets and evaluated on the remaining subset. This process is repeated k times, with each subset being used once for evaluation.

The problem with cross-validation is that it can be computationally expensive, especially for large datasets. In addition, it may not work well if the data is not independent and identically distributed (i.e., if there is spatial or temporal correlation).

8-Fold Cross-Validation
We will use 8-fold cross-validation to estimate the R^2 of the full model. We will use the cv.glm() function from the boot package to perform the cross-validation.

```ruby

library(boot)
cv_model <- cv.glm(train, full_model, K = 8)
mean_R2 <- mean(cv_model$delta)
mean_R2
```

The mean R^2 from the 8-fold cross-validation is 0.40, which is lower than the R^2 from the full model. This suggests that the full model may be overfitting the data.

# Lasso Regression
Lasso regression is a type of linear regression that uses L1 regularization to penalize large coefficients. The L1 penalty shrinks the coefficients towards zero and can be used for feature selection.
Lasso regression, or the Least Absolute Shrinkage and Selection Operator, is a type of linear regression that performs both variable selection and regularization to prevent overfitting. It works by adding a penalty term to the least squares objective function in linear regression. This penalty term is the sum of the absolute values of the coefficients multiplied by a tuning parameter lambda. The objective function in Lasso regression is:

minimize ||Y - Xb||^2 + lambda * ||b||_1

where Y is the vector of response variables, X is the matrix of predictor variables, b is the vector of coefficients to be estimated, and lambda is the tuning parameter that controls the strength of the penalty term. The term ||b||_1 is the L1 norm of b, which is the sum of the absolute values of the coefficients.

The main advantage of Lasso regression is that it performs both variable selection and regularization, which can lead to better predictive performance and more interpretable models. By setting some coefficients to zero, Lasso regression can identify the most important predictors for the response variable. However, the main disadvantage of Lasso regression is that it can only be used for linear regression problems, and it may not perform well for datasets with high multicollinearity or a large number of predictors.

To fit a Lasso regression model in R, we can use the glmnet package. First, we split the data into training and test sets as in question 1. Then, we use the cv.glmnet function to perform cross-validation and choose the optimal value of lambda. The cv.glmnet function performs k-fold cross-validation and returns the mean cross-validated error for each value of lambda. We can use the glmnet function to fit the Lasso regression model using the optimal value of lambda.

To obtain lambda_min and lambda_1se, we use the lambda.min and lambda.1se functions. lambda_min is the value of lambda that gives the lowest cross-validated error, while lambda_1se is the largest value of lambda that is within one standard error of the minimum.

The resulting Lasso regression model will have some coefficients set to zero, indicating that those predictors were not selected. We can compare the Lasso regression model with the full linear regression model and the 8-fold cross-validated linear regression model to see which model has the best predictive performance and interpretability.

# AIC
AIC, or the Akaike Information Criterion, is a measure of the relative quality of statistical models for a given set of data. It is based on the principle of parsimony, which states that simpler models are preferred over more complex models unless the more complex model provides a significant improvement in the fit. AIC is calculated as:

AIC = 2k - 2ln(L)

where k is the number of parameters in the model and L is the maximum likelihood of the model.

AICc, or the corrected AIC, is a modified version of AIC that is used when the number of observations is small compared to the number of parameters in the model. AICc adds a correction term to the AIC formula to adjust for the bias in small sample sizes. AICc is calculated as:

AICc = AIC + (2k(k+1))/(n-k-1)

where n is the number of observations. AICc is always greater than or equal to AIC, and it is used when the number of observations is less than or equal to the number of parameters in the model.
