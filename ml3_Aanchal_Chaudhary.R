#question 1
#reading csv file
df <- read.csv("heart.csv")
ncol(df)
df <- df[,colSums(is.na(df))<nrow(df)] #removing columns with all nulls
library(dplyr)
df = df %>% select(-wrist_dim) #removing column with a lot of nulls and few values , not very important
df= na.omit(df) #removing rows with nulls

nrow(df)
#The shuffle parameter is needed to prevent non-random assignment to to train and test set. With shuffle=True you split the data randomly.If you do a test/train split on the data without shuffling then you may have a different data distribution in your splits!
#using caret library to make sure test and train data have same distribution
library(caret)
set.seed(3456)
trainindex = createDataPartition(df$heart_attack, p=0.8,list =FALSE,times =1)
Train = df[trainindex,]
nrow(Train)
Test = df[-trainindex,]
nrow(Test)

#full model
model = lm(heart_attack ~., data = Train)
summary(model)

Test$predictions = predict(model, Test)
Test
nrow(Test)
#calculating out of sample r^2
nulld = sum((Test$heart_attack-mean(Train$heart_attack))^2)
residd = sum((Test$heart_attack-Test$predictions)^2)
oosr = 1-(residd/nulld)
#out of sample r^2 is 0.89 i.e 89% variation in dependent variable explained by model

#question2 
#cross-validation
## The process of using out of Sample Experiments to do model selection is called cross validation. In Cross validation we divide our dataset into multiple train and test subsets. We train the model using our train subsets and evaluate it on the test subset.  In k-fold cross validation, the process is repeated k times, with each subset serving as the test set once. At the end we take the final performance mesure as the average performance across all iterations. It is a model validation technique for assessing how the results of a statistical analysis will generalize

#Disadvantage with Cross Validation is that it is computationally expensive. Especially if we are dealing with big datasets. Another bottleneck is that CV is sensitive to the way data is partitioned. 
### if the data is not randomly partitioned, the results we find from CV may not be a good representative of the model's performance on unforeseen data. Moreover ,if the data contains a large number of outliers, k-fold CV may not be an appropriate approach.

#question3
set.seed(1)

# defining training control
# as cross-validation and 
# value of K equal to 8
train_control <- trainControl(method = "cv",number = 8)

# training the model by assigning heart_attack column
# as target variable and other columns
# as independent variable
heart_kcv <- train(heart_attack ~., data = Train, 
                   method = "lm",
                   trControl = train_control)
#results of the model
print(heart_kcv)
heart_kcv$finalModel
#mean r^2 is 0.86, 86% variation in dependent variable is explained by the model after cross-validation
#which is close enough to what we found in q1
kcv_R2 <- mean(heart_kcv$resample$Rsquared)
kcv_R2
# Predictions for each fold
heart_kcv$resample

#question 4
#Lasso regression is type of regression that adds a regularization term to the model. In Lasso, The regularization term is the sum of the absolute values of the coefficients,  and it is multiplied by a scalar value lambda, which controls the strength of the regularization.
##The main goal of the lasso regression is to decrease the coefficients of less importance to 0. Lasso regression not only helps in regularization but also feature selection. Some good points are : it performs feature selection, by shrinking coefficients of less importance to 0. It is very useful for cases when we have more features than the number of observations i.e we have less data points but large number of features. The regularization term is robust to outliers in the data.

#Diadvantage: If there is high multicollinearity between some features, Lasso regression may not be able to select true important features. Also, Lasso Regression does not perform well with categorical variables. Sometimes Depending on the value of Lambda, it can over penalize some features resulting in underfitting which we don't want.Hence, It is also important to scale data before lasso.

#question5 

library(glmnet)

x <- data.matrix(Train[, 1:16])
y <- Train$heart_attack
heart_regress_lasso_kcv <- cv.glmnet(x, y, alpha = 1, nfolds = 8, family = "gaussian", standardize = TRUE)
summary(heart_regress_lasso_kcv)
#  lambda min value 
lmbd_min <- heart_regress_lasso_kcv$lambda.min
lmbd_min
#  lambda 1se 
lmbd_1se <- heart_regress_lasso_kcv$lambda.1se
lmbd_1se
### Plot 
plot(heart_regress_lasso_kcv)

#lambda min gives us the lambda value with minimum average out-of-sample Deviance.
#Lambda 1se gives us the biggest lambda value with average OOS deviance no more than 1 SD away from the minimum.
#by the plot, we can see there is not much difference in the mean square error for both lamda min and lamda 1se. They perform similarly in terms of MSE values. Therefore, 
### We go ahead  with lambda1se as the model to predict the values of test dataset as it has less number of variables i.e simpler model with only 5 variables compared to 6 in Lambda min model without compromising on the predictive power.

# Rerun the model with lambda 1se
heart_regress_lasso_kcv2 <- glmnet(x, y,alpha=1, lambda = lmbd_1se)

# Predict the test dataset using the model
heart_test_x <- data.matrix(Test[,1:16])
heart_test_y <- Test$heart_attack
heart_test_pred2 <- predict(heart_regress_lasso_kcv2, s=lmbd_1se, newx = heart_test_x)

#obtain OOS R^2
D_lasso <- sum((heart_test_y - heart_test_pred2)^2)
D0_lasso <- sum((heart_test_y - mean(y))^2)

R2_lasso <- 1-(D_lasso/D0_lasso)
R2_lasso
# we get OOS R2 value of 80% from 8-fold Cross Validated Lasso regression

# Question 5.2 
#comparing r^2 's from all the models above
q1_res <- model$coefficients
q3_res <- heart_kcv$finalModel$coefficients
q5_res <- coef(heart_regress_lasso_kcv, select = "1se")

#Comparing model outputs from questions one, three, and five.
result_r2 <- cbind(oosr, kcv_R2, R2_lasso)
colnames(result_r2) <- c("q1-OOS_R2", "q3-kcv_R2", "q5-lasso_R2")
result_r2
model_covariates<-cbind(q1_res, q3_res, q5_res)
model_covariates
# 8 fold Cross Validated Lasso Regression with lambda 1se got rid of the 11 variables and returned an R2 value of 80% with only 5 variables which is again great OOS predictive power. 

#question 6
# AIC(Akaike Information Criterion) is a way to measure relative quality of regression models. It is one of the model selection tools. We desire a model with lower AIC values. Lower the AIC, the better the model
#AIC = Deviance + 2df, df refers to degrees of freedom AIC is trying to estimate the OOS deviance i.e what your deviance would be in another sample of size n. Usually, AIC does not perform well when the sample size is small. In fact, AIC is only good for big n/df. In big data, number of parameters can be huge. Often df=n, In those cases, AIC tends to overfit. AICc (corrected AIC) is an adjusted version of AIC that is used when the sample size is small or degrees of freedom is too big. It corrects for the bias that can occur when the sample size is small or when df is too big. AICc = Deviance + 2df(n/(n-df-1)). AICc is a more robust model selection tool.