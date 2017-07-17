library(randomForest)
library(nnet)
library(SDMTools)
library(tree)
library(naivebayes)
library(e1071)
library(Metrics)
library(neuralnet)
library(class)

classify <- function(x) {
        value = -1
        if (startsWith(x, "E"))
        {
                value = 19
        }
        else if (startsWith(x, "V"))
        {
                value = 20
        }
        else
        {
                x = as.numeric(x)
                if (x < 140) {
                        value = 1
                }
                else if (x >= 140 && x < 240) {
                        value = 2
                }
                else if (x >= 240 && x < 280) {
                        value = 3
                }
                else if (x >= 280 && x < 290) {
                        value = 4
                }
                else if (x >= 290 && x < 320) {
                        value = 5
                }
                else if (x >= 320 && x < 360) {
                        value = 6
                }
                else if (x >= 360 && x < 390) {
                        value = 7
                }
                else if (x >= 390 && x < 460) {
                        value = 8
                }
                else if (x >= 460 && x < 520) {
                        value = 9
                }
                else if (x >= 520 && x < 580) {
                        value = 10
                }
                else if (x >= 580 && x < 630) {
                        value = 11
                }
                else if (x >= 630 && x < 680) {
                        value = 12
                }
                else if (x >= 680 && x < 710) {
                        value = 13
                }
                else if (x >= 710 && x < 740) {
                        value = 14
                }
                else if (x >= 740 && x < 760) {
                        value = 15
                }
                else if (x >= 760 && x < 780) {
                        value = 16
                }
                else if (x >= 780 && x < 800) {
                        value = 17
                }
                else if (x >= 800 && x < 1000) {
                        value = 18
                }
        }
        value
}

maxidx <- function(arr) {
        return(  which(arr == max(arr)) )
}

#Fully cleaned and imputed dataset
df <- read.csv("diabetes_clean.csv", header = TRUE, strip.white = TRUE, na.strings = c("NA", "?"," ","."))
df$readmitted <- ifelse(df$readmitted == df$readmitted[1], 0,1)

train <- sample(1:nrow(df), 0.8*nrow(df)) # Split the data into 80:20 ratio for cross validation
train_data <- df[train,] # Training data
test_data <- df[-train,] # Test data

####### Main Task: Readmitted or not #######

# Model 1: Generalised Logistic Regression
model.logit <- glm(readmitted~., data=train_data[,-1], family=binomial(link='logit'))
plot(model.logit)
pred.logit <- predict(model.logit,test_data, type = "response")
pred.logit <- ifelse(pred.logit > 0.5, 1, 0)
pred.logit.2 <- ifelse(pred.logit == 1, "TRUE", "FALSE")
mean(pred.logit==test_data$readmitted) # Accuracy: 64.6%

# Model 2: Random forest
df$readmitted <- ifelse(df$readmitted == 1, "TRUE", "FALSE")
df$readmitted <- as.factor(df$readmitted)
model.rf1 <-randomForest(readmitted~., data=train_data[,-1], ntree=10, na.action=na.exclude, importance=T,proximity=T) 
print(model.rf1) #error rate: 42.08%

model.rf2 <-randomForest(readmitted~., data=train_data, ntree=20, na.action=na.exclude, importance=T,proximity=T) 
print(model.rf2) #error rate: 40.21%

model.rf3 <-randomForest(readmitted~., data=train_data, ntree=30, na.action=na.exclude, importance=T,proximity=T) 
print(model.rf3) #error rate: 39.21%

model.rf4 <-randomForest(readmitted~., data=train_data, ntree=40, na.action=na.exclude, importance=T,proximity=T) 
print(model.rf4) #error rate: 38.32%

model.rf5 <-randomForest(readmitted~., data=train_data, ntree=50, na.action=na.exclude, importance=T,proximity=T) 
print(model.rf5) #error rate: 38.57%

mtry <- tuneRF(train_data[,-1], train_data$readmitted, ntreeTry=40,stepFactor=1.5, improve=0.01, trace=TRUE, plot=TRUE)

model.rf <- randomForest(readmitted~., data=train_data, ntree=40, mtry = 8, na.action=na.exclude, importance=T,proximity=T)
pred.rf <- predict(model.rf, test_data)
mean(pred.rf==test_data$readmitted) # Accuracy: 63.55%

plot(model.rf)

# Model 3: Neural networks
model.nn <- nnet(readmitted ~., data=train_data[,-1], size=5, maxit=1000)
pred.nn <- predict(model.nn, test_data,type= "raw")
pred.nn <- ifelse(pred.nn > 0.5, "TRUE", "FALSE")
mean(pred.nn==test_data$readmitted) # Accuracy: 63.35%

plot(model.nn)

#Model 4: Decision tree
df$readmitted <- as.factor(df$readmitted)
model.tree <- tree(readmitted~., data = train_data[,-1])
pred.tree <- predict(model.tree, test_data)
plot(model.tree)
text(model.tree)
pred.response <- ifelse(pred.tree > 0.5, "FALSE", "TRUE")
mean(test_data$readmitted != pred.response) # Accuracy: 50%

# Model 5: Naive Bayes
model.nb <- naive_bayes(train_data[,-1], train_data$readmitted, laplace = 1, usekernel = T, prior = NULL)
pred.nb <- predict(model.nb, test_data, type = "prob", threshold = 0.01, eps = 0.1)
idx.nb <- apply(pred.nb, c(1), maxidx)
actual_result <- ifelse(df$readmitted == df$readmitted[1], 0,1)
mean(idx.nb-1 == actual_result) # Accuracy: 52.15%

# Model 6: SVM
model.svm <- svm(readmitted~., data = train_data, kernel = "linear",
                 type = "C-classification", cross = 10, cost = 0.01, gamma = 1000)
pred.svm <- predict(model.svm, test_data, decision.values = F)
mean(pred.svm == test_data$readmitted) # Accuracy: 63.95%
train_data$diag_2 <- as.factor(train_data$diag_2)
plot(model.svm, train_data, readmitted~number_inpatient)

anova(model.svm, model.logit, test = "ChiSq")

####### Task 1: Time in hospital #######

train.2 <- sample(1:nrow(df.2), 0.8*nrow(df.2)) # Split the data into 80:20 ratio for cross validation
train_data.2 <- df.2[train.2,] # Training data
test_data.2 <- df.2[-train.2,] # Test data

# Model 1: Linear regression
model.lm.steps <- step(lm(time_in_hospital~., data=train_data.2), direction = "both")
model.task1.lm <- lm(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                             num_procedures + num_medications + number_outpatient + number_inpatient + 
                             diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                             metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                             diabetesMed + readmitted, data = train_data.2)
pred.task1.lm <- predict(model.task1.lm,test_data.2)
rmse(test_data.2$time_in_hospital,pred.task1.lm) #RMSE: 2.198489
rmsle(test_data.2$time_in_hospital,pred.task1.lm) #RMSLE: 0.4379768
View(cbind(test_data.2$time_in_hospital,pred.lm))

# Model 2: Neural networks
model.task1.nn <- nnet(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                               num_procedures + num_medications + number_outpatient + number_inpatient + 
                               diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                               metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                               diabetesMed + readmitted, data=train_data.2, size=5, maxit=1000)
pred.task1.nn <- predict(model.task1.nn, test_data.2)
rmse(test_data.2$time_in_hospital, pred.task1.nn) #4.061443
rmsle(test_data.2$time_in_hospital, pred.task1.nn) #0.9621453
View(cbind(test_data.2$time_in_hospital,pred.task1.nn))

# Model 3: Decision tree
model.task1.tree <- tree(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                 num_procedures + num_medications + number_outpatient + number_inpatient + 
                                 diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                 metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                 diabetesMed + readmitted, data = train_data.2)
pred.task1.tree <- predict(model.task1.tree, test_data.2)
plot(model.task1.tree)
text(model.task1.tree)
rmse(test_data.2$time_in_hospital,pred.task1.tree) #2.276108
rmsle(test_data.2$time_in_hospital,pred.task1.tree) #0.4512615
View(cbind(test_data.2$time_in_hospital,pred.task1.tree))

# Model 4: Random forest
model.task1.rf1 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=10, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf1) # % Var explained: 13.57%

model.task1.rf2 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=20, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf2) # % Var explained: 24.02%

model.task1.rf3 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=30, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf3) # % Var explained: 28.21%

model.task1.rf4 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=40, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf4) # % Var explained: 30.35%

model.task1.rf5 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=50, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf5) # % Var explained: 31.43%

model.task1.rf6 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=60, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf6) # % Var explained: 31.48%

model.task1.rf7 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=70, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf7) # % Var explained: 31.81%

model.task1.rf8 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=80, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf8) # % Var explained: 32.29%

model.task1.rf9 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=90, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf9) # % Var explained: 32.59%

model.task1.rf10 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                       num_procedures + num_medications + number_outpatient + number_inpatient + 
                                       diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                       metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                       diabetesMed + readmitted, 
                               data=train_data.2,ntree=100, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf10) # % Var explained: 32.64%

model.task1.rf20 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                        num_procedures + num_medications + number_outpatient + number_inpatient + 
                                        diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                        metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                        diabetesMed + readmitted, 
                                data=train_data.2,ntree=120, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf20) # % Var explained: 32.96%

model.task1.rf40 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                        num_procedures + num_medications + number_outpatient + number_inpatient + 
                                        diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                        metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                        diabetesMed + readmitted, 
                                data=train_data.2,ntree=140, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf40) # % Var explained: 33.02%

model.task1.rf60 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                        num_procedures + num_medications + number_outpatient + number_inpatient + 
                                        diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                        metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                        diabetesMed + readmitted, 
                                data=train_data.2,ntree=160, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf60) # % Var explained: 33.74%

model.task1.rf70 <-randomForest(time_in_hospital ~ race + gender + age + num_lab_procedures + 
                                        num_procedures + num_medications + number_outpatient + number_inpatient + 
                                        diag_1 + diag_2 + number_diagnoses + max_glu_serum + A1Cresult + 
                                        metformin + glipizide + glyburide + pioglitazone + rosiglitazone + 
                                        diabetesMed + readmitted, 
                                data=train_data.2,ntree=170, na.action=na.exclude, importance=T,proximity=T) 
print(model.task1.rf70) # % Var explained: 33.52%

mtry <- tuneRF(train_data.2, train_data.2$time_in_hospital, ntreeTry=160, stepFactor=1.5, 
               improve=0.01, trace=TRUE, plot=TRUE)
model.task1.rf <- randomForest(time_in_hospital~gender+readmitted+change+diabetesMed+glimepiride, 
                               data=train_data.2, ntree=160,mtry = 27, na.action=na.exclude, importance=T,
                               proximity=T) 
pred.task1.rf <- predict(model.task1.rf, test_data.2)
rmse(test_data.2$time_in_hospital,pred.task1.rf)#2.558702
rmsle(test_data.2$time_in_hospital,pred.task1.rf)#0.5073352
View(cbind(test_data.2$time_in_hospital,pred.task1.rf))

####### Task 2: Diagnoses #######
train.3 <- sample(1:nrow(df.3), 0.8*nrow(df.3)) # Split the data into 80:20 ratio for cross validation
train_data.3 <- df.3[train.3,] # Training data
test_data.3 <- df.3[-train.3,] # Test data

# Model 1: SVM
test_data.3$diag_3 <- as.factor(test_data.3$diag_3)
model.task2.svm.1 <- svm(diag_2~., train_data.3[,-c(1,14)])
pred.task2.svm.1 <- predict(model.task2.svm.1, newdata = test_data.3)
mean(pred.task2.svm.1==test_data.3$diag_2) # Accuracy: 38.67%

model.task2.svm.2 <- svm(diag_3~., train_data.3[,-c(1)])
pred.task2.svm.2 <- predict(model.task2.svm.2, newdata = test_data.3)
mean(pred.task2.svm.2==test_data.3$diag_3) # Accuracy: 39.44%

# Model 2: GBM
library(gbm)
n <- names(train_data.3[,-c(1,14,27)])
f <- as.formula(paste("diag_2 ~", paste(n[!n %in% c("diag_2")], collapse = " + ")))
train_data.3$diag_1 <- as.factor(train_data.3$diag_1)
train_data.3$diag_2 <- as.factor(train_data.3$diag_2)
train_data.3$diag_3 <- as.factor(train_data.3$diag_3)
model.task2.gbm <- gbm(f, data=train_data.3, n.trees=200, verbose=FALSE)
pred.task2.gbm<- predict(model.task2.gbm, newdata = test_data.3, n.trees = 200, type = "response")

classes <- vector()
for(i in 1:nrow(test_data.3)){
        classes[i] <- which.max(pred.task2.gbm[(19*(i-1)+1):(19*i)])
}
final_class <- classes
for(i in 1:length(classes)){
        if(classes[i]>15){final_class[i] <- classes[i] + 1}
}
predicted_gbm_final.1 <- as.factor(as.character(final_class))
result_gbm.1 <- check.model.accuracy(predicted_gbm_final.1, test_data.3$diag_2)
Score_gbm.1 <- mean(as.numeric(as.character(result_gbm.1[,4])), na.rm = TRUE)
print(Score_gbm.1) # Accuracy: 4.66%


n <- names(train_data.3[,-c(1,27)])
f <- as.formula(paste("diag_3 ~", paste(n[!n %in% c("diag_3")], collapse = " + ")))
model.task2.gbm.2 <- gbm(f, data=train_data.3, n.trees=200, verbose=FALSE)
pred.task2.gbm.2<- predict(model.task2.gbm.2, newdata = test_data.3, n.trees = 200, type = "response")

classes <- vector()
for(i in 1:nrow(test_data.3)){
        classes[i] <- which.max(pred.task2.gbm.2[(19*(i-1)+1):(19*i)])
}
final_class <- classes
for(i in 1:length(classes)){
        if(classes[i]>15){final_class[i] <- classes[i] + 1}
}
predicted_gbm_final.2 <- as.factor(as.character(final_class))
result_gbm.2 <- check.model.accuracy(predicted_gbm_final.2, test_data.3$diag_3)
Score_gbm.2 <- mean(as.numeric(as.character(result_gbm.2[,4])), na.rm = TRUE)
print(Score_gbm.2) # Accuracy: 4.072%

# Model 3: Random forest
model.task2.rf.1 <-randomForest(x=train_data.3[,-c(1,14)], y=train_data.3$diag_2, ntree=10, na.action=na.exclude, importance=T,
                   proximity=T) 
print(model.task2.rf.1)#23.17% error

model.task2.rf.2 <-randomForest(x=train_data.3[,-c(1,14)], y=train_data.3$diag_2, ntree=20, na.action=na.exclude, importance=T,
                   proximity=T) 
print(model.task2.rf.2)#14.69% error

model.task2.rf.3 <-randomForest(x=train_data.3[,-c(1,14)], y=train_data.3$diag_2, ntree=30, na.action=na.exclude, importance=T,
                   proximity=T) 
print(model.task2.rf.3)#10.43% error

model.task2.rf.4 <-randomForest(x=train_data.3[,-c(1,14)], y=train_data.3$diag_2, ntree=40, na.action=na.exclude, importance=T,
                   proximity=T) 
print(model.task2.rf.4)#9.88% error

model.task2.rf.5 <-randomForest(x=train_data.3[,-c(1,14)], y=train_data.3$diag_2, ntree=50, na.action=na.exclude, importance=T,
                   proximity=T) 
print(model.task2.rf.5)#6.1% error

model.task2.rf.6 <-randomForest(x=train_data.3[,-c(1,14)], y=train_data.3$diag_2, ntree=60, na.action=na.exclude, importance=T,
                   proximity=T) 
print(model.task2.rf.6)#6.13% error

mtry.2 <- tuneRF(train_data.3, train_data.3$diag_2, ntreeTry=50, stepFactor=1.5, 
               improve=0.01, trace=TRUE, plot=TRUE)
model.task2.rf <- randomForest(diag_2~., data=train_data.3, ntree=50,mtry = 27, na.action=na.exclude, importance=T,
                               proximity=T) 
pred.task2.rf <- predict(model.task2.rf, test_data.3)
mean(as.character(pred.task2.rf)== as.character(test_data.3$diag_2)) # 40.68%

mtry.3 <- tuneRF(train_data.3, train_data.3$diag_3, ntreeTry=50, stepFactor=1.5, 
                 improve=0.01, trace=TRUE, plot=TRUE)
model.task2.rf.2 <- randomForest(diag_3~., data=train_data.3, ntree=50,mtry = 5, na.action=na.exclude, importance=T,
                               proximity=T) 
pred.task2.rf.2 <- predict(model.task2.rf.2, test_data.3)
mean(as.character(pred.task2.rf.2)== as.character(test_data.3$diag_3)) # 38.15%

# Model 4: Artificial Neural Networks
model.task2.nn.1 <- nnet(train_data.3$diag_2 ~ ., data=train_data.3[,-c(1,14)], size=5, maxit=1000) 
pred.task2.nn.1 <- as.factor(predict(model.task2.nn.1,newdata = test_data.3, type = "class"))
mean(as.character(pred.task2.nn.1)==as.character(test_data.3$diag_2)) # 39.19%

model.task2.nn.2 <- nnet(train_data.3$diag_3 ~ ., data=train_data.3[,-c(1)], size=5, maxit=1000) 
pred.task2.nn.2 <- as.factor(predict(model.task2.nn.2,newdata = test_data.3, type = "class"))
mean(as.character(pred.task2.nn.2)==as.character(test_data.3$diag_3)) # 37.43%

####### Clustering #######
library(klaR)

#kClust <- kmodes(as.matrix(df[,5:15]),10)
df$readmitted <- ifelse(df$readmitted == "TRUE", 1, 0)
df.eu.dist <- dist(df[,c(12,27)], method = "euclidean")
hClust1 <- hclust(df.eu.dist, method = "ward.D2")
dend <- as.dendrogram(hClust1)
plot(hClust1)

kclus <- kmodes(df[,5:15],10)


####### Outlier detection #######
library(chemometrics)

md.1 <- Moutlier(as.matrix(df[,5]), quantile = .975, plot = T) # Time in hospital
md.2 <- Moutlier(as.matrix(df[,c(9,14)]), quantile = .975, plot = T) # Diagnoses
md.6 <- Moutlier(as.matrix(df[,6]), quantile = .975, plot = T)
md.7 <- Moutlier(as.matrix(df[,c(7,8)]), quantile = .975, plot = T)
md.8 <- Moutlier(as.matrix(df[,c(15)]), quantile = .975, plot = T)

df.2 <- df[md.1$md < md.1$cutoff,]
df.3 <- df[md.2$md < md.2$cutoff,]
boxplot(df[,c(5,6,7,8,9,11,15)], col = "blue")
dev.off()
jpeg("1.jpg")
ggplot(x = df[,c(5,6,7,8,15)]) +geom_boxplot()

####### Pattern mining #######
df.initial <- read.csv("10kDiabetes.csv", header = TRUE, strip.white = TRUE, na.strings = c("NA", "?"," ","."))
mining <- df
mining$age <- df.initial$age
mining$diag_1 <- as.factor(mining$diag_1)
mining$diag_2 <- as.factor(mining$diag_2)
mining$diag_3 <- as.factor(mining$diag_3)
mining[,c(1, 5, 6, 7, 8, 9, 10, 11, 15)] <- NULL
rules <- apriori(mining)
mining$readmitted <- as.factor(mining$readmitted)
rules1 <- apriori(mining, parameter=list(minlen=2,supp=0.005,conf=0.8),
                  appearance=list (rhs=c("readmitted=TRUE","readmitted=FALSE"), default="lhs"))
rules1.sort <- sort(rules1, by="lift")
subset.matrix<-is.subset(rules1.sort,rules1.sort)
subset.matrix[lower.tri(subset.matrix,diag=T)] <- 0
redudant<-colSums(subset.matrix) >= 1
rules1.pruned <- rules1.sort[!redudant]
plot(rules1.pruned)
rules.sub <- subset(rules1.pruned, subset = lhs %pin% "Yes" & rhs %pin% "FALSE")
plot(rules.sub,method="grouped",measure = "lift", control = list(main="Diabetes medicine",cex=.8,itemLabels=T,arrowSize=0))
plot(rules.sub,method="grouped",measure = "lift")
plot(rules.sub,method="paracoord",measure = "lift", control = list(main = "AfricanAmerican", reorder = F))

plot(rules1.pruned,method="grouped",measure = "lift", control = list(main="Set of rules",cex=.8,itemLabels=T,arrowSize=0))
##### Not working properly #####
#mining$diag_1 <- as.factor(mining$diag_1)
#mining$diag_2 <- as.factor(mining$diag_2)
#mining$diag_3 <- as.factor(mining$diag_3)
#mining[,c(1,4,7)] <- NULL
#r <- c("diag_2=1","diag_2=2","diag_2=3","diag_2=4","diag_2=5","diag_2=6","diag_2=7","diag_2=8","diag_2=9","diag_2=10","diag_2=11","diag_2=12","diag_2=13","diag_2=14","diag_2=15","diag_2=17","diag_2=18","diag_2=19","diag_2=20")
#rules1 <- apriori(mining, parameter=list(minlen=2,supp=0.005,conf=0.8),
#                 appearance=list (rhs=r,default="lhs"))
##### Not working properly #####