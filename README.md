# Data-Analysis-on-Hospital-Readmission-Data

Diabetes is a chronic condition prevalent among more than 25 million people across the world, affecting people of all ages. It can be said as a condition of the body where it cannot produce enough insulin to break down the sugar or it cannot use the insulin produced by the body. It can also be considered as a “slow poison”, which does not show its entire effects immediately but destroys the body step by step, rather slowly. Despite major advances in science and technology, diabetes continues to be a chronic disease, with a thirty-day readmission rate of around 20%, as compared to an average of 12% for the rest of the diseases. Additionally, readmissions cost hospitals a fair amount of money, so the end goal is to identify and reduce the possibility of a readmission. Prevention of patient readmission has been given a greater importance due to large cost involvement.

Objective: The primary objective of this project is to predict whether the patient will be readmitted to the hospital or not. We have used different classification methods for this purpose. Detailed explanation on the same is present in the below sections. Additionally, we created models to predict, 1) the time a patient is likely to spend in the hospital based on the preliminary diagnoses (data available on day 0), and, 2) future diagnoses (diagnoses 2 and 3).

This is a classification problem. To predict whether a patient will be readmitted or not, we created six different classifiers, namely, Support Vector Machines, Generalized logistic regression, Artificial Neural Networks, Random Forest Classifier, Naïve Bayes Classifier, Decision Trees.

Feature selection for these classes was done by conducting Correlation Analysis, and eliminating features with class imbalance.

Readmission rate was found to be high among the Caucasians. A1C test result was a good predictor of readmission (Normal results implies less chance of readmission).
