rm(list = ls(all = T))
getwd()

setwd("C:/Users/Hanu/Documents/insofe/internship/bank/bank-additional")


library(DMwR)
library(tidyr)
library(ggplot2)
library(e1071)
library(tidyverse)
library(dplyr)


library(caret)

bank_m<- read.csv("bank-additional-full.csv", sep = ";")

str(bank_m)

sum(is.na(bank_m))

df2<- filter(bank_m, age<75)
df2

distinct(df2, month)

select(df1, day_of_week)

df2<- arrange(df2, age)

df2



#------------------------------------------------------------------------
set.seed(123)

train_rows<- createDataPartition(df2$y, p = 0.7, list = F)

pre_train_data<- df2[train_rows,]

pre_test_data<- df2[-train_rows, ]

#-------------------------------------------------------------------------

std_m = preProcess(df2[, !(names(df2) %in% "y")], method = c("center","scale"))

train_data <- predict(std_m, pre_train_data)
test_data<- predict(std_m, pre_test_data)

summary(df2)

#--------------------------------------------------------------------------
library(randomForest)

model = randomForest(y~., data = train_data, keep.forest = TRUE, ntree = 50, max_features = "auto", min_samples_leaf = 50)

print(model)

rf_Imp_Attr = data.frame(model$importance)
rf_Imp_Attr = data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr) = c('Attributes', 'Importance')
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]

print(rf_Imp_Attr)

varImpPlot(model)

pred_Train = predict(model, 
                     train_data[,setdiff(names(train_data), "y")],
                     type="response", 
                     norm.votes=TRUE)

cm_Train = table("actual"= train_data$y, "predicted" = pred_Train);
accu_Train= sum(diag(cm_Train))/sum(cm_Train)

print(cm_Train)

print(accu_Train)

pred_Test = predict(model, test_data[,setdiff(names(test_data), "y")],
                    type="response", norm.votes=TRUE)
cm_Test = table("actual" = test_data$y, 
                "predicted" = pred_Test);
accu_Test_Imp = sum(diag(cm_Test))/sum(cm_Test)

print(cm_Test)

print(accu_Test_Imp)

confusionMatrix(pred_Test, test_data$y,mode = "prec_recall" ,positive = "yes" )

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


num_attr<- c("duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx","cons.conf.idx","euribor3m","nr.employed")
cat_attr_train<- setdiff(x = colnames(train_data), y = num_attr)

cat_attr_test<- setdiff(x = colnames(test_data), y = num_attr)

library(xgboost)
?xgboost

dummies <- dummyVars(~ cat_attr_train, data = train_data)