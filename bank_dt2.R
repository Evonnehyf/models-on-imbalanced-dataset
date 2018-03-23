
rm(list = ls(all = T))

getwd()

setwd("C:/Users/Hanu/Documents/insofe/internship/bank/bank-additional")

#-----------------------------------------------------------------------------------------


library(caret)
library(DMwR)
library(tidyr)
library(ggplot2)
library(e1071)
library(tidyverse)
library(dplyr)


bank_dt<- read.csv("bank-additional-full.csv", sep = ";")

bank_dt

str(bank_dt)

summary(bank_dt)

df<- distinct(bank_dt, age)

df


no_var<- nearZeroVar(bank_dt)

no_var

col_names<-colnames(bank_dt)


df1<-filter(bank_dt, age>75)

df1

df2<- filter(bank_dt, age<75)
df2

df3<- slice(bank_dt, 10:15)
df3

distinct(df2, month)

select(df1, day_of_week)

df2<- arrange(df2, age)

df2


#-------------------------------------univariate analysis--------------------------------

plot(x = df2$age, y = df2$euribor3m, "h" )

plot(x = df2$age, y = df2$duration, type = "h") #right skewed

kurtosis(df2$age)


sd(df2$age)

range(df2$age)

boxplot(df2$age)


df2 %>% 
  ggplot(aes(x = age, 
             y = duration)) + 
  geom_point(size = 2, alpha = 0.5)


df2 %>% 
  ggplot(aes(x = age,
             y = duration))  +
  geom_point(size = 2, 
             alpha = 0.5, aes(color = age)) + 
  labs(title = "age vs duration",
       x = "age",
       y = "duration per minutes") +
  theme_minimal()+scale_color_gradient(low = "red", high = "blue")




#----------------pre process----------------------------------------------------
#-------------------------------------------------------------------------------

pre = preProcess(df2, method = c("center","scale"))

pre = standardize(df2, centerFun = mean, scaleFun = sd)


num_attr<- c("duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx","cons.conf.idx","euribor3m","nr.employed")

cat_attr<- setdiff(x = colnames(df2), y = num_attr)

df2[,cat_attr] <- data.frame(apply(df2[,cat_attr], 2, function(x) as.factor(as.character(x))))

str(df2)

df_num <- df2[, num_attr]
df_cat<-df2[,cat_attr]

library(vegan)
df_stand<- decostand(x = df_num, method = "range")

df_stand2<- decostand(x = df_num, method = "standardize")
summary(df_stand2)

df_final <- cbind(df_stand2,df_cat)

head(df_final)

#--------------------------train&test split-------------------------------------
#-------------------------------------------------------------------------------

library(caret)

set.seed(123)

train_rows<- createDataPartition(df_final$y, p = 0.7, list = F)

train_data<- df_final[train_rows,]

test_data<- df_final[-train_rows, ]

#--------------------------------------model------------------------------

log_reg<- glm(y~., data = train_data, family = binomial)

summary(log_reg)

prob_train <- predict(log_reg, type = "response")

library(ROCR)

pred <- prediction(prob_train, train_data$y)

perf <- performance(pred, measure="tpr", x.measure="fpr")

plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))

perf_auc <- performance(pred, measure="auc")

# Access the auc score from the performance object

auc <- perf_auc@y.values[[1]]

print(auc)


prob_test <- predict(log_reg, test_data, type = "response")

preds_test <- ifelse(prob_test > 0.1, "yes", "no")


test_data_labs <- test_data$y

conf_matrix <- table(test_data_labs, preds_test)

print(conf_matrix)


#--------------------------------specificity---------------------------
specificity <- conf_matrix[1, 1]/sum(conf_matrix[1, ])

print(specificity)


#-------------------------------sensitivity---------------------------
sensitivity <- conf_matrix[2, 2]/sum(conf_matrix[2, ])

print(sensitivity)


#-------------------------------accuracy------------------------------
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)

print(accuracy)


library(caret)

# Using the argument "Positive", we can get the evaluation metrics according to our positive referene level

confusionMatrix(preds_test, test_data$y, positive = "yes")

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#---------------------------------decision tree----------------------------------

library(C50)

#Tree based model
c5_tree <- C5.0(y ~ . , train_data)

# Use the rules = T argument if you want to extract rules later from the model
#Rule based model
c5_rules <- C5.0(y ~ . , train_data, rules = T)

#variable importance
C5imp(c5_tree, metric = "usage")

summary(c5_rules)

plot(c5_tree)

library(caret)

preds<- predict(c5_tree, test_data)

confusionMatrix(preds, test_data$y, mode = "prec_recall" ,positive = "yes")

#------------------------------cart trees--------------------------------

library(rpart)
rpart_tree <- rpart(y ~ . , data = train_data, method="class", control=rpart.control(minsplit=30, cp=0.001))

rpart_tree$variable.importance

preds_rpart <- predict(rpart_tree, test_data,type="class")

printcp(rpart_tree)

plot(rpart_tree)
text(rpart_tree)

summary(rpart_tree)

library(caret)

confusionMatrix(preds_rpart, test_data$y,mode = "prec_recall" ,positive = "yes" )

#install.packages("rpart.plot")
library(rpart.plot)

plot(rpart_tree, uniform=TRUE,
     main="Classification Tree for term deposit")
text(rpart_tree, use.n=TRUE, all=TRUE, cex=.8)

rpart.plot(rpart_tree)

pfit<- prune(rpart_tree, cp= rpart_tree$cptable[which.min(rpart_tree$cptable[,"xerror"]),"CP"])

# plot the pruned tree
plot(pfit, uniform=TRUE,
     main="Pruned Classification Tree for bank data")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)
post(pfit, file = "c:/ptree.ps",
     title = "Pruned Classification Tree for bank data")

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

summary(train_data)




library(randomForest)

model = randomForest(y~., data = train_data, keep.forest = TRUE, ntree = 50)
