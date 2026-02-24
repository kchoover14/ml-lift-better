library(dplyr)        # data wrangling
library(GGally)       # extended ggplot2 visualizations e.g. ggpairs correlation matrix
library(caret)        # machine learning training and evaluation framework
library(rattle)       # GUI and visualization tools for machine learning
library(party)        # conditional inference trees, used by ctreeBag in bagging
library(rpart.plot)   # visualization for rpart decision trees
library(randomForest) # random forest algorithm
library(ggplot2)      # plots
library(gbm)          # gradient boosting models

##################### READ DATA
training <- read.csv('training.csv')
testing <- read.csv('testing.csv')
names(training)[1:10]

#################### CLEAN DATA
#remove columns unrelated to exercise and classe
training <- training[,-c(1:7)]; testing <- testing[,-c(1:7)]

#remove columns with high NA values
training <- training[, colSums(is.na(training))==0]
testing <- testing[, colSums(is.na(testing))==0]

#################### PARTITION
#set random seed for reproducibility
set.seed(12345)

#partition data into training and test sets
inTrain <- createDataPartition(y=training$classe,
                               p=0.6,list=FALSE)
myTraining <- training[inTrain,]; myTesting <- training[-inTrain,]

#make classe a factor
myTraining$classe <- factor(myTraining$classe)
myTesting$classe <- factor(myTesting$classe)


#################### RANDOM FOREST
control <- trainControl(method="cv", number=5)
fit_rf <- train(classe~., data=myTraining, method="rf",
                trControl=control, verbose=FALSE)

pred_rf <- predict(fit_rf, myTesting)
confusionMatrix(myTesting$classe, pred_rf)

ggplot(fit_rf)+ ggtitle("Random Forest Model Accuracy ~ Predictors")+
  theme_classic()
ggsave("rf-modelAccuracy.png",
       plot = last_plot(),
       width = 6,
       height = 6,
       units = "in",
       dpi = 300)

png("rf-errorVtrees.png", width=8, height=6, units="in", res=300)
plot(fit_rf$finalModel, main="Random Forest Model Error ~ Trees")
dev.off()

predict(fit_rf, newdata=testing)

png("rf-varImportance.png", width=8, height=6, units="in", res=300)
plot(varImp(fit_rf), main="Random Forest Variable Importance")
dev.off()

#################### DECISION TREE
#Control the computational nuances of the train function
control <- trainControl(method="cv", number=5)

#fit model using regression trees
fit_rpart <- train(classe ~. , data=myTraining, method='rpart', trControl = control)

pred_rpart <- predict(fit_rpart, myTesting)
confusionMatrix(myTesting$classe, pred_rpart)

#################### DECISION TREE BAGGING
predictors = myTraining[, -53]
classe = myTraining$classe
fit_bag <- train(predictors, classe, data=myTraining,
                 method='bag', B = 10,
                 bagControl = bagControl(fit=ctreeBag$fit,
                                         predict=ctreeBag$pred,
                                         aggregate=ctreeBag$aggregate))

pred_bag <- predict(fit_bag, myTesting)
confusionMatrix(myTesting$classe, pred_bag)

png("bag-varImportance.png", width=8, height=6, units="in", res=300)
plot(varImp(fit_bag), main="Bagged Tree Variable Importance")
dev.off()

#################### RANDOM FOREST BOOSTING
control <- trainControl(method="cv", number=5)
model_GBM <- train(classe~., data=myTraining, method="gbm",
                   trControl=control, verbose=FALSE)

pred_gbm <- predict(model_GBM, newdata=myTesting)
confusionMatrix(myTesting$classe, pred_gbm)

png("gbm-varImportance.png", width=8, height=6, units="in", res=300)
plot(varImp(model_GBM), main="GBM Variable Importance")
dev.off()
