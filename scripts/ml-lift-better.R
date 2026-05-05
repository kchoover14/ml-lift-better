library(dplyr)        # data wrangling
library(tidyr)        # reshape data from wide to long
library(GGally)       # extended ggplot2 visualizations (e.g., ggpairs correlation matrix)
library(caret)        # machine learning training and evaluation framework
library(rattle)       # GUI and visualization tools for machine learning
library(party)        # conditional inference trees, used by ctreeBag in bagging
library(rpart.plot)   # visualization for rpart decision trees
library(randomForest) # random forest algorithm
library(ggplot2)      # plots
library(gbm)          # gradient boosting models
library(viridis)      # color-blind friendly color palettes


######################## READ DATA

training = read.csv('data/training.csv')
testing  = read.csv('data/testing.csv')


######################## CLEAN DATA

# Columns 1-7 are metadata unrelated to exercise (row ID, participant name,
# timestamps, window flags) -- remove by name to avoid hard-coded index dependence
meta_cols = c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",
              "cvtd_timestamp", "new_window", "num_window")

training = training[, !names(training) %in% meta_cols]
testing  = testing[,  !names(testing)  %in% meta_cols]

# Remove columns with any NA values
training = training[, colSums(is.na(training)) == 0]
testing  = testing[,  colSums(is.na(testing))  == 0]


######################## PARTITION

# Set random seed for reproducibility
set.seed(12345)

# Partition 60/40 into training and test sets, stratified on classe
inTrain    = createDataPartition(y = training$classe, p = 0.6, list = FALSE)
myTraining = training[inTrain, ]
myTesting  = training[-inTrain, ]

# Make classe a factor
myTraining$classe = factor(myTraining$classe)
myTesting$classe  = factor(myTesting$classe)


######################## RANDOM FOREST

control = trainControl(method = "cv", number = 5)

fit_rf = train(classe ~ ., data = myTraining, method = "rf",
               trControl = control, verbose = FALSE)

pred_rf = predict(fit_rf, myTesting)
cm_rf   = confusionMatrix(myTesting$classe, pred_rf)

# Export confusion matrix and per-class statistics
write.csv(as.data.frame(cm_rf$table), "outputs/rf-confusion-matrix.csv", row.names = FALSE)
write.csv(as.data.frame(cm_rf$byClass), "outputs/rf-stats-byclass.csv")

# Accuracy by number of predictors
ggplot(fit_rf) +
  theme_classic()
ggsave("outputs/rf-modelAccuracy.png",
       plot   = last_plot(),
       width  = 6,
       height = 6,
       units  = "in",
       dpi    = 300)

# Error by number of trees with class legend
rf_colors = viridis(6)
png("outputs/rf-errorVtrees.png", width = 8, height = 6, units = "in", res = 300)
plot(fit_rf$finalModel, col = rf_colors)
legend("topright",
       legend = c("OOB", "A", "B", "C", "D", "E"),
       col    = rf_colors,
       lty    = 1,
       cex    = 0.8)
dev.off()

# Predict on held-out test set
predict(fit_rf, newdata = testing)

# Variable importance
rf_imp = varImp(fit_rf)$importance
rf_imp$predictor = rownames(rf_imp)
rf_imp = rf_imp[order(rf_imp$Overall), ]
rf_imp$predictor = factor(rf_imp$predictor, levels = rf_imp$predictor)

ggplot(rf_imp, aes(x = Overall, y = predictor)) +
  geom_segment(aes(x = 0, xend = Overall, y = predictor, yend = predictor), color = "grey60") +
  geom_point(color = viridis(3)[2], size = 2) +
  labs(x = "Importance", y = NULL) +
  theme_classic() +
  theme(axis.text.y = element_text(size = 6.5))
ggsave("outputs/rf-varImportance.png",
       plot   = last_plot(),
       width  = 8,
       height = 10,
       units  = "in",
       dpi    = 300)


######################## DECISION TREE

control = trainControl(method = "cv", number = 5)

fit_rpart = train(classe ~ ., data = myTraining, method = "rpart",
                  trControl = control)

pred_rpart = predict(fit_rpart, myTesting)
cm_rpart   = confusionMatrix(myTesting$classe, pred_rpart)

# Export confusion matrix and per-class statistics
write.csv(as.data.frame(cm_rpart$table),
          "outputs/rpart-confusion-matrix.csv", row.names = FALSE)
write.csv(as.data.frame(cm_rpart$byClass),
          "outputs/rpart-stats-byclass.csv")


######################## DECISION TREE -- BAGGING

# Separate predictors from outcome to pass to bag method
predictors = myTraining[, names(myTraining) != "classe"]
classe     = myTraining$classe

fit_bag = train(predictors, classe,
                method     = "bag",
                B          = 10,
                bagControl = bagControl(fit       = ctreeBag$fit,
                                        predict   = ctreeBag$pred,
                                        aggregate = ctreeBag$aggregate))

pred_bag = predict(fit_bag, myTesting)
cm_bag   = confusionMatrix(myTesting$classe, pred_bag)

# Export confusion matrix and per-class statistics
write.csv(as.data.frame(cm_bag$table),
          "outputs/bag-confusion-matrix.csv", row.names = FALSE)
write.csv(as.data.frame(cm_bag$byClass),
          "outputs/bag-stats-byclass.csv")

# Variable importance -- faceted by class
bag_imp = varImp(fit_bag)$importance
bag_imp$predictor = rownames(bag_imp)

bag_long = tidyr::pivot_longer(bag_imp,
                               cols      = -predictor,
                               names_to  = "Class",
                               values_to = "Importance")

# Order predictors by mean importance across classes for consistent y-axis
predictor_order = bag_imp |>
  dplyr::mutate(mean_imp = rowMeans(dplyr::across(-predictor))) |>
  dplyr::arrange(mean_imp) |>
  dplyr::pull(predictor)

bag_long$predictor = factor(bag_long$predictor, levels = predictor_order)

ggplot(bag_long, aes(x = Importance, y = predictor)) +
  geom_segment(aes(x = 0, xend = Importance, y = predictor, yend = predictor), color = "grey60") +
  geom_point(color = viridis(3)[2], size = 1) +
  facet_wrap(~ Class, nrow = 2) +
  labs(x = "Importance", y = NULL) +
  theme_classic() +
  theme(axis.text.y = element_text(size = 5),
        strip.background = element_blank(),
        strip.text = element_text(size = 9, face = "bold"))
ggsave("outputs/bag-varImportance.png",
       plot   = last_plot(),
       width  = 12,
       height = 10,
       units  = "in",
       dpi    = 300)


######################## RANDOM FOREST -- GRADIENT BOOSTING

control = trainControl(method = "cv", number = 5)

model_GBM = train(classe ~ ., data = myTraining, method = "gbm",
                  trControl = control, verbose = FALSE)

pred_gbm = predict(model_GBM, newdata = myTesting)
cm_gbm   = confusionMatrix(myTesting$classe, pred_gbm)

# Export confusion matrix and per-class statistics
write.csv(as.data.frame(cm_gbm$table),
          "outputs/gbm-confusion-matrix.csv", row.names = FALSE)
write.csv(as.data.frame(cm_gbm$byClass),
          "outputs/gbm-stats-byclass.csv")

# Variable importance
gbm_imp = varImp(model_GBM)$importance
gbm_imp$predictor = rownames(gbm_imp)
gbm_imp = gbm_imp[order(gbm_imp$Overall), ]
gbm_imp$predictor = factor(gbm_imp$predictor, levels = gbm_imp$predictor)

ggplot(gbm_imp, aes(x = Overall, y = predictor)) +
  geom_segment(aes(x = 0, xend = Overall, y = predictor, yend = predictor), color = "grey60") +
  geom_point(color = viridis(3)[2], size = 2) +
  labs(x = "Importance", y = NULL) +
  theme_classic() +
  theme(axis.text.y = element_text(size = 6.5))
ggsave("outputs/gbm-varImportance.png",
       plot   = last_plot(),
       width  = 8,
       height = 10,
       units  = "in",
       dpi    = 300)


######################## MODEL COMPARISON SUMMARY

model_summary = data.frame(
  Model    = c("Decision Tree (rpart)", "Bagged Tree", "GBM", "Random Forest"),
  Accuracy = c(cm_rpart$overall["Accuracy"], cm_bag$overall["Accuracy"],
               cm_gbm$overall["Accuracy"], cm_rf$overall["Accuracy"]),
  CI_low   = c(cm_rpart$overall["AccuracyLower"], cm_bag$overall["AccuracyLower"],
               cm_gbm$overall["AccuracyLower"], cm_rf$overall["AccuracyLower"]),
  CI_high  = c(cm_rpart$overall["AccuracyUpper"], cm_bag$overall["AccuracyUpper"],
               cm_gbm$overall["AccuracyUpper"], cm_rf$overall["AccuracyUpper"]),
  Kappa    = c(cm_rpart$overall["Kappa"], cm_bag$overall["Kappa"],
               cm_gbm$overall["Kappa"], cm_rf$overall["Kappa"]),
  OOS_Error = 1 - c(cm_rpart$overall["Accuracy"], cm_bag$overall["Accuracy"],
                    cm_gbm$overall["Accuracy"], cm_rf$overall["Accuracy"])
)

write.csv(model_summary, "outputs/model-comparison.csv", row.names = FALSE)


######################## TIDY

# rm(list = ls())  # clear environment
# gc()             # release memory