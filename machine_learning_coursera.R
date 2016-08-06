install.packages("dplyr")
library("dplyr")
library("caret")
install.packages("tidyr")
library("tidyr")
library(rpart)
install.packages("e1071")
library(e1071)

set.seed(54356)
#Loading
pml.training <- read.csv("pml-training.csv", na.strings = c("NA","#DIV/0!", ""), dec = ".")

#Cleaning
x <- pml.training %>% filter(new_window == "no")
x <- x[8:length(x)]
x <- x[ , ! apply(x ,2 ,function(x) any(is.na(x)) ) ]
#Cleaning Test & Training
inTrain <- createDataPartition(y=x$classe,p=0.6, list=FALSE)
trainingset <- subset(x[inTrain,])
testset <- subset(x[-inTrain,])
#DecisionTree Model
Ctrl <- trainControl(method = "repeatedcv", repeats = 3)
model_rpart <- train(classe ~ roll_belt + pitch_forearm + yaw_belt + roll_forearm + magnet_dumbbell_z + pitch_belt + magnet_dumbbell_y + magnet_dumbbell_x + accel_belt_z + magnet_belt_z, data=trainingset, method="rpart", tuneLength = 30, trControl = Ctrl)
model_rpart$finalModel
#RandomForest model
Ctrl <- trainControl(method = "oob")
model_rf <- train(classe ~ roll_belt + pitch_forearm + yaw_belt + roll_forearm + magnet_dumbbell_z + pitch_belt + magnet_dumbbell_y + magnet_dumbbell_x + accel_belt_z + magnet_belt_z, data=trainingset, method="rf", trControl = Ctrl, tuneGrid = data.frame(.mtry = 2))
model_rf
#Confusion Matrix for both models
#DecisionTree
predictions_rparttest <- predict(model_rpart, testset)
confusionMatrix(predictions_rparttest, testset$classe)[3]
#Graph for Decision Tree
conf_rpart <- as.data.frame(confusionMatrix(predictions_rparttest, testset$classe)[2])
conf_rpart <- conf_rpart %>% rename(prediction = table.Prediction, reference = table.Reference, count = table.Freq) %>% 
  arrange(desc(prediction)) %>% group_by(prediction) %>% mutate(prob = count/sum(count)) %>% ungroup
ggplot(conf_rpart, aes(reference, prediction)) + 
  geom_tile(aes(fill = prob), colour = "white") + 
  geom_text(aes(fill = prob, label = round(prob, 2)), size=3, colour="grey25") +
  scale_fill_gradient(low = "white", high = "red") +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0), limits = c("E","D","C","B","A")) 
#RandomForest
predictions_rftest <- predict(model_rf, testset)
confusionMatrix(predictions_rftest, testset$classe)[3]
#Graph for RandomForest
conf_rf <- as.data.frame(confusionMatrix(predictions_rftest, testset$classe)[2])
conf_rf <- conf_rf %>% rename(prediction = table.Prediction, reference = table.Reference, count = table.Freq) %>% 
  arrange(desc(prediction)) %>% group_by(prediction) %>% mutate(prob = count/sum(count)) %>% ungroup
ggplot(conf_rf, aes(reference, prediction)) + 
  geom_tile(aes(fill = prob), colour = "white") + 
  geom_text(aes(fill = prob, label = round(prob, 2)), size=3, colour="grey25") +
  scale_fill_gradient(low = "white", high = "red") +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0), limits = c("E","D","C","B","A")) 
#SampleError
model_rf$finalModel
confusionMatrix(predictions_rftest, testset$classe)[3]
#Predict using RF
bestfit <- model_rf

pml.submission <- read.csv("pml-testing.csv", na.strings = c("NA","#DIV/0!", ""), dec = ".")

predprob <- predict(bestfit, pml.submission, type = "prob")
predprob$testcase <- 1:nrow(predprob)
predprob <- gather(predprob, "class", "prob", 1:5)
ggplot(predprob, aes(testcase, class)) + 
  geom_tile(aes(fill = prob), colour = "white") + 
  geom_text(aes(fill = prob, label = round(prob, 2)), size=3, colour="grey25") +
  scale_fill_gradient(low = "white", high = "red") +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) 
final_predictions <- predict(bestfit, pml.submission)
final_predictions
#Generate File

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(final_predictions)