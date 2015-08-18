rm(list = ls())
library(caret)

#read data, remove columns with invalid or missing entries, remove timestamps and usernames, use all remaining as features
data<-read.csv("pml-training.csv", na.strings = c("NA", ""))
invalid <- unlist(lapply(data, function(x) any(is.na(x)) || any(is.nan(x)) || any(!is.numeric(x)) ))
validnames<-colnames(data)[!invalid]
validnames<-validnames[5:length(validnames)] #remove timestamps etc
data2<-subset(data,select=validnames)
subjects<-as.factor(data$user_name)
labels<-as.factor(data$classe)
full<-cbind(data2,labels)

# split into training and validation set, randomly drawing over all subjects
#alternative would be to split by subject if generalization over new subjects was the objective
set.seed(42)
trainsize = 0.75  #0.3  #0.75
trainid<-createDataPartition(y=full$labels,p=trainsize,list=FALSE)
train<-full[trainid,]
valid<-full[-trainid,]

# use all cores for crossvalidation inside caret::train
library(doParallel)
numcores<-detectCores()
cl <- makeCluster(numcores) 
registerDoParallel(cl)
getDoParWorkers()

# inside each training process, additionally perform n-fold crossvalidation
ctrl <- trainControl(method = "cv", number = numcores, allowParallel = TRUE)
svmTuneGrid <- data.frame( .C = c(0.01,0.1,0.5,1,2,4,8,16,32))  #only for svm based methods, soft error cost

# all methods use scale/center preprocessing (stored inside model for application at test time)

# nonlinear, tree-based ensemble
# random forest  1,98  0.3
model.rf <- train(labels~.,data=train,method="rf",preProcess=c("center","scale"),trControl = ctrl)

#nonlinear, robust against overfitting
# svmRadialCost  97,95 0.3  C=8
model.svmRadialCost <- train(labels~.,data=train, method="svmRadialCost",preProcess=c("center","scale"),tuneGrid = svmTuneGrid,trControl = ctrl)

#linear, fast, as baseline
#svmLinear  81, 79    0.3 C=8
model.svmLinear <- train(labels~.,data=train, method="svmLinear",preProcess=c("center","scale"),tuneGrid = svmTuneGrid,trControl = ctrl)

# nonlinear w boosting
# boosted trees 98,96 0.3
model.treeboost <- train(labels~.,data=train,method="gbm",preProcess=c("center","scale"),trControl = ctrl,verbose=FALSE)

#select the one model with the best validation accuracy
models<-list(model.rf,model.svmRadialCost,model.svmLinear ,model.treeboost );
bestvalid<-0
for (model in models) {
  trainest<-predict(model,newdata=train)  #for info only
  validest<-predict(model,newdata=valid)
  #table(trainest, train$labels)
  ctrain<-confusionMatrix(data = trainest, reference = train$labels)  #for info only
  cvalid<-confusionMatrix(data = validest, reference = valid$labels)
  if (bestvalid<cvalid$overall["Accuracy"]) {
    bestvalid<-cvalid$overall["Accuracy"]
    bestmodel<-model
    bestc<-cvalid
  }
}
print(bestc)
print(bestvalid)

# 0.3 training: 0.986  with rf

stopCluster(cl);

#visual confusion plot
data<-melt(bestc$table)
ggplot(data, aes(as.factor(Prediction), Reference, group=Reference)) +
       geom_tile(aes(fill = value)) + 
       geom_text(aes(fill = data$value, label = round(data$value, 1))) +
       scale_fill_gradient(low = "white", high = "red") 

# library(ROCR)  binary only
# pred <- prediction(trainest,train$labels)
# perf <- performance(pred,"prec","rec")
# plot(perf, avg= "threshold", colorize=T, lwd= 3,
#      main= "... Precision/Recall graphs ...")
# plot(perf, lty=3, col="grey78", add=T)

#test cases
testdata<-read.csv("pml-testing.csv", na.strings = c("NA", ""))
testdata2<-subset(testdata,select=validnames)

testest<-predict(bestmodel,newdata=testdata2)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(testest)

# B A B A A E D B A A B C B A E E A B B B