# install.packages("minerva")
library(minerva)
library(gbm)

# dose-respose function on IPW
eveluation_weight <- function(trainData, testData, trainWeight,model = 'tree'){
  
  if(model == 'lm'){
    # SCM
    if(length(adj_name) == 0){
      formula_y <- 'y~poly(t,2)'
      
    } else{
      formula_y <- paste0('y~poly(t,5)+',paste0(paste0('poly(', adj_name,sep = ","),'5)',collapse="+"))
    }
    
    ind <- sample(1:dim(trainData)[1],replace = TRUE,prob = trainWeight)
    trainData <- trainData[ind,]
    
    model <- lm(formula = formula_y,data = trainData)
    predict_train <- model$fitted.values
    predict_test <- predict(model,newdata = testData)
    
    # MTEF
    ## MTEF of train
    newdata_train <- trainData
    newdata_train$t <- newdata_train$t - 1
    predict_train_detal <- predict(model,newdata = newdata_train)
    True_MTEF_train <- trainData['y'] - trainData['y_delta']
    prdict_MTEF_train <- predict_train - predict_train_detal
    error_MTEF_train <- True_MTEF_train - prdict_MTEF_train
    Rmse_MTEF_train <- mean((error_MTEF_train ^ 2)[,1])^ 0.5
    
    ## MTEF of test
    newdata_test <- testData
    newdata_test$t <- newdata_test$t - 1
    predict_test_detal <- predict(model,newdata = newdata_test)
    True_MTEF_test <- testData['y'] - testData['y_delta']
    prdict_MTEF_test <- predict_test - predict_test_detal
    error_MTEF_test <- True_MTEF_test - prdict_MTEF_test
    Rmse_MTEF_test <- mean((error_MTEF_test ^ 2)[,1])^ 0.5
  }
  
  if(model == 'tree'){
    
    if(length(adj_name) == 0){
      form <- formula('y~t')
    } else{
      form <- formula(paste0('y~t+', paste(adj_name,collapse="+"),collapse=""))
    }
    # print(form)
    
    ind <- sample(1:dim(trainData)[1],size = 800, replace = TRUE,prob = trainWeight)
    trainData <- trainData[ind,]
    
    model <- gbm(formula = form,data = trainData, distribution = "gaussian", 
                 n.trees = 100, shrinkage = 0.1,             
                 interaction.depth = 5, bag.fraction = 1.0, train.fraction = 0.8,  
                 n.minobsinnode = 80, cv.folds = 1, keep.data = TRUE, 
                 verbose = FALSE, n.cores = 1
    )
    best.iter <- gbm.perf(model, method = "test")
    print(best.iter)
    predict_train <- predict(model,newdata = trainData,n.trees = best.iter,type = 'response')
    predict_test <- predict(model,newdata = testData,n.trees = best.iter,type = 'response')
    
    # MTEF
    ## MTEF of train
    newdata_train <- trainData
    newdata_train$t <- newdata_train$t - 1
    predict_train_detal <- predict(model,newdata = newdata_train,n.trees = best.iter,type = 'response')
    True_MTEF_train <- trainData['y'] - trainData['y_delta']
    prdict_MTEF_train <- predict_train - predict_train_detal
    error_MTEF_train <- True_MTEF_train - prdict_MTEF_train
    Rmse_MTEF_train <- mean((error_MTEF_train ^ 2)[,1])^ 0.5
    
    ## MTEF of test
    newdata_test <- testData
    newdata_test$t <- newdata_test$t - 1
    predict_test_detal <- predict(model,newdata = newdata_test,n.trees = best.iter,type = 'response')
    True_MTEF_test <- testData['y'] - testData['y_delta']
    prdict_MTEF_test <- predict_test - predict_test_detal
    error_MTEF_test <- True_MTEF_test - prdict_MTEF_test
    Rmse_MTEF_test <- mean((error_MTEF_test ^ 2)[,1])^ 0.5
  }
  return(c(Rmse_MTEF_train,Rmse_MTEF_test))
  
}

# load data
##################################
# case1
fileName = './dataset/data-G-Case1.csv'
dataSource <- read.csv(fileName)
colname <- colnames(dataSource)
# split data
set.seed(20180808)
train_index <-  sort(sample(nrow(dataSource), nrow(dataSource)*.8))
data_train <- dataSource[train_index,]
data_test <-  dataSource[-train_index,]
## Total
XNum = 8
x_name <- paste0('X', seq(XNum));adj_name <- c()
## Ans
x_name <- c('X3');adj_name <- c('X4','X5')
##################################

##################################
# case2
fileName = './dataset/data-G-Case2.csv'
dataSource <- read.csv(fileName)
colname <- colnames(dataSource)
# split data
set.seed(20180810)
train_index <-  sort(sample(nrow(dataSource), nrow(dataSource)*.8))
data_train <- dataSource[train_index,]
data_test <-  dataSource[-train_index,]
## Total
XNum = 9
x_name <- paste0('X', seq(XNum));adj_name <- c()
## Ans
x_name <- c('X3');adj_name <- c('X6','X7','X8')
x_name <- c('X3');adj_name <- c('X7','X8')
##################################

##################################
# case3
fileName = './dataset/data-G-Case3.csv'
dataSource <- read.csv(fileName)
colname <- colnames(dataSource)
# split data
set.seed(20180810)
train_index <-  sort(sample(nrow(dataSource), nrow(dataSource)*.8))
data_train <- dataSource[train_index,]
data_test <-  dataSource[-train_index,]
## Total
XNum = 7
x_name <- paste0('X', seq(XNum));adj_name <- c()
## Ans
x_name <- c('X3');adj_name <- c('X4','X5')
##################################

##################################
# case4
fileName = './dataset/data-G-Case4.csv'
dataSource <- read.csv(fileName)
colname <- colnames(dataSource)
# split data
set.seed(20180809)
train_index <-  sort(sample(nrow(dataSource), nrow(dataSource)*.8))
data_train <- dataSource[train_index,]
data_test <-  dataSource[-train_index,]
## Total
XNum = 7
x_name <- paste0('X', seq(XNum));adj_name <- c()
## Ans
x_name <- c('X2','X3','X4');adj_name <- c('X3','X4','X5')
##################################

exNum = 100
##################################
# IPW
t_mean <- mean(data_train['t'][,1])  
t_sigma <- sd(data_train['t'][,1])

## acquire density value 
## dnorm(-6, mean = t_mean, sd = t_sigma) 

## p(t|x) line
formula_x <- paste(x_name,collapse="+")
formula_t <- paste0('t~', formula_x,collapse="")
meanF <- lm(formula = formula_t,data = data_train)
# summary(meanF)
t_x_mean <- meanF$fitted.values
residuals <- meanF$residuals
t_x_sigma <- sd(residuals)

## acquire density value 
## dnorm(-6, mean = t_x_mean, sd = t_x_sigma) 

## acquire weight
w_ipw <- c()
for (i in seq(1,length(data_train['t'][,1]))) {
  w_ipw[i] <- dnorm(data_train['t'][i,1], mean = t_mean, sd = t_sigma) / 
    dnorm(data_train['t'][i,1], mean = t_x_mean[i], sd = t_x_sigma)
}

w_ipw <- w_ipw / sum(w_ipw)

eveluation_ipw <- data.frame(Rmse_MTEF_train = c(),Rmse_MTEF_test = c())

for (i in seq(exNum)) {
  eveluation_ipw_ <- eveluation_weight(data_train,data_test,w_ipw)
  eveluation_ipw[i,'Rmse_MTEF_train'] <- eveluation_ipw_[1]
  eveluation_ipw[i,'Rmse_MTEF_test'] <- eveluation_ipw_[2]
}

summary(eveluation_ipw);sd(eveluation_ipw$Rmse_MTEF_train);sd(eveluation_ipw$Rmse_MTEF_test)
##################################

##################################
#gbm
# install.packages('gbm')
library(gbm)
formula_x <- paste(x_name,collapse="+")
formula_t <- paste0('t~', formula_x,collapse="")
form <- formula(formula_t)
model <- gbm(formula = form,
             data = data_train, shrinkage = 0.01,
             interaction.depth = 4, 
             distribution = 'gaussian',
             n.trees = 2000)
# summary(model)
f.t <- dnorm(data_train['t'][,1], mean = t_mean, sd = t_sigma)

best_cor <- 1
for (i in seq(50,500,by = 50)) {
  GBM.fitted <- predict(model, newdata = data_train, n.trees = i,type = 'response')
  
  ps.den <- dnorm((data_train$t-GBM.fitted)/sd(data_train$t-GBM.fitted),0,1)
  w_gbm <- f.t/ps.den
  w_gbm <- w_gbm / sum(w_gbm)
  cor_ <- c()
  for (j in seq(1,50)) {
    cor_[j]  <- weight_cor(data_train, w_gbm)
  }
  cor_mean <- mean(cor_)
  if (cor_mean < best_cor) {
    
    best_cor <- cor_mean
    best_tree <- i
    best_w_gbm <- w_gbm
  }
}

w_gbm <- best_w_gbm

eveluation_gbm <- data.frame(Rmse_MTEF_train = c(),Rmse_MTEF_test = c())

for (i in seq(exNum)) {
  eveluation_gbm_ <- eveluation_weight(data_train,data_test,w_gbm)
  eveluation_gbm[i,'Rmse_MTEF_train'] <- eveluation_gbm_[1]
  eveluation_gbm[i,'Rmse_MTEF_test'] <- eveluation_gbm_[2]
}

summary(eveluation_gbm);sd(eveluation_gbm$Rmse_MTEF_train);sd(eveluation_gbm$Rmse_MTEF_test)
##################################

##################################
# cbgps
library(CBPS)
CBPS_model <- CBPS(formula_t,data = data_train,
                   twostep = TRUE, 
                   method = "exact"
)
# summary(CBPS_model)

w_CBGPS <- CBPS_model$weights

eveluation_CBGPS <- data.frame(Rmse_MTEF_train = c(),Rmse_MTEF_test = c())

for (i in seq(exNum)) {
  eveluation_CBGPS_ <- eveluation_weight(data_train,data_test,w_CBGPS)
  eveluation_CBGPS[i,'Rmse_MTEF_train'] <- eveluation_CBGPS_[1]
  eveluation_CBGPS[i,'Rmse_MTEF_test'] <- eveluation_CBGPS_[2]
}

summary(eveluation_CBGPS);sd(eveluation_CBGPS$Rmse_MTEF_train);sd(eveluation_CBGPS$Rmse_MTEF_test)
##################################

##################################
#npCBGPS
npCBPS_model <- npCBPS(formula_t, data = data_train,corprior=.1/dim(data_train)[1], print.level=1) 
# summary(npCBPS_model)

w_npCBGPS <- npCBPS_model$weights

eveluation_npCBGPS <- data.frame(Rmse_MTEF_train = c(),Rmse_MTEF_test = c())
for (i in seq(exNum)) {
  eveluation_npCBGPS_ <- eveluation_weight(data_train,data_test,w_npCBGPS)
  eveluation_npCBGPS[i,'Rmse_MTEF_train'] <- eveluation_npCBGPS_[1]
  eveluation_npCBGPS[i,'Rmse_MTEF_test'] <- eveluation_npCBGPS_[2]
}

summary(eveluation_npCBGPS);sd(eveluation_npCBGPS$Rmse_MTEF_train);sd(eveluation_npCBGPS$Rmse_MTEF_test)
##################################