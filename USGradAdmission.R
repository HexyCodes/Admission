## ----loadlibraries, echo=F, inlcude=F, message=F--------------------------------------------------------------------
needed.packages <- c("tidyverse", "caret", "knitr", "MASS",
                     "ppcor" , "doParallel", "ggthemes", 
                     "corrplot", "glmnet", "car", "Rborist",
                     "caretEnsemble", "PerformanceAnalytics", 
                     "DataExplorer", "forecast", "kableExtra", "readr")

new.packages <- needed.packages[!(needed.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) 
  install.packages(new.packages)


library(tidyverse)
library(MASS)
library(forecast)
library(caretEnsemble)
library(caret)
library(DataExplorer)
library(ppcor)
library(doParallel)
library(ggthemes)
library(corrplot)
library(glmnet)
library(car)
library(Rborist)
library(kableExtra)
library(readr)
library(PerformanceAnalytics)


## ----download_data, results='asis'----------------------------------------------------------------------------------
url="https://raw.githubusercontent.com/HexyCodes/Admission/master/US_grad_admission.csv"
adm=read_csv(url)
dim(adm) # find the dimensions of the data.frame
colnames(adm)=c("Serial.No", "GRE.Score", "TOEFL.Score", "University.Rating", 
                "SOP", "LOR", "CGPA", "Research", "Chance.of.Admit")


## ----exploratory_analysis-------------------------------------------------------------------------------------------
class(adm)
head(adm,5) # look at the data type of columns
print(anyNA(adm))# check for missing values
summary(adm) # quantile distribution of the predicted values
adm%>%ggplot(aes(x=Chance.of.Admit))+geom_density(fill="blue") + theme_bw() 

  # distribution 
#pattern of the predicted value
qqnorm(adm$Chance.of.Admit)
 
 
adm%>%dplyr::select(-Serial.No)->adm # Remove Serial number from the daa. 
# distribution all predictors in the data frame
plot_density(adm, 
             geom_density_args = list("fill"="blue", 
                                      "alpha"=0.6), 
             ggtheme = theme_bw()) 
 


## ----dataexploration------------------------------------------------------------------------------------------------

any(is.na(adm)) # Checking for any missing values

#Function to check the outliers
IQR.outliers <- function(x) {
  if(any(is.na(x)))
    stop("x is missing values")
  if(!is.numeric(x))
    stop("x is not numeric")
  Q3<-quantile(x,0.75)
  Q1<-quantile(x,0.25)
  IQR<-(Q3-Q1)
  left<- (Q1-(1.5*IQR))
  print(left)

  right<- (Q3+(1.5*IQR))
    print(right)
  c(x[x <left],x[x>right])
}


#list of outliers
outliers=IQR.outliers(adm$Chance.of.Admit)





## ----correlation_analysis-------------------------------------------------------------------------------------------
# Converting the data into matrix format for conduction correlation analysis
data.matrix(adm)->adm_mat 
# plotting the hrent matrix results
plot_correlation(adm_mat) 
# plotting the correlation strength through size of squares.
corrplot(cor(adm_mat), method = "square") 


## ----boxplots_features----------------------------------------------------------------------------------------------
adm%>% 
  ggplot(aes(x=TOEFL.Score,y=Chance.of.Admit, fill=factor(University.Rating))) +
  geom_boxplot() +theme_bw()+
  ggtitle("Effects of University Rating and TOEFL scores on admission")

adm%>% 
  ggplot(aes(x=GRE.Score,
             y=Chance.of.Admit, fill=factor(University.Rating))) +
  geom_boxplot() +theme_bw()+
  ggtitle("Effects of University Rating and GRE scores on admission")

adm%>% 
  ggplot(aes(x=CGPA,y=Chance.of.Admit, fill=factor(University.Rating))) +
  geom_boxplot() +theme_bw()+
  ggtitle("Effects of University Rating and CGPA scores on admission")





## ----partialcor-----------------------------------------------------------------------------------------------------
partials=pcor(adm_mat) # Conducting partial correlation analysis
print("Partial Correlations for the Dependent Variable: Rent")
Estimates=data.frame(partials$estimate[,7:8])
P.values=data.frame(partials$p.value[, 7:8])
# printing the results
kable((Estimates), format = "pandoc", digits=2, 
      caption="Partial correlations  of the input dataset features") 

kable((P.values), format = "pandoc", digits=2, 
      caption="Probability values  of the input dataset features") 




## ----data_cleansing-------------------------------------------------------------------------------------------------
# data cleansed after removing SOP
adm%>%dplyr::select(-SOP, -University.Rating)->admfea 


## ----data_splitting-------------------------------------------------------------------------------------------------
set.seed(1, sample.kind = "Rounding")
test_indices=createDataPartition(admfea$Chance.of.Admit, 
                                 times=1, 
                                 p=0.5, # portion of data split into test
                                 list=F)
admfea[-test_indices,]->traindf # dataset reserved for training
admfea[test_indices,]->valdf # dataset held for validation


## ----tuning_demo, warning=F-----------------------------------------------------------------------------------------

# cl=makePSOCKcluster(detectCores())
# registerDoParallel(cl)
# #Create control function for training with 10 folds
# #and keep 3 folds for training. search method is grid.
# 
# control <- trainControl(method='repeatedcv',
#                         number=10,
#                         repeats=3,
#                         search='grid')
# 
# tunegrid <- expand.grid(predFixed = c(1:5), minNode=1:3)
# rf_gridsearch <- train(Chance.of.Admit ~ .,
#                        data = trainset,
#                        method = 'Rborist',
#                        metric = c("RMSE"),
#                       tuneGrid = tunegrid)
# print(rf_gridsearch)
# stopCluster(cl)


## ----glm------------------------------------------------------------------------------------------------------------
#General linear model
mod.lm=lm(Chance.of.Admit~., data=traindf) 
summary(mod.lm)
#predicting results from the model
pred.lm=predict(mod.lm, newdata=valdf)
RMSE(pred.lm, valdf$Chance.of.Admit)

#Stepwise Regression procedure
stepAIC(mod.lm, direction="both")

#Predicting results from stepwise regression
mod.lm.step=lm(Chance.of.Admit~GRE.Score+
                 TOEFL.Score+LOR+CGPA, data=traindf)
summary(mod.lm.step)



## ----MachineLearningModels------------------------------------------------------------------------------------------
set.seed(1, sample.kind = "Rounding")
 
cl=makePSOCKcluster(detectCores()-1) #setting up clusters for parallel processing
registerDoParallel(cl) # register the clusters for parellel processing

my.con=trainControl(method="cv", # choose cross validation
                    number=3, # number of times the process is run
                    savePredictions = "final", allowParallel = T)
models=caretList(Chance.of.Admit~., data=traindf,
trainControl=my.con, 
methodList = c("Rborist",  # list of ML models chosen
               "knn", 
               "glmnet",
               "xgbLinear",
               "brnn", 
               "ridge"), 
continue_on_fail = T) # making sure all models are running without error
models$xgbLinear
models$knn
models$Rborist
models$glmnet
models$brnn
stopCluster(cl)
registerDoSEQ()
rm(cl) # removing cluster
varImp(models$Rborist) # variable importance for Rborist
varImp(models$glmnet)






## ----feature_reduction----------------------------------------------------------------------------------------------
set.seed(1, sample.kind = "Rounding")

cl=makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)
 
ctrl=rfeControl(functions = rfFuncs, #random forest function 
           method = "cv", # cross validation
           number = 2, # times
           verbose = F)
subsets=c(1:5)
registerDoSEQ() # avoid warning of parallel clusters not existing
lmProfile=rfe(as.matrix(traindf[,-6]), 
              as.matrix(traindf[,6]), 
              sizes=subsets, 
              rfeControl=ctrl) # recursive feature elimination

lmProfile
plot(lmProfile)


## ----cross_validation_results---------------------------------------------------------------------------------------

 cl=makePSOCKcluster(detectCores()-1)
 registerDoParallel(cl)
 #retrieve resamples from cross validation
 resamples<-resamples(models)
 dotplot(resamples, metric="RMSE")
 summary(resamples)



## ----ensemble_output, warning=F-------------------------------------------------------------------------------------
cl=makePSOCKcluster(detectCores())
registerDoParallel(cl)
# 
ens=caretEnsemble(models, metric="RMSE", 
                  trControl=my.con) # Run Ensemble model to gather the best result
summary(ens)
plot(ens)




## ----validation_results---------------------------------------------------------------------------------------------
######### Results- Validation ############

#Prediction from Linear model 
pred.lm.step=predict(mod.lm.step, newdata = valdf) 

#Predicted from each ML model

 predicted.Rborist=predict(models$Rborist, newdata=valdf)
 predicted.knn=predict(models$knn, newdata=valdf)
 predicted.glmnet=predict(models$glmnet, newdata=valdf)
 predicted.xgbLinear=predict(models$xgbLinear, newdata=valdf)
 predicted.brnn=predict(models$brnn, newdata=valdf)
 predicted.ridge=predict(models$ridge, newdata=valdf)
 
 
 
 
#Prediction from Ensemble of the models
 predicted.ens=predict(ens, newdata=valdf)
 
  
 # Construct the table to output th results.
 data.frame(Rborist=RMSE(predicted.Rborist, valdf$Chance.of.Admit), 
            Ensemble=RMSE(predicted.ens, valdf$Chance.of.Admit),
            Linear=RMSE(pred.lm.step, valdf$Chance.of.Admit))->rmses
 
 data.frame(Rborist=cor(predicted.Rborist, valdf$Chance.of.Admit), 
            Ensemble=cor(predicted.ens, valdf$Chance.of.Admit),
            Linear=cor(pred.lm.step, valdf$Chance.of.Admit))->r2s
 
 Results=data_frame(Model="Linear-Stepwise-Selected", 
                    RMSE=round(RMSE(pred.lm.step, valdf$Chance.of.Admit), 3), 
                    R2=round(cor(pred.lm.step, valdf$Chance.of.Admit), 3))
 

Results[2,]=list(Model="knn",
                    RMSE=round(RMSE(predicted.knn, valdf$Chance.of.Admit), 3), 
                    R2=round(cor(predicted.knn, valdf$Chance.of.Admit), 3))
Results[3,]=list(Model="GLMnet", 
                    RMSE=round(RMSE(predicted.glmnet, valdf$Chance.of.Admit), 3), 
                    R2=round(cor(predicted.glmnet, valdf$Chance.of.Admit), 3))

Results[4,]=list( Model="XGBLinear", 
                    RMSE=round(RMSE(predicted.xgbLinear, valdf$Chance.of.Admit), 3), 
                    R2=round(cor(predicted.xgbLinear, valdf$Chance.of.Admit), 3))
Results[5,]=list(Model="brnn", 
                    RMSE=round(RMSE(predicted.brnn, valdf$Chance.of.Admit), 3), 
                    R2=round(cor(predicted.brnn, valdf$Chance.of.Admit), 3))
Results[6,]= list(Model="Ridge Regression", 
                    RMSE=round(RMSE(predicted.ridge, valdf$Chance.of.Admit), 3), 
                    R2=round(cor(predicted.ridge, valdf$Chance.of.Admit), 3))

Results[7,]= list(Model="Ensemble", 
                    RMSE=round(RMSE(predicted.ens, valdf$Chance.of.Admit), 3), 
                    R2=round(cor(predicted.ens, valdf$Chance.of.Admit), 3))
Results[8,]=list(Model="Rborist", 
                    RMSE=round(RMSE(predicted.Rborist, valdf$Chance.of.Admit), 3), 
                    R2=round(cor(predicted.Rborist, valdf$Chance.of.Admit), 3))
 


 
kable(Results, format="pandoc", 
      digits=3, caption = "Performance of the different models: RMSE and R2 values" ) 

Results%>% ggplot(aes(reorder(Model, -RMSE), RMSE, fill=Model))+geom_bar(stat="identity")+theme_bw()+
  theme(axis.text.x = element_text(angle=90))+xlab("Model")
 

