setwd("~/SoilSalinityMapColombia_RF_LIME_Unc")


setwd("G:\\My Drive\\IGAC_2020\\SALINIDAD\\R_CODES")
library(vip) #variable importance plot
library(pdp) #partial dependence plot
library(cowplot) #plots 
library(plyr) 
library(readxl)
library(raster)
library(rgdal)
library(sp)
library(caret)
library(dplyr)
library(doParallel)
library(ranger)
library(GSIF)
library(aqp)
library(lime)
library(caret)#neural network
library(reticulate)
library(magrittr) #pipe operator
library(randomForest)
library(Metrics)
library(quantregForest)


## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:
  
  
### Data and covariates loading
  
data <- read.csv('G:\\My Drive\\IGAC_2020\\SALINIDAD\\R_CODES\\PRUEBAPILOTOCVC\\RegMatrix_VF_COL.csv')
data <- data[,-c(1:3,5:7,52:73,75:108,110:119,122,124:127,129:134)]%>% na.omit
#(na_count <-sapply(data, function(y) sum(is.na(y))) %>% data.frame) 
cov <-stack("G:\\My Drive\\IGAC_2020\\SALINIDAD\\INSUMOS\\COVARIABLES\\COV_SSMAP\\CovSSMAP_16062020.tif")
names(cov) <- readRDS("G:\\My Drive\\IGAC_2020\\SALINIDAD\\INSUMOS\\COVARIABLES\\COV_SSMAP\\NamesCovSSMAP_16062020.rds")
cov <- cov[[names(data)[-1]]]


### Checking covariates redundancy in data loaded

cormat <- cor(data[,c(1:44,48,49)]) %>% data.frame
#write.csv(cormat,"cormat.csv")


### Data splitting into training and testing datasets

set.seed(524)
inTrain <- createDataPartition(y = data$pH.0_30, p = .70, list = FALSE)
train_data <- data[ inTrain,]
dim(train_data)
test_data <- data[-inTrain,]
dim(test_data)


### Recursive feature elimination

#### Warning: it may take time!
cl <- makeCluster(detectCores(-2), type='PSOCK')
registerDoParallel(cl)
control2 <- rfeControl(functions=rfFuncs, method="repeatedcv", number=5, repeats=5)
(rfmodel <- rfe(x=data[,-1], y=data[,1], sizes=c(1:10), rfeControl=control2))
plot(rfmodel, type=c("g", "o"))
predictors(rfmodel)[c(1:10)]



fm <-  as.formula(paste0("pH.0_30~",paste0(as.character(predictors(rfmodel)[c(1:5)]),collapse = "+")))
fm
modelo_ranger <- ranger(
  formula=fm,
  data=train_data,
  num.trees = 500,
  mtry = 2,
  importance = "impurity"
)

vimp <- data.frame(variables=predictors(modelo_ranger), 
                   importance=as.vector(modelo_ranger$variable.importance))
ggplot(vimp, aes(x=reorder(variables,importance), y=importance,fill=importance))+ 
  geom_bar(stat="identity", position="dodge")+ coord_flip()+
  ylab("Variable Importance")+
  xlab("Variables")+
  ggtitle("Variable importance plot")



pred <- predict(modelo_ranger, test_data)
(AVE <- 1 - sum((pred$predictions-test_data$pH.0_30)^2, na.rm=TRUE)/
    sum((test_data$pH.0_30 - mean(test_data$pH.0_30, na.rm = TRUE))^2,
        na.rm = TRUE))
Metrics::rmse(pred$predictions,test_data$pH.0_30)
cor(pred$predictions,test_data$pH.0_30)



ggplot(data.frame("pH0_30"=test_data$pH.0_30, "pred"=pred$predictions), aes(pH0_30, pred)) + 
  geom_point() + 
  geom_abline(slope=1, intercept=0) +
  scale_y_continuous(labels = scales::comma) +
  scale_x_continuous(labels = scales::comma)



ggplot(data.frame("pH0_30"=test_data$pH.0_30, "residual"=test_data$pH.0_30 - pred$predictions),
       aes(pH0_30, residual)) + 
  geom_point() + 
  geom_abline(slope=0, intercept=0) +
  scale_y_continuous(labels = scales::comma) +
  scale_x_continuous(labels = scales::comma)



td.sort <- train_data[order(train_data$pH.0_30),]
td.sort <- td.sort[,predictors(rfmodel)[c(1:5)]]
explainer <- lime(td.sort, modelo_ranger)


# Explaining selecting samples

explanation_ini <- explain(td.sort[1:5,],
                           explainer,
                           n_permutations = 5000,
                           dist_fun = "euclidean",
                           n_features = 5,
                           feature_select = "highest_weights")




explanation_fin <- explain(td.sort[317:321,],
                           explainer,
                           n_permutations = 5000,
                           dist_fun = "euclidean",
                           n_features = 5,
                           feature_select = "highest_weights")



# Visualising the model explanations


plot_features(explanation_ini)


plot_features(explanation_fin)


plot_explanations(explanation_ini)


plot_explanations(explanation_fin)



#### Warning: it may take time!
rf <- randomForest(x = data[,predictors(rfmodel)[c(1:5)]], y =data[,1],ntree=500,mtry=2)
pred <- predict(cov[[predictors(rfmodel)[c(1:5)]]],rf)
#,rf,filename = "RF_pH0.30_COL_.tif",format = "GTiff", overwrite = T)
writeRaster(pred,"ESP.0_30.tif")
plot(pred)


### Uncertainty by using Quantile Random Forest

dat <- read.csv('G:\\My Drive\\IGAC_2020\\SALINIDAD\\R_CODES\\PRUEBAPILOTOCVC\\RegMatrix_VF_COL.csv')
dat <- dat[,-c(1,5:7,52:73,75:108,110:119,122,124:127,129:134)]%>% na.omit
coordinates(dat) <- ~ longitude + latitude
dat@proj4string <- proy <- CRS("+proj=longlat +datum=WGS84")
dat <- spTransform (dat, CRS=projection(cov))

ctrl <- trainControl(method = "cv", savePred=T)
validation <- data.frame(rmse=numeric(), r2=numeric())
i=1
for (i in 1:10){
  # We will build 10 models using random samples of 25%
  smp_size <- floor(0.25 * nrow(dat))
  train_ind <- sample(seq_len(nrow(dat)), size = smp_size)
  train <- dat[train_ind, ]
  test <- dat[-train_ind, ]
  modn <- train(fm, data=train@data, method = "rf",
                trControl = ctrl)
  pred <- stack(pred, predict(cov, modn))
  test$pred <- extract(pred[[i+1]], test)
  # Store the results in a dataframe
  validation[i, 1] <- rmse(test$pH.0_30, test$pred)
  validation[i, 2] <- cor(test$pH.0_30, test$pred)^2
}
