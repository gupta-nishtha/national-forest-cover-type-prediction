library(caret)
library(rpart)
library(rpart.plot)
library(rattle) #pretty rpart plot
library(randomForest) #rf
library(caret) #knn
library(e1071) #naive bayes
library(dplyr)

source('/Users/jacobnyamu/Desktop/Fall 2022/Machine Learning /Data/BabsonAnalytics.r')

#LOAD 
df <- read.csv("/Users/jacobnyamu/Desktop/Fall 2022/Machine Learning /forest-cover-type-prediction/train.csv")

#MANAGE 
## Check for missing values in the data frame
sum(is.na(df))

#MANAGE
df$Id <- NULL
df$Hillshade_9am <- as.numeric(df$Hillshade_9am)
df$Hillshade_Noon <- as.numeric(df$Hillshade_Noon)
df$Hillshade_3pm <- as.numeric(df$Hillshade_3pm)
df$Wilderness_Area1 <- as.factor(df$Wilderness_Area1)
df$Wilderness_Area2 <- as.factor(df$Wilderness_Area2)
df$Wilderness_Area3 <- as.factor(df$Wilderness_Area3)
df$Wilderness_Area4 <- as.factor(df$Wilderness_Area4)
df$Soil_Type1 <- as.factor(df$Soil_Type1)
df$Soil_Type2 <- as.factor(df$Soil_Type2)
df$Soil_Type3 <- as.factor(df$Soil_Type3)
df$Soil_Type4 <- as.factor(df$Soil_Type4)
df$Soil_Type5 <- as.factor(df$Soil_Type5)
df$Soil_Type6 <- as.factor(df$Soil_Type6)
# Dropping the column as it contains 1 unique value 
df$Soil_Type7 <- as.factor(df$Soil_Type7)
df$Soil_Type8 <- as.factor(df$Soil_Type8)
df$Soil_Type9 <- as.factor(df$Soil_Type9)
df$Soil_Type10 <- as.factor(df$Soil_Type10)
df$Soil_Type11 <- as.factor(df$Soil_Type11)
df$Soil_Type12 <- as.factor(df$Soil_Type12)
df$Soil_Type13 <- as.factor(df$Soil_Type13)
# Dropping the column as it contains 1 unique value 
df$Soil_Type15 <- as.factor(df$Soil_Type15)
df$Soil_Type16 <- as.factor(df$Soil_Type16)
df$Soil_Type17 <- as.factor(df$Soil_Type17)
df$Soil_Type18 <- as.factor(df$Soil_Type18)
df$Soil_Type19 <- as.factor(df$Soil_Type19)
df$Soil_Type20 <- as.factor(df$Soil_Type20)
df$Soil_Type21 <- as.factor(df$Soil_Type21)
df$Soil_Type22 <- as.factor(df$Soil_Type22)
df$Soil_Type23 <- as.factor(df$Soil_Type23)
df$Soil_Type24 <- as.factor(df$Soil_Type24)
df$Soil_Type25 <- as.factor(df$Soil_Type25)
df$Soil_Type26 <- as.factor(df$Soil_Type26)
df$Soil_Type27 <- as.factor(df$Soil_Type27)
df$Soil_Type28 <- as.factor(df$Soil_Type28)
df$Soil_Type29 <- as.factor(df$Soil_Type29)
df$Soil_Type30 <- as.factor(df$Soil_Type30)
df$Soil_Type31 <- as.factor(df$Soil_Type31)
df$Soil_Type32 <- as.factor(df$Soil_Type32)
df$Soil_Type33 <- as.factor(df$Soil_Type33)
df$Soil_Type34 <- as.factor(df$Soil_Type34)
df$Soil_Type35 <- as.factor(df$Soil_Type35)
df$Soil_Type36 <- as.factor(df$Soil_Type36)
df$Soil_Type37 <- as.factor(df$Soil_Type37)
df$Soil_Type38 <- as.factor(df$Soil_Type38)
df$Soil_Type39 <- as.factor(df$Soil_Type39)
df$Soil_Type40 <- as.factor(df$Soil_Type40)
df$Cover_Type = recode_factor(df$Cover_Type, `1` = "Spruce/Fir", `2` = "Lodgepole Pine", `3` = "Ponderosa Pine",
                              `4` = "Cottonwood/Willow", `5` = "Aspen" ,`6`  = "Douglas-fir", `7`  = "Krummholz" )


#PARTITION
N = nrow(df)
training_size = round(N*0.6)
set.seed(1234)
training_cases = sample(N, training_size)
training = df[training_cases,]
test = df[-training_cases,]


### MODEL 1: CLASSIFICATION TREE

stopping_rules <- rpart.control(minsplit = 0, minbucket = 0, cp = -1)
model <- rpart(Cover_Type ~ . , data =training, control = stopping_rules)

#PREDICT
predictions <- predict(model, test, type="class")

##Evaluate
observations <- test$Cover_Type
table(predictions, observations)

error_rate <- sum(predictions!=test$Cover_Type)/nrow(test)
error_bench <- benchmarkErrorRate(training$Cover_Type,test$Cover_Type)

##PRUNING
cart_model <- easyPrune(model)

#PREDICT
cart_predictions <- predict(cart_model, test, type="class")

##Evaluate
cart_observations <- test$Cover_Type
table(cart_predictions, cart_observations)

cart_error_rate_pruned <- sum(cart_predictions !=test$Cover_Type)/nrow(test)
accuracy_rate <- 1-error_rate
accuracy_rate_pruned <- 1-cart_error_rate_pruned

pred_cart_full <- predict(cart_model, df, type="class")

### MODEL 2: KNN

# MANAGE
df_knn = df
df_knn$Wilderness_Area1 <- NULL
df_knn$Wilderness_Area2 <- NULL
df_knn$Wilderness_Area3 <- NULL
df_knn$Wilderness_Area4 <- NULL
df_knn$Soil_Type1 <- NULL
df_knn$Soil_Type2 <- NULL
df_knn$Soil_Type3 <- NULL
df_knn$Soil_Type4 <- NULL
df_knn$Soil_Type5 <- NULL
df_knn$Soil_Type6 <- NULL
df_knn$Soil_Type7 <- NULL
df_knn$Soil_Type8 <- NULL
df_knn$Soil_Type9 <- NULL
df_knn$Soil_Type10 <- NULL
df_knn$Soil_Type11 <- NULL
df_knn$Soil_Type12 <- NULL
df_knn$Soil_Type13 <- NULL
df_knn$Soil_Type14 <- NULL
df_knn$Soil_Type15 <- NULL
df_knn$Soil_Type16 <- NULL
df_knn$Soil_Type17 <- NULL
df_knn$Soil_Type18 <- NULL
df_knn$Soil_Type19 <- NULL
df_knn$Soil_Type20 <- NULL
df_knn$Soil_Type21 <- NULL
df_knn$Soil_Type22 <- NULL
df_knn$Soil_Type23 <- NULL
df_knn$Soil_Type24 <- NULL
df_knn$Soil_Type25 <- NULL
df_knn$Soil_Type26 <- NULL
df_knn$Soil_Type27 <- NULL
df_knn$Soil_Type28 <- NULL
df_knn$Soil_Type29 <- NULL
df_knn$Soil_Type30 <- NULL
df_knn$Soil_Type31 <- NULL
df_knn$Soil_Type32 <- NULL
df_knn$Soil_Type33 <- NULL
df_knn$Soil_Type34 <- NULL
df_knn$Soil_Type35 <- NULL
df_knn$Soil_Type36 <- NULL
df_knn$Soil_Type37 <- NULL
df_knn$Soil_Type38 <- NULL
df_knn$Soil_Type39 <- NULL
df_knn$Soil_Type40 <- NULL

standardizer = preProcess(df_knn ,c("center","scale")) #center to mean 0 scale so SD 1
df_knn  = predict(standardizer, df_knn)

# partition the data
training_KNN = df_knn[training_cases, ]
test_KNN = df_knn[-training_cases, ]

# BUILD
best_k = knnCrossVal(Cover_Type ~., training_KNN) 
model_kbest = knn3(Cover_Type ~ ., data=training_KNN, k=best_k) #use k value of 3 for best stacking

#EVALUATE
predictions_knn = predict(model_kbest,test_KNN, type="class")
table(predictions_knn,observations)
knn_error = sum(predictions_knn != observations)/nrow(test)


### MODEL 3: NAIVE BAYES'

#MANAGE
df_PCA = df
df_Hillshades = df[ ,c("Aspect","Slope","Hillshade_9am","Hillshade_Noon","Hillshade_3pm")]
pca =prcomp(df_Hillshades, center=TRUE, scale=TRUE)
summary(pca) #PC1 52%, PC2 32% PC3 10% PC4 6% PC5 .06% 
pcs = predict(pca, df_Hillshades)
df_PCA = cbind(df_PCA,pcs)
df_PCA$PC3 = NULL
df_PCA$PC4 = NULL
df_PCA$PC5 = NULL
df_PCA$Hillshade_9am = NULL
df_PCA$Hillshade_Noon = NULL
df_PCA$Hillshade_3pm = NULL
df_PCA$Aspect = NULL
df_PCA$Slope = NULL
df_PCA = rename(df_PCA, PC1_hillshade=PC1)
df_PCA = rename(df_PCA, PC2_hillshade=PC2)


df_Hydrology = df_PCA[ ,c("Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology" )]
pca =prcomp(df_Hydrology, center=TRUE, scale=TRUE)
summary(pca) #PC 1 83% PC 2 17%
pcs = predict(pca, df_Hydrology)
df_PCA = cbind(df_PCA,pcs)
df_PCA$Horizontal_Distance_To_Hydrology = NULL
df_PCA$Vertical_Distance_To_Hydrology = NULL
df_PCA = rename(df_PCA, PC1_hydro=PC1)
df_PCA = rename(df_PCA, PC2_hydro=PC2)

#converting PC1 to categorical
df_NB = df_PCA
PC1_hillshade_binned = cut(df_NB$PC1_hillshade,10)
levels(PC1_hillshade_binned) #factor 
df_NB = cbind(df_NB,PC1_hillshade_binned) 
df_NB$PC1_hillshade = NULL
summary(PC1_hillshade_binned)

#converting PC2 to categorical
PC2_hillshade_binned = cut(df_NB$PC2_hillshade,10)
levels(PC2_hillshade_binned) #factor 
df_NB = cbind(df_NB,PC2_hillshade_binned) 
df_NB$PC2_hillshade = NULL
summary(PC2_hillshade_binned)

# converting elevation to factor
Elevation_binned = cut(df_NB$Elevation,c(1800,2000,2400,2640,2920,3160,3320,3440,3900)) 
levels(Elevation_binned) #factor 
df_NB = cbind(df_NB,Elevation_binned) 
df_NB$Elevation = NULL
summary(Elevation_binned)

# converting horizontal distance to fire points
Horizontal_Distance_To_Fire_Points_binned = cut(df_NB$Horizontal_Distance_To_Fire_Points,10) 
levels(Horizontal_Distance_To_Fire_Points_binned) #factor 
df_NB = cbind(df_NB,Horizontal_Distance_To_Fire_Points_binned) 
df_NB$Horizontal_Distance_To_Fire_Points = NULL
summary(Horizontal_Distance_To_Fire_Points_binned)

# converting horizontal distance to roadways
Horizontal_Distance_To_Roadways_binned = cut(df_NB$Horizontal_Distance_To_Roadways,10) 
levels(Horizontal_Distance_To_Roadways_binned) #factor 
df_NB = cbind(df_NB,Horizontal_Distance_To_Roadways_binned) 
df_NB$Horizontal_Distance_To_Roadways = NULL
summary(Horizontal_Distance_To_Roadways_binned)

#converting PC1 Hydro to categorical
PC1_hydro_binned = cut(df_NB$PC1_hydro,10)
levels(PC1_hydro_binned) #factor 
df_NB = cbind(df_NB,PC1_hydro_binned) 
df_NB$PC1_hydro = NULL
summary(PC1_hydro_binned)

#converting PC2 Hydro to categorical
PC2_hydro_binned = cut(df_NB$PC2_hydro,10)
levels(PC2_hydro_binned) #factor 
df_NB = cbind(df_NB,PC2_hydro_binned) 
df_NB$PC2_hydro = NULL
summary(PC2_hydro_binned)
training_NB = df_NB[training_cases, ]
test_NB = df_NB[-training_cases, ]

#BUILD
NB_model = naiveBayes(Cover_Type ~., data=training_NB)

# Predictions
predictions_NB = predict(NB_model, test_NB) 
observations = test$Cover_Type

# Evaluate
error_NB = sum(predictions_NB != observations)/nrow(test)
table(predictions_NB,observations)
NB_error_bench = benchmarkErrorRate(training_NB$Cover_Type, test_NB$Cover_Type)

####Stacking
pred_ctree_full = predict(cart_model,df,type="class") 
pred_KNN_full = predict(model_kbest,df_knn, type="class")
pred_NB_full = predict(NB_model, df_NB)
df_stacked = cbind(df,pred_ctree_full,pred_KNN_full,pred_NB_full) # binding the columns to the main dataframe 

#Using Random Forest as our manager model for our helper models (KNN, Classification Tree & Naive Bayes) 
train_stacked = df_stacked[training_cases,] 
test_stacked = df_stacked[-training_cases,]
stacked_rf = randomForest(Cover_Type ~.,method = "class", data=train_stacked,ntree=500) 
pred_stacked = predict(stacked_rf,test_stacked,type="class")
error_stacked = sum(pred_stacked != observations)/nrow(test_stacked)
table(pred_stacked,observations)




