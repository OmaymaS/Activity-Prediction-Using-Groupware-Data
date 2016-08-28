# Activity Prediction Using Groupware Data




# Introduction

One of the interestng things is to gather data about individuals' excercise and analyze the manner in which they did the exercise. Using modern gadgets such as Jawbone Up, Nike FuelBand, and Fitbit facilitates collecting large amounts of data about personal activity. In this report we will data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The goal to predict the manner in which they did the exercise. 

# Dataset

The data for this project is provided by [Groupware]( http://groupware.les.inf.puc-rio.br/har). Training set is available [Here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and Test set is available [Here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).
The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways, while placing accelerometers on the belt, forearm, arm, and dumbell

# Data Exploration And Cleaning

## Loading Libraries


```r
#load libraries
library(caret)
library(randomForest)
library(rpart)
library(dplyr)
library(rattle)
```
## Exploring Data


```r
traind<-read.csv("./data/pml-training.csv",
              stringsAsFactors = FALSE,
              header = TRUE,
              na.strings = c("","NA"))

testd<-read.csv("./data/pml-testing.csv",
                stringsAsFactors = FALSE,
                header = TRUE,
                na.strings = c("","NA"))

dim(traind)
```

```
## [1] 19622   160
```
We can see that we have 160 column, including the **classe** variable which we want to predict in the test set.

## Cleaning Data

Here we will:
- remove any columns that include more than 80% Nas.


```r
#check percentage of NAs
check_na<-sapply(1:dim(traind)[2], function(x) mean(is.na(traind[,x])))

#get columns with less than 20% Na
traind<- traind[,which(check_na<0.2)]
testd<- testd[,which(check_na<0.2)]
```

- remove columns that include ids and data that do not come from measurements, and won't help in our predictions.

```r
#remove the first 7 columns as they seem to have no effect (id, name..etc.)
traind<-traind[,8:dim(traind)[2]]
testd<-testd[,8:dim(testd)[2]]
```

- convert the **classe** variable into factor as it is our output variable.

```r
traind$classe<-as.factor(traind$classe)
```

# Prediction

## Data Partitioning 
 
Here we'll devide the training set into 2 parts 60%, 40%


```r
##divide training set into 2 parts 60%, 40%
inTrain<-createDataPartition(y=traind$classe,p=0.6,list = F)

training <- traind[inTrain,]
testing <- traind[-inTrain,]
```

## Decision Tree
Here we will start with a basic classification tree


```r
model<-train(classe ~ .,data = training, method = "rpart")

fancyRpartPlot(model$finalModel, sub="")
```

![](Activity_Predcition_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

If we check confusion matrix, we can see that the result is unsatisfactory, the accuracy is low. So we'll try a random forest.

```r
confusionMatrix(testing$classe, predict(model, testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2028   32  164    0    8
##          B  667  495  356    0    0
##          C  645   32  691    0    0
##          D  582  229  475    0    0
##          E  195  193  368    0  686
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4971          
##                  95% CI : (0.4859, 0.5082)
##     No Information Rate : 0.5247          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3423          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4926  0.50459  0.33642       NA  0.98847
## Specificity            0.9453  0.85098  0.88311   0.8361  0.89430
## Pos Pred Value         0.9086  0.32609  0.50512       NA  0.47573
## Neg Pred Value         0.6279  0.92320  0.78960       NA  0.99875
## Prevalence             0.5247  0.12503  0.26179   0.0000  0.08845
## Detection Rate         0.2585  0.06309  0.08807   0.0000  0.08743
## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7189  0.67779  0.60977       NA  0.94138
```

## Random Forest
Here we will fit a random forest and check the accuracy.


```r
model2<- randomForest(classe ~ ., data = training)

predict_class<-predict(model2, testing, type = "class")
```

Let's look at the confusion matrix


```r
confusionMatrix(testing$classe,predict_class)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2228    2    1    1    0
##          B   10 1502    6    0    0
##          C    0    5 1356    7    0
##          D    0    0    9 1277    0
##          E    0    0    2    3 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9922, 0.9957)
##     No Information Rate : 0.2852          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9926          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9955   0.9954   0.9869   0.9915   1.0000
## Specificity            0.9993   0.9975   0.9981   0.9986   0.9992
## Pos Pred Value         0.9982   0.9895   0.9912   0.9930   0.9965
## Neg Pred Value         0.9982   0.9989   0.9972   0.9983   1.0000
## Prevalence             0.2852   0.1923   0.1751   0.1642   0.1832
## Detection Rate         0.2840   0.1914   0.1728   0.1628   0.1832
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9974   0.9964   0.9925   0.9950   0.9996
```

It seems that the random forest gave a good level of accuracy that we can rely on. So we'll use our test set to predict **classe**.


```r
p<-predict(model2, testd, type = "class")
```

And here is the predicition for the  20 test cases:

```
##    p
## 1  B
## 2  A
## 3  B
## 4  A
## 5  A
## 6  E
## 7  D
## 8  B
## 9  A
## 10 A
## 11 B
## 12 C
## 13 B
## 14 A
## 15 E
## 16 E
## 17 A
## 18 B
## 19 B
## 20 B
```
