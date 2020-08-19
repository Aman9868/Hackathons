#READING
test <- read.csv("test.csv",stringsAsFactors = F)
train <- read.csv("train.csv",stringsAsFactors = F)
#PACKAGES
require(dplyr)
require(ggplot2) 
require(stringr)
require(plyr)
require(h2o)
#COMBINE THE DATASET
master <- bind_rows(train,test)

#DUPLICACY 
sum(duplicated(master[,-2]))#PID
110896/nrow(master)*100#14.15
sum(duplicated(master[,-1]))# ID
15967/nrow(master)*100#2.03

master <- master[!duplicated(master), ]

#NA
colSums(master=="",na.rm = T)#No BLANKS


#FIRST CORRECTING THE DATA SET AS DATA SET CONTAIN MOSTLY NO TYPE DATA SO CONVERTING CHARCTER INTO NUMERIC AND THEN MAKING FACTOR
#
table(master$Age)
#TAKING AVERAGE OF 5 CLASS INTERVAL BINING TO PREDICT NEXT ONE
master$Age <- as.factor(mapvalues(master$Age,from = c("55+"),to=c("55-63")))
?ddply
 require(plyr)
 master <- ddply(master, .(Age),function(x){
  level=x$Age[1]
  min_max=as.numeric(str_split(as.character(level), '-')[[1]])
  x$Age=runif(nrow(x),min = min_max[1],max=min_max[2])
  return(x)
})
 mc<- str_split(as.character(master$Age),pattern = "\\.",simplify = T)
master$Age <- as.numeric(str_split(mc[,1],pattern = "[.]",simplify = T))
 
#Gender
table(master$Gender)
master$Gender <- as.numeric(mapvalues(master$Gender,from = c("F","M"),to=c("0","1")))

#CITY CATEGORY
table(master$City_Category)

master$City_Category <- as.factor(master$City_Category)


#STAY IN CURRENT 
table(master$Stay_In_Current_City_Years)
#makin intervals by taking bining average
master$Stay_In_Current_City_Years <- as.factor(mapvalues(master$Stay_In_Current_City_Years,from=c("4+"),to=c("4")))
master$Stay_In_Current_City_Years <- as.numeric(master$Stay_In_Current_City_Years)

#MARITAL
table(master$Marital_Status)

#OCCUPATION
table(master$Occupation)
master$Occupation <- as.numeric(master$Occupation)
#NA
round(colSums(is.na(master))/nrow(master) * 100)

master$Product_Category_2 <- NULL
master$Product_Category_3 <- NULL


#PRODUCT ID
length(unique(master$Product_ID))

#COMBINE THE PID AND AND CUSTOM ID
master$Combine_ID <- paste(master$User_ID,master$Product_ID,sep = "")
master$User_ID <- NULL
master$Product_ID <- NULL
                  
master<- master%>%
  dplyr::select(Combine_ID,everything())

                  ##******EDA*******###



## UNIVARIATE/BIVARIATE ANALYSIS
#1.Combine ID
n_distinct(master$Combine_ID) == nrow(master)
 master%>%
  dplyr::group_by(Combine_ID)%>%
  dplyr::summarise(total_p=sum(Purchase,na.rm = T))%>%
  dplyr::arrange(desc(total_p))


#3.GENDER
master %>%
  dplyr::group_by(Gender) %>%
  dplyr::summarise(total_p=sum(Purchase,na.rm = T)) %>%
  dplyr::arrange(desc(total_p))

ggplot(master[!is.na(master$Purchase),],aes(x=Purchase,fill=Gender))+geom_histogram(binwidth =3000,color="black")+facet_wrap(~Gender)

#4.AGE
hist(master$Age,col="blue")
master%>%
  dplyr::group_by(Age)%>%
  dplyr::summarise(total_p=sum(Purchase,na.rm = T))%>%
  dplyr::arrange(desc(total_p))
  
   #outlier
plot(quantile(master$Age,seq(0,1,by=0.01)))
quantile(master$Age,seq(0,1,by=0.01))
master$Age[which(master$Age<12)] <- 12

#5.OCCUPATION
hist(master$Occupation,col="red")
master%>%
  dplyr::group_by(Occupation)%>%
  dplyr::summarise(total_p=sum(Purchase,na.rm = T))%>%
  dplyr::arrange(desc(total_p))
   #Outlier
 plot(quantile(master$Occupation,seq(0,1,by=0.01)))#no


#6.CITY CATEGORY
master%>%
  dplyr::group_by(City_Category)%>%
  dplyr::summarise(total_p=sum(Purchase,na.rm = T))%>%
  dplyr::arrange(desc(total_p))
ggplot(master[!is.na(master$Purchase),],aes(x=Purchase,fill=as.factor(City_Category)))+geom_histogram(bins=10,position = "dodge")+facet_wrap(~City_Category)


#7.STAY IN CURRENT YEARS
master%>%
  dplyr::group_by(Stay_In_Current_City_Years)%>%
  dplyr::summarise(total_p=sum(Purchase,na.rm = T))%>%
  dplyr::arrange(desc(total_p))
ggplot(master[!is.na(master$Purchase),],aes(x=Purchase,fill=as.factor(Stay_In_Current_City_Years)))+geom_histogram(bins=20,position = "dodge")+facet_wrap(~Stay_In_Current_City_Years)


#8.MARITAL STATUS
master%>%
  dplyr::group_by(Marital_Status)%>%
  dplyr::summarise(total_p=sum(Purchase,na.rm = T))%>%
  dplyr::arrange(desc(total_p))
ggplot(master[!is.na(master$Purchase),],aes(x=Purchase,fill=as.factor(Marital_Status)))+geom_histogram(bins=10,position = "dodge",col="black")+facet_wrap(~Marital_Status)

#9.Product category 1
master%>%
  dplyr::group_by(Product_Category_1)%>%
  dplyr::summarise(total_p=sum(Purchase,na.rm = T))%>%
  dplyr::arrange(desc(total_p))
#Outlier
plot(quantile(master$Product_Category_1,seq(0,1,by=0.01)))
quantile(master$Product_Category_1,seq(0,1,by=0.01))
master$Product_Category_1[which(master$Product_Category_1>8)] <- 8
boxplot(master$Product_Category_1)

###FEATURE ENGINEERING
master$Age_Category <-as.factor(ifelse(master$Age<=12,"KIDS",ifelse(master$Age<=24," YOUNG-ADULTS",ifelse(master$Age<=36,"ADULTS",ifelse(master$Age<=50,"MID-ADULTS","SIN-CITIZEN")))))


 ####SPLITTING STAGE####
colnames(master)
require(dummies)
master <- dummy.data.frame(master,names = c("Age_Category","City_Category"),sep="_")

trn <- master[1:nrow(train), ]
tst <- master[-(1:nrow(train)), ]

#LAUNCH H20
localh20 <- h2o.init(nthreads = -1)
h2o.init()

#DATA TRAIN AND TEST

trn.h20 <- as.h2o(trn)
tst.h20 <- as.h2o(tst)
colnames(trn.h20)

#DEPENDENT-PURCHASE
y.dep <- 11
#INDEPENDENT-ALL EXCEPT PURCHASE
x.indep <- c(2:10,12:16)

##MULTIPLE REG IN H20
regmod <- h2o.glm(y=y.dep,x=x.indep,training_frame = trn.h20,family = "gaussian")
h2o.performance(regmod)
h2o.varimp(regmod)

#####Random Forest#####
  rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame = trn.h20, ntrees = 1000, mtries = 3, max_depth = 4, seed = 1122)
h2o.performance(rforest.model)

#check variable importance
h2o.varimp(rforest.model)


####SVM####
svm.model <- h2o.psvm(y=y.dep, x=x.indep, training_frame = trn.h20,kernel_type ="linear")



#####GBM######
system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = trn.h20, ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1122)
)
h2o.performance(gbm.model)


####Deep learning models###
system.time(
  dlearning.model <- h2o.deeplearning(y = y.dep,
                                      x = x.indep,
                                      training_frame = trn.h20,
                                      epoch = 60,
                                      hidden = c(100,100),
                                      activation = "Rectifier",
                                      seed = 1122 ))
h2o.performance(dlearning.model)
