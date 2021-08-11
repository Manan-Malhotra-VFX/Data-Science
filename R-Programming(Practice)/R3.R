# Join (Merge) data frames (inner, outer, left, right) 
df1 = data.frame(CustomerId = c(1:6), Product = c(rep("Oven", 3), rep("Television", 3)))
df1
df2 = data.frame(CustomerId = c(2, 4, 6,7), State = c(rep("California", 3), rep("Texas", 1)))
df2
# Merging happens based on the common column name in both the data sets
# Inner Join
df<-merge(x=df1,y=df2,by="CustomerId")
df
# Outer Join
df<-merge(x=df1,y=df2,by="CustomerId",all=TRUE)
df

# Left outer join
df<-merge(x=df1,y=df2,by="CustomerId",all.x=TRUE)
df
# Right outer join 
df<-merge(x=df1,y=df2,by="CustomerId",all.y=TRUE)
df


# apply 
# Returns a vector or array or list of values obtained by 
#applying a function to margins of an array or matrix.
x <- matrix(rnorm(30), nrow=5, ncol=6)
x
?apply
apply(x, 1 ,sum)
apply(x, 2 ,sum)


# lapply 
# lapply function takes list, vector or Data frame  as input and returns only list as output
A <- matrix(1:9, 3,3)
A
B <- matrix(4:15, 4,3)
B
C <- matrix(8:10, 3,2)
C
MyList <- list(A,B,C)
MyList
class(MyList)
lapply(MyList, mean)

# sapply 
# sapply function takes list, vector or Data frame  as input. It is similar to lapply function but returns only vector as output
sapply(MyList, mean)

random <- c("This","is","random","vector")
random
lapply(random, nchar)

# mapply
# mapply applies FUN to the first elements of each (.) argument, the second elements, the third elements, and so on.
mapply(sum,c(1,2,3), c(4,5,6), c(7,8,9))
M1 = matrix(c(1,2,3,4))
M2 = matrix(c(5,6,7,8))
mapply(sum,M1,M2)
# tapply
# tapply is used when you want to apply a function to subsets of a vector and the subsets are 
# defined by some other vector, usually a factor
attach(iris)
View(iris)
tapply(Sepal.Length, Species, mean) # mean of Sepal.Length for all 3 Species 
tapply(Sepal.Width, Species, median)


### getting present working directory
getwd()

### changing the present working directory

setwd("path")


data = read.csv('D:\\Assignments Python\\Association rules\\book.csv')

## to read excel sheet
library(readxl)
data = read_xlsx('C:\\Users\\Tarun Reddy\\Downloads\\EastWestAirlines.xlsx' , sheet = 2)


#install.packages("dplyr")
library(dplyr)
data()
data(mtcars)
View(mtcars)
?mtcars
head(mtcars)
str(mtcars)
summary(mtcars)


#1. Filter or subset

#You can use "," or "&" to use and condition
filter(mtcars,cyl==8,gear==5)


filter(mtcars,cyl==8&gear==5)



filter(mtcars,cyl==8 |gear==5)


#select columns
select(mtcars,mpg,cyl,gear)

# Use ":" to select multiple contiguous columns, 
#and use "contains" to match columns by name

select(mtcars,"carb",mpg:disp,"gear")


select(mtcars,mpg:disp,contains("ge"),
       contains("carb"))


#Exclude a particular column 
select(mtcars,c(-gear,-carb))
select(mtcars,-mpg:-disp)


select(mtcars,-contains("ge"))


filter(select(mtcars,gear,carb,cyl),
       cyl==8)


#To select all columns that start with the character string "c", 
#use the function starts_with()
head(select(mtcars, starts_with("c")))


##Some additional options to select columns based on specific criteria:
#ends_with() : Select columns that end with a character string
#contains() : Select columns that contain a character string


#3. Arrange : Reorder rows

#Syntax:
#arrange(dataframe,orderby)
arrange(mtcars,cyl)
arrange(select(mtcars,"cyl","gear"),cyl)
arrange(select(mtcars,"cyl","gear"),desc(cyl))


#mutate: Add new variable
#Base R Approach
temp <- mtcars
temp$new_variable <- temp$hp + temp$wt
View(temp$)
temp$new_variable1 <- 0
str(temp)

temp$new_variable <- NULL
str(temp)

##dplyr Approach
temp <- mutate(temp,mutate_new = temp$hp + temp$wt)
str(temp)
View(temp)
# Fetch the unique values in dataframe

#Base Package approach - unique function
#unique()


#dplyr approach

#distinct() 
distinct(mtcars["cyl"])
distinct(mtcars[c("cyl","gear")])


#NA is a logical constant of length 1, which contains a missing value indicator.
#NULL represents the null object

c(1,2,3,NA,NULL)
sum(10,20,NA)
mean(c(2,2,2,2,NA),na.rm = T)
sum(10,20,NA)
sum(10,20,NA,na.rm = T)

#Table is very handy to find the frequencies (mode)
#Base Package Approach 
table(mtcars$cyl)


# Rowsums ,colsums rowmeans and Columnmeans 
data("iris")
View(iris)
rowSums(iris[,-5])
colSums(iris[,-5])
rowMeans(iris[,-5])
colMeans(iris[,-5])


unique(iris$Species)

# no. of rows
data("iris")
length(iris)
length(iris[,1])
# which() on data frame
data("iris")
View(iris)
which(iris$Petal.Width == 0.2)

Range <- function(x){
  max(x, na.rm = T) - min(x, na.rm = T)
}

Range(iris$Sepal.Length)
Range(iris$Sepal.Width)
Range(iris$Petal.Length)
Range(iris$Petal.Width)

var(iris$Sepal.Length, na.rm = T)
var(iris$Sepal.Width, na.rm = T)
var(iris$Petal.Length, na.rm = T)
var(iris$Petal.Width, na.rm = T)

sd(iris$Sepal.Length, na.rm = T)
sd(iris$Sepal.Width, na.rm = T)
sd(iris$Petal.Length, na.rm = T)
sd(iris$Petal.Width, na.rm = T)

