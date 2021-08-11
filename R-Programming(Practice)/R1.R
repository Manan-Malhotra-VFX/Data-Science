#The Comprehensive R Archive Network (CRAN) is the main repository for R packages.
# R is a an open source programming language & software environment
# for statistical analysis, graphical representation,data analysis & visualization 
# R is a simple and easy to learn,read & write
# Developed by Ross Ihaka & Robert Gentleman at the University of 
# Auckland, Newzealand


# R is a case sensitive 

## Define a value in R
1
2+2 # Defining a value
x<-2+2  # storing a value 
x
a # can't define a character variable like this
"a" # A character variable must be define in the qutations
a<-'a' 
a

# Variables or Objects in R

# valid & invalid variable types
# Valid
# only takes character,number, _ , .
var_name2. <- 1 # Has letters, numbers, dot and underscore
.var_name <- 2 # Can start with a dot(.) but the dot(.)should not be followed by a number
var.name <- 3
# Invalid 
var_name% <- 4 # Only dot(.) and underscore allowed 
2var_name <- 5 # cannot start with a number
.2var_name <- 6 # dot followed by a number makes it invalid
_var_name <- 7 # cannot start with _


'''
Data types :  1. Numeric
              2. Integer
              3. Charector
              4. Logical
              5. complex
           

'''
# Defining a Numeric data type 
# All the numbers will come under this data type
a<-1
class(a)

a<-134
a
class(a)

# Defining an integer (takes only Integer value)
a<-1L
a
class(a)
x<-1.2L
x
class(x) # overwrites the type of class
s<-1:10
s
s<-seq(1,10)
s
s<-seq(1,10,2)
s
#defining a complex number
1+2i
x<-1+2i
class(x)
Re(x) # print real value in x
Im(x) # print Imaginary Numbers

# defining a charector
a<-'a'
class(a)
LETTERS
x<-LETTERS[1:6] #sequesnce of letters in caps
x
y<-letters[1:6] # sequesnce of letters in smalls
y

# Defining Logical value
a<-TRUE #T
b<- FALSE #F
class(a)
class(b)

# Logical --> Integer --> Numeric --> complex --> Char
0
x<-as.logical(0)
class(x)
'''
Data structure :  1.Vector - 1dimention,Homogenious
                  2.Matrix - 2dim,Homogenious
                  3.Array - multi dim ,Homogenious
                  4.List - Multi dimention , Hetrogenious
                  5.DataFrame - 2 dimention , hetrogenious
                  6.Factors - data structure to store multiple charectors
'''
## Defining vector
## vector is a combination of values c()
a<-c(1,8,2,6)
a
class(a)
is.vector(a)

a<-c(1,2,3,4,'a')
a
class(a)
is.vector(a)
a[1]
a[c(1,5)]

a[1,5]
a[1:3]
## Defining a matrix
## matrix is nothing but a combination of rows and combinations
a<-matrix(c(1,2,3,4),nrow = 1)
a
a<-matrix(c(1,2,3,4,5,6),ncol = 2)
a
# data always stores in a clumn format
#if you want to allot thge values in a rowwise
x<-matrix(c(1:5),nrow=5,ncol =5,byrow = T,dimnames = list(c(),
                                                          c('c1','c2','c3','c4','c5')))
x
?matrix
y<-matrix(c(1:4),nrow=3,ncol = 3)
y

# Matrix[rows,columns]
x[1,]
x[,1]
x[2,1]
x[c(1,5),5]
rownames(x)<-c('row1','row2','row3','row4','row5')
colnames(x)<-c('col1','col2','col3','col4','col5')
x
# Define array 
#multi dimentional (x,y,z,....)
arr<-array(c(1:6),dim = c(2,4,3))
arr
arr[,,1]
arr[1,,1]
arr[,c(1,3),3]
arr[1,2,3]

# Define a List 
List<-list(Id=c(1,2,3,4),name=c('D','k','L','S'),marks=c(9,9.5,8,8.5))
List
List2<-list(List,x,arr)
List2
List2[[2]]
class(List2)

## Data Frame
DF<-data.frame(Id=c(1,2,3,4),name=c('D','k','L','S'),marks=c(9,9.5,8,8.5))
DF
DF2<-data.frame(Id=c(5,6,7,8),name=c('A','T','V','p'),marks=c(7,9.5,8.5,7.5))
DF2
DF[,1]
DF$name
DF$Id[1]
rbind(DF,DF2)
cbind(DF,DF2)

## Factor
'''
Factor - Factors are the data objects which are used to 
categorize the data and store it as levels. They can store 
both strings and integers. They are useful in the columns which have a
limited number of unique values. Like "Male, "Female" and True, False etc.
They are useful in data analysis for statistical modeling.
''' 
k <- c("red", "green", "blue", "blue", "red", "red")
class(k)
k
factor_k <- factor(k)
class(factor_k)
factor_k

l <- c(1, 2, 3, 4, 4, 5, 3, 2, 5.1)
l
class(l)
factor_l <- factor(l)
class(factor_l)
factor_l

# Finding Variables in Environment
ls()
ls(pattern = "var") # displays variable with var text in it
ls(all.name = TRUE) # also displays hidden(starting with .) variables

# Deleting variables 
var1 <- 1
ls(pattern = "var")
rm(list = ls(pattern = "var")) # will remove all variables with var text in it

#Operators In R
'''
operators : 1.Arthmetic - +,-,*,/,^,%%,%/%
            2.Relational - >,<,>=,<=,==,!=
            3.Logical - !,&,&&,|,||
            4.Assignmet - a. Leftwards assignment - <-, <<-, =
                          b. Rightwards assignment - ->, ->>
'''
# Defining Arthmetic Opertor

n <- c(1, 2.4, 5)
o <- c(3, 4.7, 9.1)
p <- c(1.9, 2, 7)
# Addition (+)
n+o
n+o+p
# sub (-)
n-o
n-o-p
# Multiplication (*)
x<-5
y<-10
x*y
# Division (/) # Quotient
x<-10
y<-3
x/y
# Integer division
x<-10
y<-3
x %/% y
# Modulus (%%) # Remainder
x<-10
y<-3
x%%y

# Relational Operators - returns boolean values
q <- c(1, 3, 4, 3, 2)
r <- c(2, 4, 4, 5, 2)
# 1. > greater than 
q[1] > r[2]

# 2. < lesser than
q < r

# 3. == equal to
q == r

# 4. >= greater than or equal to
q >= r

# 5. <= lesser than or equal to 
q <= r

# 6. != not equal to 
q != r

# Logical operator
# Condition: All numbers greater than or equals to  1 are considered as logical value TRUE
x <- c(TRUE,FALSE,0,6)
y <- c(FALSE,TRUE,FALSE,TRUE)

!x

x&y

x&&y

x|y

x||y


# Assignment Operators
# 1. Left Assignment <- or == or <<-
v1 <- c(3,1,TRUE,2+3i)
v2 <<- c(3,1,TRUE,2+3i)
v3 = c(3,1,FALSE,2+3i)
v1
v2
v3
# 2. Right Assignment -> or ->>
c(3,1,TRUE,2+3i) -> v4
c(3,1,TRUE,2+3i) ->> v5
v4
v5
# 2. %in% - an element belongs to a vector
v7 <- c(8, 9, 10)
v8 <- c(12, 13, 14)
t <- 1:10
t
v7 %in% t
v8 %in% t
t %in% v7
# 3. %*% - multiply a matrix with its transpose
v9 <- matrix( c(2, 6, 5, 1, 10, 4), nrow = 2, ncol = 3, byrow = TRUE)
v9
t(v9)
v9t <- v9 %*% t(v9)
v9t

# End of Operators# 2. %in% - an element belongs to a vector
v7 <- c(8, 9, 10)
v8 <- c(12, 13, 14)
t <- 1:10
v7 %in% t
t %in% v7
v8 %in% t
# 3. %*% - multiply a matrix with its transpose
v9 <- matrix( c(2, 6, 5, 1, 10, 4), nrow = 2, ncol = 3, byrow = TRUE)
v9
t(v9)
v9t <- v9 %*% t(v9)
v9t

# End of Operators
