x=2 # x is a vector instead of variable here
2+2
?mean #help
4+2
4-2
2*2
7/6
8%%3
2^3
abs(-5)
sqrt(2)
abc <- 5  #use alt- 
abc=3
print(abc)
abc
ls() #to print variables that are created
rm(x) #to remove variables one by one that are created
ls()

# use brush on right to clear global environment, environment on right keeps the runtime data within it, if cleared need to re-run all variables(its like kernal)
# ctrl L to clear console 
# ctrl enter to run single line
# ctrl shift enter to run all lines
my_age <- 26
my_name <- "Manan"
is_datascientist <- TRUE
'My friend\'s name is "Manan"'
class(my_age)  #basically telling data type
class(my_name)
class(is_datascientist)
is.numeric(my_age) #to check if variable iss numeric
is.numeric(my_name)
my_age=as.character(my_age) # to convert to character/String
class(my_age)
is.numeric(my_age)

friends_Age <- c(10,20,30) #c is concatenate and we are using vector datatype here but it should contain single typ of datatypes like int ,char/string or logical/boolean
are_married <- c(TRUE,FALSE,TRUE)
names(friends_Age) <- c("manan","naman","ashu") #names is a func here to add names to ages 
friends_Age
friends_name_age <- c(manan=26,ashu=22, #another way
                      raju=47)
friends_name_age
friends_name_age[friends_name_age>23]
length(friends_name_age)

have_child <- c(manam="no",ashu="yes",raj=NA)
have_child
is.na(have_child) # to check for missing values
sum(is.na(have_child))

friends_Age
friends_Age[1]#indexes
friends_Age[c(1,3)]
friends_Age[1:3]
friends_Age["abc"]# to check if abc is present in friends_Age, if not it will return NA
friends_Age["manan"]
friends_Age[-2] # this excludes the - value like in this case 2nd index will be excluded
friends_Age[-c(1,3)]
friends_Age[-(1:3)]

have_child[!is.na(have_child)]
have_child
have_child[!is.na(have_child)] <- "no" #replacing values with no
have_child
have_child[is.na(have_child)] <- "yes" 
have_child

# sqrt(x),max(x),min(x),range(x){range gives max and min value},length(X),sum(x),prod(x)
# mean(X),sd(x),var(x),sort(x)

c1 <- c(1,2,3)
c2 <- c(4,5,6)
c3 <- c(7,8,9)
my_combined_data <- cbind(c1,c2,c3) # to combine data column wise(matrix)
my_combined_data
my_combined_data_r <- rbind(c1,c2,c3) # to combine data row wise(matrix)
my_combined_data_r
my_combined_data_general <- c(c1,c2,c3)
my_combined_data_general

rownames(my_combined_data) <- c("row1","row2","row3")#to add row names or to view them
my_combined_data
rownames(my_combined_data)
colnames(my_combined_data)#to add column names or to view them
my_combined_data
t(my_combined_data) #t is for transpose which inverts rows and columns

?matrix
matrix(data = NA, nrow = 1, ncol = 1, byrow = FALSE,dimnames = NULL)# to create matrix directly
var <- matrix(data = c(1,2,3,11,12,13), nrow = 2,byrow = TRUE, 
              dimnames = list(c("row1","row2"),c("c.1","c.2","c.3")))
var
var <- matrix(data = c(1,2,3,11,12,13), nrow = 2,byrow = FALSE, 
              dimnames = list(c("row1","row2"),c("c.1","c.2","c.3")))
var

#ncol(x),nrows(x)  to calculate the number of rows and columns
#dim(x) dimensions of rows and column, works like.shape command

var[1:2,1:2]
var["row2","c.2"]
var[,-1]
var[1,"c.3"]

my_friends <- c("manan","naman","ashu","raj")
friends_Age <- c(10,20,30,40) 
are_married <- c(TRUE,FALSE,TRUE,TRUE)

friends_data <- data.frame(
              name=my_friends,
              age=friends_Age,
              height=c(180,170,165,181),
              married=are_married
)
friends_data

friends_data$name  #dollar sign works like dot here, it will display values inside that perticular variable like when we press tab in python
friends_data[friends_data$age>=20,]
friends_data[friends_data$age>=20,c(1,2)]
friends_data[friends_data$age>=20,c("name","age")]
age27 <- friends_data$age>=20
cols <- c("name","age")
friends_data[age27,cols]

friends_data$age
age
attach(friends_data) #Attach works like $ just now i don't have to define main variable/vector, i can call age directly now without calling friends_data
age
detach(friends_data)#it will detach it now we have to call the vector(friends_data) before age
age

friends_group <- as.factor(c(1,2,1,1))  # basically grouping data then assigning name values to it
friends_group
levels(friends_group)=c("not_best_friend","best_friends") #assigning name values to it
friends_group
#friends_group <- factor(friends_group,levels = c("not_best_friend","best_friends"))  # another way but gives Na values so lets not use it for now
#friends_group

colors=as.factor(c("red","green","blue","red","green"))
colors
str(colors) #str is structure here, it orders it as 3,2,1 depending on the alphabetical order as b is the first one here for blue
#so thee categorical data has been converted here to numerical data

friends_data$group <- friends_group #to add "friend_group" data in "group" column in "friend_data"
friends_data

#different methods to do the same thing
cbind(friends_data,group=friends_group) #does same thing to add column
levels(friends_group)
levels(friends_group) <- c("not_best_friend","best_friends")
friends_group

# is.factor(x) to check if data is a factor

#create a list
my_family <- list(
    mother="Veronica",
    father="Michel",
    sisters=c("Alice","Monica"),
    sis_age=c(12,16)
)
my_family
names(my_family)
length(my_family)
my_family$father
my_family[["father"]]#double brackets gives just value
my_family["father"]# single as shows vector name
my_family[[1]]
my_family[[3]][1]
my_family[3][1] #won't work correctly

my_family$grand_father <- "Stoner" #to add values
my_family$grand_mother <- "Smily"
my_family

list_abc <- c(list_1,list_2) # to concatenate,wont workas no vectors defined
paste("One",1,2,"Two") #it concatinates the different types of values into char
paste("x",1:5,sep=" ") 
paste(c("one","two","three"),collapse = " and ")
paste(c("x","y"),1:5,sep = "_", collapse = " and ")

paste0("One",1,2,"Two")

v1 <- c(1,2,5,4,3)
sort(v1)
sort(v1,decreasing = TRUE)

v2 <- c("abc","def","ijk")
sort(v2)
sort(v2,decreasing = TRUE)

mylist=list(c(1,4,6),"dog",3,TRUE)
mylist
class(mylist)
unlist(mylist) #divides data and converts it to char
class(unlist(mylist))

week=c("sunday","monday","tuesday","wed","thus","fri","sat","monday","fri")
week
table(week) #will give counts like nunique and value.counts
prop.table(table(week))*100 #gives proportion/probability

week_ordered=factor(week,levels = c("sunday","monday","tuesday","wed","thus","fri","sat"),ordered = TRUE)
week_ordered
table(week_ordered)
prop.table(table(week_ordered))
sort(prop.table(table(week_ordered)))
sort(prop.table(table(week_ordered)),decreasing=TRUE)

?gl
v <- gl(3,4,labels = c("India","USA","Russia")) # to generate factor levels
v

#apply
x <- matrix(rnorm(30),nrow = 5,ncol = 6)
x
apply(x, 1, sum) # row wise sum will be done
apply(x, 2, sum) # column wise sum will be done
apply(x, 1:2, function(x) x/2) # column/row will be divided by 2

#lapply
a <- matrix(1:9, 3,3)
b <- matrix(4:15, 4,3)
c <- matrix(8:10, 3,2)
a
b
c
mylist <- list(a,b,c)
mylist
class(mylist)
lapply(mylist, mean)

#sapply
sapply(mylist, mean)

?apply
?lapply
?sapply

random <- c("This","is","Sparta")
random
sapply(random,nchar) #count number of char

#mapply
mapply(sum,c(1,2,3),c(2,3,4),c(4,5,6))

#tapply
attach(iris) #pre defined sample dataset
iris
tapply(iris$Sepal.Length, Species,mean)
tapply(iris$Sepal.Width, Species,median)

#cat
?cat
cat("one",2,"three",4,"five")
cat(1:10,sep = "\t")
cat(1:10,sep = "\n")

getwd() # to see working directory
setwd("C:/Users/manan/Documents") # to set working directory, if gives error replace forward slashes with backward

set.seed(123) # to reproduce same results for random values, need to set it for each execution i.e run this before generation random numbers always
rnorm(5)

#if statement
x <- 5
if(x > 0){
  print("Non-negative number")
} else {
  print("Negative number")
}

#nested if
x <- 0
if (x < 0) {
  print("Negative number")
} else if (x > 0) {
  print("Positive number")
} else print("Zero")

#another method
a = c(5,7,2,9)
ifelse(a %% 2 == 0,"even","odd")

# for more than 2 conditions
client <- c("private", "public", "other" , "private")
VAT <- ifelse(client =='private', 1.12, ifelse(client == 'public', 1.06, 1))
VAT


# While loop
i <- 1
while (i < 6) {
  print(i)
  i = i+1
}

# Repeat loop
v <- c("Hello","loop")
cnt <- 2

repeat {
  print(v)
  cnt <- cnt+1
  
  if(cnt > 5) {
    break
  }
}

# For Loop 
v <- LETTERS[1:4]
for ( i in v) {
  print(i)
}

# Function 
#function_name <- function(arg_1, arg_2, ...) {
 # Function body 
#}
# User-defined Function
new.function <- function(a) {
  for(i in 1:a) {
    b <- i^2 #^ for square
    print(b)
  }
}
new.function(6)
