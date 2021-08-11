# Decision Making/ conditional Statements in R
# If Statement
# Syntax: 
if (test_expression) {
  statement
  }

x <- -5

if(x > 0){
  print("Positive number")
}

# If Else Statement
# syntax: 
if (test_expression) {
  statement1
} else {
  statement2
  }

x <- 5
if(x > 0){
  print("Non-negative number")
} else {
  print("Negative number")
}

# Nested If Else Statement
if ( test_expression1) {
  statement1
} else if ( test_expression2) {
    statement2
  } else if ( test_expression3) {
  statement3
  } else  statement4

x <- 0
if (x < 0) {
  print("Negative number")
} else if (x > 0) {
  print("Positive number")
} else print("Zero")

# There is an easier way to use if...else statement specifically for vectors in R programming
# Syntax: ifelse(test_expression,x,y)
a = c(5,7,2,9)
ifelse(a %% 2 == 0,"even","odd")

# for more than 2 conditions
client <- c("private", "public", "other" , "private")
VAT <- ifelse(client =='private', 1.12, ifelse(client == 'public', 1.06, 1))
VAT

# Loops in R
# While loop
# Syntax
while (test_expression) 
{
  statement
}

i <- 1
while (i < 6) {
  print(i)
  i = i+1
}

# Repeat loop
# Syntax 
repeat { 
  commands 
  if(condition) {
    break
  }
}

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
# Syntax
for (value in vector) {
  statements
}

v <- LETTERS[1:4]
for ( i in v) {
  print(i)
}


# Loop Control Statements
# break Statement - terminates the loop statement and transfers execution to the statement 
# immediately following the loop
v <- c("Hello","loop")
cnt <- 2

repeat {
  print(v)
  cnt <- cnt + 1
  
  if(cnt > 5) {
    break
  }
}

# next Statement - useful when we want to skip the current iteration of a loop without terminating it
v <- LETTERS[1:6]
for ( i in v) {
  
  if (i == "D") {
    next
  }
  print(i)
}

# Function 
function_name <- function(arg_1, arg_2, ...) {
  Function body 
}
# User-defined Function
new.function <- function(a) {
  for(i in 1:a) {
    b <- i^2
    print(b)
  }
}

# Calling a Function
new.function <- function(a) {
  for(i in 1:a) {
    b <- i^2
    print(b)
  }
}
new.function(6)

# Calling a Function without an Argument
new.function <- function() {
  for(i in 1:5) {
    print(i^2)
  }
}	
new.function()

# Calling a Function with Default Argument
# Create a function with arguments.
new.function <- function(a = 3, b = 6) {
  result <- a * b
  print(result)
}
# Call the function without giving any argument.
new.function()
# Call the function with giving new values of the argument.
new.function(9,5)

# Lazy Evaluation of Function
# Create a function with arguments.
new.function <- function(a, b) {
  print(a^2)
  print(a)
  print(b)
}
# Evaluate the function without supplying one of the arguments.
new.function(6)


