# simple calculator
# addition
def add(num1 , num2) :
    addition =  num1 + num2
    print(f"The sum of {num1} and {num2} is {addition}") #formatted strings

#subtraction
def sub(num1 , num2) :
    subtraction =  num1 - num2
    print(f"The result of subtracting {num2} from {num1} is {subtraction}")

#multiplication
def mult(num1 , num2) :
    multiplication =  num1 * num2
    print(f"The mult of {num1} and {num2} is {multiplication}") 

#division
def div(num1 , num2) :
    division =  num1 / num2
    print(f"The div of {num1} and {num2} is {division}") 

#modulus
def mod(num1 , num2) :
    modulus = num1 % num2 
    print(f"The mod of {num1} and {num2} is {modulus}") 



number1,operator,number2 = map(str, input("Enter your equation: ").split()) 
number1 = int(number1)
number2 = int(number2)


if operator ==  '+' :
    add(number1,number2)

elif operator ==  '-' :
    sub(number1,number2)

elif operator ==  '*' :
    mult(number1,number2)

elif operator ==  '/' :
    div(number1,number2)

elif operator ==  '%' :
    mod(number1,number2)

else:
    print("Error!! equation not right. try equation - 5 + 5")