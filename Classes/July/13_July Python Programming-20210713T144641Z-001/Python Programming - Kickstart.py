{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Chapter 1 - Hello World"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1[a] Hello World"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Welcome Dhanashreee!!!\n"
     ]
    }
   ],
   "source": [
    "print('Welcome Dhanashreee!!!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Welcome Sai\n"
     ]
    }
   ],
   "source": [
    "print(\"Welcome Sai\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mentoring makes learner's mind push beyonf the limits\n"
     ]
    }
   ],
   "source": [
    "print(\"Mentoring makes learner's mind push beyonf the limits\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1[b] Create a variable "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "numbers = [\"Aditi\",\"Ashwathy\",\"Sneha\",123,78]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Aditi', 'Ashwathy', 'Sneha', 123, 78]\n"
     ]
    }
   ],
   "source": [
    "print(numbers)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "34\n"
     ]
    }
   ],
   "source": [
    "#Dont make a variable name with a keyword\n",
    "len_ = 34"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### -----------------------------------------------------------------------------------------------------------------------------------------------------------"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Chapter 2 - Strings and String Methods"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2[a] What is a String?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Manan'"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Anything whihc is enclosed within quotes.\n",
    "learner_names = \"Manan\"\n",
    "learner_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "str"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(learner_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "394\n",
      "<class 'int'>\n"
     ]
    }
   ],
   "source": [
    "target_score = 394\n",
    "print(target_score)\n",
    "print(type(target_score))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "float"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "batting_average = 139.34\n",
    "type(batting_average)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2[b] Concatenation - Indexing and Slicing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Aswathy Palathingalis my new learner DS_11am_batch.\n"
     ]
    }
   ],
   "source": [
    "fname = \"Aswathy\"\n",
    "lname = \"Palathingal\"\n",
    "batch = \"DS_11am_batch\"\n",
    "print(fname + \" \" + lname + \"is my new learner\" + \" \" +batch + \".\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "34"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "name= \"Naresh Chandra sdhskjdkfs sjdhakja\"\n",
    "len(name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Chandra'"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Index number in python always starts from 0.\n",
    "name[7:14]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2[c] Manipulate Strings with methods"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Methods/Function - both are same. Methods are something which is ending with ()."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "learner_1 = \"Darshan\"\n",
    "learner_2 = \"Chandramohan\"\n",
    "learner_3 = \"wasim\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Darshan\n",
      "Chandramohan\n",
      "wasim\n"
     ]
    }
   ],
   "source": [
    "print(learner_1)\n",
    "print(learner_2)\n",
    "print(learner_3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DARSHAN\n",
      "CHANDRAMOHAN\n",
      "WASIM\n"
     ]
    }
   ],
   "source": [
    "print(learner_1.upper())\n",
    "print(learner_2.upper())\n",
    "print(learner_3.upper())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "learner_1 = \"darshan\"\n",
    "learner_2 = \"chandramohan\"\n",
    "learner_3 = \"wasim\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Darshan\n",
      "Chandramohan\n",
      "Wasim\n"
     ]
    }
   ],
   "source": [
    "print(learner_1.capitalize())\n",
    "print(learner_2.capitalize())\n",
    "print(learner_3.capitalize())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2[d] - Interact with user input"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Username: Manoj Yadav\n",
      "11\n"
     ]
    }
   ],
   "source": [
    "name = input(\"Username: \")\n",
    "len(name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter you name: Akash\n",
      "Akash\n",
      "akash\n",
      "akash\n"
     ]
    }
   ],
   "source": [
    "#Get name from the user and make all the alphabets in lower letter\n",
    "name =  input(\"Enter you name: \")\n",
    "print(name)\n",
    "print(name.lower())\n",
    "print(name.casefold())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2[e] Get and print the first letter of the username"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter the username: pavan\n",
      "Hint: First letter:  p\n"
     ]
    }
   ],
   "source": [
    "username = input(\"Enter the username: \")\n",
    "#username[0]\n",
    "print(\"Hint: First letter: \",username[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2[f] Working with Strings and Numbers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter a number: 7.8\n",
      "Enter another number11.6677\n",
      "The product of 7.8 and 11.6677 is 91.00806.\n"
     ]
    }
   ],
   "source": [
    "#Product of 2 number. User preference.\n",
    "num_1 = input(\"Enter a number: \")\n",
    "num_2 = input(\"Enter another number\")\n",
    "product = float(num_1) * float(num_2)\n",
    "print(\"The product of \"+ num_1 + \" and \" + num_2 + \" is \" + str(product) +\".\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "float"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(product)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2[g] Streamline your print Statements"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DhanaShree is from my DataScience 11 am batch.\n"
     ]
    }
   ],
   "source": [
    "name = \"DhanaShree\" #str\n",
    "batch = 11 # int\n",
    "print(\"{} is from my DataScience {} am batch.\".format(name,batch))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Name: Bhpendra\n",
      "Batch Time: 8.30\n",
      "Bhpendra is from my DataScience 8.30 am batch.\n"
     ]
    }
   ],
   "source": [
    "name = input(\"Name: \")\n",
    "batch = input(\"Batch Time: \")\n",
    "print(\"{} is from my DataScience {} am batch.\".format(name,batch))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2[h] Find-a-string-in-a-string"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "comment = \"I dont know why EXCELR is paying money to John. Worst Trainer\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "48"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "comment.find(\"Worst\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'I dont know why EXCELR is paying money to John. Somewhat OK Trainer'"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "comment.replace(\"Worst\",\"Somewhat OK\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'You have won $1000 lottery. Congradulations!!!!!'"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mail = \"You have won $1000 lottery. Congradulations!!!!!\" \n",
    "mail"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'You have won 1000 lottery. Congradulations!!!!!'"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mail.replace(\"$\",\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Study about Regular expressions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Chapter 3 - Numbers in Python"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3[a] Integers and floating point numbers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "team_goal = 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "int"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(team_goal)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "float"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "avg_marks = -8.56\n",
    "type(avg_marks)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "8.56\n"
     ]
    }
   ],
   "source": [
    "print(abs(avg_marks))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Chapter 4 - Functions and Loops"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3 Types of Function:\n",
    "* Predefined Functions  - is something which gets installed while you instal jupyter notebook.\n",
    "* Userdefined Functions - sequence of methods which repeated sequencially, ie, step by step - one after the another.\n",
    "* Anonymous Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "def chicken_shop():\n",
    "    print(\"250g of chicken\")\n",
    "    print(\"Ginger garlic Paste\")\n",
    "    print(\"4eggs\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "250g of chicken\n",
      "Ginger garlic Paste\n",
      "4eggs\n"
     ]
    }
   ],
   "source": [
    "chicken_shop()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Positional Argument"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "def add_numbers(num_1,num_2): #Arguments\n",
    "    print(\"Product of 2 numbers:\",num_1 * num_2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Product of 2 numbers: 63\n"
     ]
    }
   ],
   "source": [
    "add_numbers(7,9) # Parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Product of 2 numbers: 30.24\n"
     ]
    }
   ],
   "source": [
    "add_numbers(5.4,5.6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Product of 2 numbers: 30\n"
     ]
    }
   ],
   "source": [
    "add_numbers(5,6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Get 3 different datatypes and create your own function which returns some meaningful statement."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "def user_details(age,weight,name):\n",
    "    return \"You name is {} and you are {} years old and your weight is {} kg.\".format(name,age,weight)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "You name is Dhanashree and you are 23 years old and your weight is 58.3 kg.\n"
     ]
    }
   ],
   "source": [
    "user_details(23,58.3,\"Dhanashree\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
