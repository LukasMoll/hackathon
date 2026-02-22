# BotBattle - Tronkscript

- URL: https://botbattle.be/docs/tronkscript
- Fetched at (UTC): 2026-02-20T21:41:56.401339+00:00

## Extracted Content

BotBattle
Info
Docs
Tournaments
Rankings
Games
Livestream
Editor
Sign In
Tronkscript

Dig deeper into the Tronkscript programming language.

Tronkscript is a programming language designed for BotBattle. It provides a simple syntax with support for variables, control structures (conditionals and loops), and functions. Tronkscript programs execute from top to bottom, starting with the first statement. The language uses integer-only arithmetic and supports both global and local variable scoping.

Tronkscript is case-sensitive and uses a structured syntax with explicit block endings (e.g., end if, end while, end function). This makes the code more readable and helps prevent common programming errors.

Syntax

Tronkscript uses a straightforward syntax where statements are written one per line. Variables are created implicitly when first assigned a value. The language supports decimal, binary, and hexadecimal number formats.

Variable names must start with a letter or underscore, followed by any combination of letters, digits, or underscores. Examples of valid variable names:

x
my_variable
value123
Value123 -- different variable from value123!
_counter

Tronkscript supports three number formats:

Decimal: 42, 100, 0
Binary: 0b1010 (prefix with 0b)
Hexadecimal: 0x1A (prefix with 0x)

Comments start with double dashes (--) and continue to the end of the line:

-- This is a comment
x = 10  -- This is also a comment

Tronkscript is case-sensitive, meaning myVar and myvar are different variables. Whitespace (spaces and tabs) is generally ignored except where it's needed to separate tokens.

 Expression Limitations: Tronkscript expressions are limited to exactly two operands. Binary operations like x + y are valid, but chained operations like x + y + z are not. To compute complex expressions, break them into multiple statements using intermediate variables.
Operators

Tronkscript supports a variety of operators for arithmetic, bitwise operations, and comparisons.

Binary Operators

Binary operators perform operations on two values:

Operator	Description
+	Addition
-	Subtraction
*	Multiplication
/	Division (integer division, truncates toward zero)
%	Modulus (remainder after division)
&	Bitwise AND
^	Bitwise XOR
|	Bitwise OR
Unary Operators

Unary operators operate on a single value:

Operator	Description
-	Negation (unary minus)
+	Positive (absolute value)
~	Bitwise NOT
!	Logical NOT (returns 1 if value is 0, otherwise 0)
Comparison Operators

Comparison operators are used in conditions and return 1 for true, 0 for false:

Operator	Description
==	Equal to
!=	Not equal to
>	Greater than
<	Less than
>=	Greater than or equal to
<=	Less than or equal to
Variables and Assignments

Variables in Tronkscript are created implicitly when first assigned a value. There is no need to declare variables before using them.

Basic assignment uses the = operator:

x = 10
y = 20
result = x + y

Tronkscript supports compound assignment operators that combine an operation with assignment:

x = 10
x += 5    -- x is now 15 (equivalent to x = x + 5)
x -= 3    -- x is now 12 (equivalent to x = x - 3)
x *= 2    -- x is now 24 (equivalent to x = x * 2)
x /= 4    -- x is now 6 (equivalent to x = x / 4)

Unary operations can also be used in assignments:

x = 10
y = -x    -- y is -10
z = !x    -- z is 0 (because x is not zero)
w = ~x    -- w is bitwise NOT of x

Binary operations combine exactly two values. You cannot chain multiple operations in a single statement:

x = 10
y = 5
sum = x + y        -- 15 (valid: two operands)
diff = x - y       -- 5 (valid: two operands)
prod = x * y       -- 50 (valid: two operands)
quot = x / y       -- 2 (integer division, valid: two operands)
mod = x % y        -- 0 (valid: two operands)
and_result = x & y -- bitwise AND (valid: two operands)
or_result = x | y  -- bitwise OR (valid: two operands)
xor_result = x ^ y -- bitwise XOR (valid: two operands)
 Important: Binary operations can only have exactly two operands. To compute expressions with more than two values, you must break them into multiple statements. For example, to compute a * b * c, you would write:
temp = a * b
result = temp * c
Variable Scoping

Tronkscript has two scopes: global and local. Variables assigned outside of any function are global. Variables assigned inside a function are local to that function, unless a global variable with the same name already exists.

When you assign a variable inside a function, Tronkscript checks in this order:

If the variable already exists as a local in the current function, update it.
If a global variable with that name exists, update the global. There is no shadowing. You cannot create a local variable that hides a global.
Otherwise, create a new local variable in the current function.

This means that functions can read and modify global variables, and new variables created inside a function are private to that function call:

score = 0  -- global variable

function add_score(points) do
    score = score + points  -- updates the global
    temp = score * 2        -- temp is local, not visible outside
end function

add_score(10)
println(score)  -- prints 10
 Recursion: Each function call gets its own local variables. In recursive calls, local variables from one call cannot be seen by another call of the same function.
Control Structures
Conditional Statements

Conditional statements allow your program to execute different code based on conditions. The basic syntax is:

if (x == 10) then
    print(x)
end if

You can add an else clause for when the condition is false:

if (x > 10) then
    print(1)  -- print 1 to indicate "greater"
else
    print(0)  -- print 0 to indicate "not greater"
end if

Multiple conditions can be chained using else if:

if (x > 10) then
    print(1)  -- greater than 10
else if (x == 10) then
    print(0)  -- equal to 10
else
    print(-1)  -- less than 10
end if
While Loops

While loops execute a block of code repeatedly as long as a condition is true:

x = 0
while (x < 10) do
    print(x)
    x = x + 1
end while

The condition is checked before each iteration. If the condition is false initially, the loop body will never execute.

For Loops

For loops iterate over a range of values:

for (i = 0 to 10) do
    print(i)
end for

The loop variable (in this case i) is automatically incremented after each iteration. The loop runs from the start value (inclusive) to the end value (inclusive). If the start value is greater than the end value, the loop body never executes.

You can use variables or numbers for the start and end values:

start = 5
end = 15
for (i = start to end) do
    print(i)
end for
Functions

Functions allow you to organize code into reusable blocks. Functions can take arguments and return values.

Defining Functions

Functions are defined using the function keyword, followed by the function name, parameters in parentheses, and the function body:

function add(a, b) do
    result = a + b
    return result
end function

Functions can take multiple parameters, separated by commas. To compute expressions with multiple operations, you must break them into multiple statements:

function multiply(a, b, c) do
    temp = a * b
    result = temp * c
    return result
end function

Functions can return multiple values. Note that return statements can only contain variables or numbers, not expressions:

function divide_and_mod(a, b) do
    quotient = a / b
    remainder = a % b
    return quotient, remainder
end function
 Important: Return statements can only contain variables or number literals, not expressions. You must compute the result into a variable first, then return that variable. For example, return a + b is invalid; you must write result = a + b followed by return result.
Calling Functions

Functions can be called without capturing return values:

print(42)
my_function(x, y)

Or you can capture return values using assignment:

result = add(5, 3)
-- result is now 8

When a function returns multiple values, you can capture them all:

q, r = divide_and_mod(10, 3)
-- q is 3, r is 1

Functions cannot be defined inside other functions. All functions are defined at the top level of your program and can be called from anywhere.

  Function parameters are local to the function. Variables created inside a function are also local, but assigning to a name that already exists as a global will update the global instead. See the Variable Scoping section above for details.
Execution Model & Tick System

TronkScript is a tick-based interpreter. Each game tick, your bot executes exactly one statement (an assignment, a condition check, a loop check, or a function call). The game then advances to the next tick.

In Tronk, players move once every 10,000 ticks, so your bot has a budget of 10,000 ticks between moves to decide what to do.

What costs a tick?
Assignments (x = 5, y = x + 1): 1 tick
Condition checks (if, else if, while condition, for condition): 1 tick
Function calls: depends on the function (see below)
Return statements: 1 tick
Function tick costs

Every function has a tick cost. When your bot calls a function, it is paused for that many ticks. For example, calling a game function like turnLeft() costs 100 ticks.

Built-in language functions have the following costs:

Function	Tick cost
print(), println()	0 (free)
min(), max(), abs(), rand(), square(), pow(), sqrt(), root(), exp2(), gcd(), lcm(), is_prime(), reverse()	1
wait(n)	n (variable)

User-defined functions cost 1 tick to enter (the function call itself), plus whatever ticks the function body uses.

 Zero-cost functions: Functions with a tick cost of 0 (like print) do not consume a tick. Execution immediately continues to the next statement without advancing the game.
 Stay alive: Your program must run in an infinite loop (e.g., while (1) do ... end while). If your code reaches the end and stops executing, your bot dies immediately. Use wait() to skip ticks without doing anything if you have no more work to do before the next move.
Coding Style

While Tronkscript doesn't enforce a specific coding style, following these conventions will make your code more readable and maintainable:

Indentation: Use consistent indentation (spaces or tabs) for nested blocks. This makes it easier to see the structure of your code.
Naming: Use descriptive variable and function names. Consider using snake_case for multi-word names (e.g., my_variable, calculate_sum).
Comments: Add comments to explain complex logic or document what your code does. Comments start with --.
Spacing: Add spaces around operators for readability:x = y + z is clearer than x=y+z.

Example of well-formatted code:

-- Calculate the sum of numbers from 1 to n
function sum_to_n(n) do
    total = 0
    for (i = 1 to n) do
        total = total + i
    end for
    return total
end function

-- Main program
max_value = 10
result = sum_to_n(max_value)
print(result)

Note that in the loop, total = total + i is valid because it uses exactly two operands (total and i). The assignment operator = is separate from the binary + operator.

Built-in Functions

Here you find an extended description of the built-in functions available in Tronkscript.

### print
Outputs a value to the debug console. Takes any number of arguments and prints them space-separated.

Arguments
Argument	Type
The value to print	The argument can be a number or a variable
Examples
Example	Description
print(42)
Print a number
print(x)
Print a variable
Notes
This function takes no ticks to complete.
### println
Same as print, but adds a newline at the end.

Arguments
Argument	Type
The value to print	The argument can be a number or a variable
Examples
Example	Description
println(x)
Print x with newline
Notes
This function takes no ticks to complete.
### min
Takes two integers and returns the smaller one.

Arguments
Argument	Type
First integer	The argument can be a number or a variable
Second integer	The argument can be a number or a variable
Examples
Example	Description
result = min(5, 3)
result is 3
result = min(x, y)
result is the smaller of x and y
Notes
This function takes 1 tick to complete.
### max
Takes two integers and returns the larger one.

Arguments
Argument	Type
First integer	The argument can be a number or a variable
Second integer	The argument can be a number or a variable
Examples
Example	Description
result = max(5, 3)
result is 5
result = max(x, y)
result is the larger of x and y
Notes
This function takes 1 tick to complete.
### abs
Takes an integer and returns its absolute value (makes it positive).

Arguments
Argument	Type
Integer value	The argument can be a number or a variable
Examples
Example	Description
result = abs(-5)
result is 5
result = abs(x)
result is the absolute value of x
Notes
This function takes 1 tick to complete.
### gcd
Takes two integers and returns their greatest common divisor using the Euclidean algorithm.

Arguments
Argument	Type
First integer	The argument can be a number or a variable
Second integer	The argument can be a number or a variable
Examples
Example	Description
result = gcd(48, 18)
result is 6
result = gcd(x, y)
result is the GCD of x and y
Notes
This function takes 1 tick to complete.
### lcm
Takes two integers and returns their least common multiple. Uses gcd internally.

Arguments
Argument	Type
First integer	The argument can be a number or a variable
Second integer	The argument can be a number or a variable
Examples
Example	Description
result = lcm(4, 6)
result is 12
result = lcm(x, y)
result is the LCM of x and y
Notes
This function takes 1 tick to complete.
### is_prime
Returns 1 if the integer is prime, 0 otherwise. Handles edge cases: 0, 1, and negative numbers return 0.

Arguments
Argument	Type
Integer to check	The argument can be a number or a variable
Examples
Example	Description
result = is_prime(7)
result is 1 (prime)
result = is_prime(8)
result is 0 (not prime)
Notes
This function takes 1 tick to complete.
### reverse
Takes an integer and reverses its bits (32-bit). The result is a new integer with reversed bit pattern.

Arguments
Argument	Type
Integer whose bits to reverse	The argument can be a number or a variable
Examples
Example	Description
result = reverse(1)
result is 2147483648 (bit reversed)
result = reverse(x)
result is x with bits reversed
Notes
This function takes 1 tick to complete.
### rand
Returns a random integer. With no arguments: rand(0, 0xffff). With one argument: rand(0, max). With two arguments: rand(min, max). Min can be negative but must be less than max.

Arguments
Argument	Type
Minimum value (optional, default 0)	The argument can be a number or a variable
Maximum value (optional, default 0xffff)	The argument can be a number or a variable
Examples
Example	Description
result = rand()
result is random between 0 and 65535
result = rand(100)
result is random between 0 and 100
result = rand(10, 20)
result is random between 10 and 20
Notes
This function takes 1 tick to complete.
### square
Takes an integer and returns x² (x * x).

Arguments
Argument	Type
Integer to square	The argument can be a number or a variable
Examples
Example	Description
result = square(5)
result is 25
result = square(x)
result is x * x
Notes
This function takes 1 tick to complete.
### pow
Takes two integers and returns x^y. If exponent is negative, returns 0. If both are 0, returns 1.

Arguments
Argument	Type
Base	The argument can be a number or a variable
Exponent	The argument can be a number or a variable
Examples
Example	Description
result = pow(2, 3)
result is 8
result = pow(x, y)
result is x raised to power y
Notes
This function takes 1 tick to complete.
### sqrt
Takes an integer and returns the floor of its square root. For negative numbers, returns 0.

Arguments
Argument	Type
Integer to take square root of	The argument can be a number or a variable
Examples
Example	Description
result = sqrt(25)
result is 5
result = sqrt(x)
result is floor of square root of x
Notes
This function takes 1 tick to complete.
### root
Takes two integers and returns the floor of the nth root of x. If n is 0, returns 0. For negative x, returns 0.

Arguments
Argument	Type
Integer to take root of	The argument can be a number or a variable
Root degree	The argument can be a number or a variable
Examples
Example	Description
result = root(8, 3)
result is 2 (cube root)
result = root(x, n)
result is floor of nth root of x
Notes
This function takes 1 tick to complete.
### exp2
Takes an integer and returns 2^x. For negative exponents, returns 0. Uses bit shift when possible for efficiency.

Arguments
Argument	Type
Exponent	The argument can be a number or a variable
Examples
Example	Description
result = exp2(3)
result is 8
result = exp2(x)
result is 2 raised to power x
Notes
This function takes 1 tick to complete.
### wait
Pauses the bot's execution for the specified number of ticks. The bot will do nothing during this time. The argument must be a non-negative integer.

Arguments
Argument	Type
Number of ticks to wait	The argument can be a number or a variable
Examples
Example	Description
wait(10)
Wait for 10 ticks
wait(x)
Wait for x ticks
Notes
This function takes variable ticks to complete.

Copyright © BotBattle FV 2023-2026
