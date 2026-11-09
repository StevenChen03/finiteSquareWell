## The first exercise of Chapter 8

## The Goal:
## Now that we have an idea of how we might implement the finding of roots through bisection, let’s
## test out our idea on a specific function. The sinc function is a function that pops up from time
## to time. It happens to be the Fourier transform of the top-hat window function, and so might be
## imprinted on data that you have Fourier transformed. The sinc function is defined as

## sinc(x)=(sin(x))/x

## where it takes the value of 1 at x = 0. As you can see from the definition, we should already know
## where the zeros of the sinc function happen to be, e.g. where ever the sine function has a zero,
## pi, 2pi, 3pi, . . ., making it a good function to test our algorithm.

## Importations:
import numpy as np
from scipy.optimize import bisect
import time

"""-----Recording How Long The Program Runs For------"""
start_time = time.time()
"""--------------------------------------------------"""


## 1. Write an implementation of the bisection method in your language of choice. (Python)

def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    """
    Finds a root of function 'f' in the interval [a, b] using the bisection method.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("The function must have opposite signs at the interval endpoints.")
    
    for _ in range(max_iter):
        c = (a + b) / 2.0
        # Check if the midpoint is a root or if the interval is within tolerance
        if abs(f(c)) < tol or (b - a) / 2.0 < tol:
            return c
        
        # Decide which side to narrow down
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
            
    return (a + b) / 2.0

## Definition of the unnormalized sinc function
def sinc(x):
    if x == 0:
        return 1.0
    return np.sin(x) / x


## 2. Have your code find the zeros of the sinc function for 0 ≤ x ≤ 10. The answers should be pi,
## 2pi, and 3pi.

## We bracket the expected roots at pi (~3.14), 2pi (~6.28), and 3pi (~9.42)
intervals = [(2, 4), (5, 7), (8, 10)]

print(f"{'Target Root':<15}{'Custom Bisect':<18}{'SciPy Bisect':<18}{'Exact Value'}")
print("-" * 65)

for idx, (back_a, back_b) in enumerate(intervals):
    n = idx + 1
    exact_root = n * np.pi
    
    ## Run custom algorithm
    custom_root = bisection_method(sinc, back_a, back_b, tol=1e-6)

    
    ## 4. Compare your results with the results of a library function if available.
    scipy_root = bisect(sinc, back_a, back_b, xtol=1e-6)
    
    print(f"{f'{n}pi':<15}{custom_root:<18.6f}{scipy_root:<18.6f}{exact_root:.6f}")

## This section here records the time it takes for the program to fully run its code.
## Make sure it records only one program execution.

"""-----Recording How Long The Program Runs For------"""
end_time = time.time()
print("\n")
print(f"Execution time: {(end_time - start_time)/60:.4f} minutes, or {(end_time - start_time):.4f} seconds.")
"""--------------------------------------------------"""

## 3. Are the results of your code as expected? If not go back and see if you may have made a
## mistake somewhere.

## The Code Should Display:

## Target Root    Custom Bisect     SciPy Bisect      Exact Value
## -----------------------------------------------------------------
## 1pi            3.141594          3.141593          3.141593
## 2pi            6.283188          6.283185          6.283185
## 3pi            9.424774          9.424777          9.424778

## By all indications, the results are as expected. To be honest, the SciPy Bisect seems to
## be the more accurate of the two when it comes to matching the values themselves,