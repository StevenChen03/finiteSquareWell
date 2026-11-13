## This file will use the lessons learned from the Chapter 6 Exercises and calculate how long
## it takes for a photon to journey from the SUn's core to the surface.

## Importations:
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e


"""-----Recording How Long The Program Runs For------"""
start_time = time.time()
"""--------------------------------------------------"""


# Physical constants and initial parameters
a = 1e-10          # Half-width of the well (m)
V0_initial = 4.0e-17  # Potential depth (J)

def calc_z0(V0):
    return (a / hbar) * np.sqrt(2 * m_e * V0)

# Transcendental equations (8.13 and 8.14)
def f_sym(z, z0):
    return np.sqrt((z0 / z)**2 - 1) - np.tan(z)

def f_anti(z, z0):
    return np.sqrt((z0 / z)**2 - 1) + 1.0 / np.tan(z)


## Step 1: Using a=1×10^(-10)  m, and v_0=4.0×10^(-17)  J, create a plot showing the
## functions √((z_0⁄z)^2-1) and tan⁡z versus z. Create a separate plot with √((z_0⁄z)^2-1)
## and -cot⁡z versus z. Note that you may want to set the y-limits of your plot to be
## between 0 and 8 or 10. The intersections of the lines are the solutions we seek to find.


# ---------------------------------------------------------
# STEP 1 & STEP 2: Plotting the Curves and Root Equations
# ---------------------------------------------------------
z0 = calc_z0(V0_initial)
z_vals = np.linspace(0.01, z0 - 1e-5, 2000)
rhs = np.sqrt((z0 / z_vals)**2 - 1)

# Mask large jumps near asymptotes so Matplotlib doesn't draw vertical lines
tan_vals = np.tan(z_vals)
tan_vals[np.abs(tan_vals) > 20] = np.nan

neg_cot_vals = -1.0 / np.tan(z_vals)
neg_cot_vals[np.abs(neg_cot_vals) > 20] = np.nan

# Step 1 Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(z_vals, rhs, label=r'$\sqrt{(z_0/z)^2 - 1}$', color='black')
ax1.plot(z_vals, tan_vals, label=r'$\tan(z)$ (Symmetric)', color='tab:blue')
ax1.set_ylim(0, 10)
ax1.set_xlabel('z')
ax1.set_ylabel('Value')
ax1.set_title('Step 1a: Symmetric Intersections')
ax1.legend()
ax1.grid(True)

ax2.plot(z_vals, rhs, label=r'$\sqrt{(z_0/z)^2 - 1}$', color='black')
ax2.plot(z_vals, neg_cot_vals, label=r'$-\cot(z)$ (Antisymmetric)', color='tab:orange')
ax2.set_ylim(0, 10)
ax2.set_xlabel('z')
ax2.set_title('Step 1b: Antisymmetric Intersections')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig("FiniteSquareWell_One_graph.pdf")
plt.clf() ## This helps clears the previous graph and make room for the second graph


## Step 2: With the same numerical values as in the previous step, create a plot of the
## functions in equations (8.13) and (8.14). Here you may want to set your y-limits to be
## between -8 and 8. Looking at these plots, do you see anything that might be problematic
## for our bisection algorithm? Are there places where the sign of the function might
## change but we don’t expect to find a zero? (i.e. Are there nearly vertical lines?)

## Equation 8.13: √(z_0^2-z^2 )-z tan⁡z=0     ->      √((z_0⁄z)^2-1)-tan⁡z=0

## Equation 8.14: √(z_0^2-z^2 )+z cot⁡z=0     ->      √((z_0⁄z)^2-1)+cot⁡z=0


# Step 2 Plot
sym_vals = f_sym(z_vals, z0)
sym_vals[np.abs(sym_vals) > 20] = np.nan
anti_vals = f_anti(z_vals, z0)
anti_vals[np.abs(anti_vals) > 20] = np.nan

plt.figure(figsize=(8, 5))
plt.plot(z_vals, sym_vals, label='Eq 8.13 (Symmetric)', color='tab:blue')
plt.plot(z_vals, anti_vals, label='Eq 8.14 (Antisymmetric)', color='tab:orange')
plt.axhline(0, color='black', linewidth=1)
plt.ylim(-8, 8)
plt.xlabel('z')
plt.ylabel('f(z)')
plt.title('Step 2: Root-Finding Functions')
plt.legend()
plt.grid(True)
plt.savefig("FiniteSquareWell_Two_graph.pdf")
plt.clf()


## Step 3: Our pair finding algorithm will find pairs where there is a sign change in our
## function, but it doesn’t account for any discontinuities there may be in the equation.
## Because of the tangent function, there are points where the functions whose zeros we are
## trying to find will have a sign change, but the function is actually diverging at that point.
## We want to ignore these points, otherwise our bisection algorithm will analyze them and
## never actually converge. The easiest way to deal with this is to introduce a max_iter
## parameter in our zero finder and then only add a zero to the list if the number of iterations
## to find the zero is less than the maximum number. Modify your bisection algorithm so
## that we can avoid the code hanging while trying to find a zero where no zero exists.


## Step 4: Run your zero finding algorithm on the functions in equations (8.13) and (8.14).
## You should find a total of 5, three from the symmetric case and two from the
## antisymmetric case.


# ---------------------------------------------------------
# STEP 3 & STEP 4: Modified Bisection & Finding Zeros
# ---------------------------------------------------------
def find_pairs(func, z0, n_steps=10000):
    zs = np.linspace(1e-6, z0 - 1e-9, n_steps)
    ys = func(zs, z0)
    pairs = []
    for i in range(len(zs) - 1):
        if ys[i] * ys[i + 1] < 0:
            pairs.append((zs[i], zs[i + 1]))
    return pairs

def bisection(func, low, high, z0, tol=1e-10, max_iter=50):
    f_low = func(low, z0)
    f_high = func(high, z0)
    if f_low * f_high >= 0:
        return None
    
    for iteration in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = func(mid, z0)
        
        # Converged only if function value is genuinely close to 0
        if abs(f_mid) < tol:
            return mid
        
        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
            
    # Reached max_iter without |f(mid)| < tol (likely an asymptote)
    return None

def find_all_zeros(func, z0):
    pairs = find_pairs(func, z0)
    zeros = []
    for low, high in pairs:
        root = bisection(func, low, high, z0)
        if root is not None:
            zeros.append(root)
    return zeros

zeros_sym = find_all_zeros(f_sym, z0)
zeros_anti = find_all_zeros(f_anti, z0)
print(f"Step 4 (V0 = 4.0e-17 J, z0 = {z0:.4f}):")
print(f"Symmetric zeros ({len(zeros_sym)}): {np.round(zeros_sym, 4)}")
print(f"Antisymmetric zeros ({len(zeros_anti)}): {np.round(zeros_anti, 4)}")


## Step 5: As you might expect, if we lower bottom of the well (in our case increase V_0 since
## we included the minus sign elsewhere) enough, we should approach the case of the
## infinite square well. The allowed energies for the infinite square well are given by

## E_n=(n^2 π^2 ℏ^2)/(2ma^2 ) 

## but that is for a being the entire width of the well. Our finite square well has a width of
## 2a and we have a negative V_0 where as the infinite square well solution assumes that to
## be zero, so

## E_n=(n^2 π^2 ℏ^2)/(8ma^2 )-V_0.

## From this equation and the values given to you in this project, have your code compute
## the first N values of this equation where N is the total number of zeros you have found.
## The odd n should correspond to energies from the symmetric case and the even n from
## the antisymmetric. Compare the energies you found to those from the infinite square well.
## Note that to do this comparison we need to take the locations of our zeros and convert
## them to the actual energies. You should find

## E=(ℏ^2 z^2)/(2ma^2 )-V_0

## How close are the energies?


## Step 6: Try making the well deeper, i.e. increasing the value of V_0. Start by doubling it to
## V_0=8.0×10^(-17). How many zeros do you find now? Note that this is also the number
## of bound states. How do the energies compare to the infinite square well case? Is there
## any kind of trend in the agreement (or lack there of) of the energies?


## Step 7: Now make the well much deeper by setting V_0=8.0×10^(-15). How many zeros
## do you find now? How do the energies compare with those of the infinite square well? As
## a quick visual test is to plot the energies versus each other and compare to a line with a
## slope of 1. You should find that the energies agree pretty well for the lowest energy states,
## but disagree more and more as you get near the top of the well. Does this make sense?
## Why or why not?


# ---------------------------------------------------------
# STEP 5, 6, 7: Energy Comparisons Across Well Depths
# ---------------------------------------------------------
def z_to_energy(z, V0):
    return (hbar**2 * z**2) / (2 * m_e * a**2) - V0

def infinite_well_energy(n, V0):
    return (n**2 * np.pi**2 * hbar**2) / (8 * m_e * a**2) - V0

def analyze_well(V0, label):
    z0_val = calc_z0(V0)
    z_s = find_all_zeros(f_sym, z0_val)
    z_a = find_all_zeros(f_anti, z0_val)
    all_z = sorted(z_s + z_a)
    
    E_finite = np.array([z_to_energy(z, V0) for z in all_z])
    E_inf = np.array([infinite_well_energy(n, V0) for n in range(1, len(all_z) + 1)])
    
    print(f"\n--- {label} (V0 = {V0:.1e} J, z0 = {z0_val:.4f}) ---")
    print(f"Total bound states found: {len(all_z)} ({len(z_s)} sym, {len(z_a)} anti)")
    for n in range(min(5, len(all_z))):
        print(f"  n={n+1}: E_finite = {E_finite[n]:.4e} J | E_inf = {E_inf[n]:.4e} J")
    if len(all_z) > 5:
        print(f"  ... (highest state n={len(all_z)}): E_finite = {E_finite[-1]:.4e} J | E_inf = {E_inf[-1]:.4e} J")
    return E_finite, E_inf

E_fin_1, E_inf_1 = analyze_well(4.0e-17, "Step 5")
E_fin_2, E_inf_2 = analyze_well(8.0e-17, "Step 6")
E_fin_3, E_inf_3 = analyze_well(8.0e-15, "Step 7")


# Step 7 Visual Comparison Plot
plt.figure(figsize=(6, 6))
plt.scatter(E_inf_3, E_fin_3, color='tab:purple', s=15, label='Bound States (V0 = 8e-15 J)')
min_e, max_e = min(E_inf_3), max(E_fin_3)
plt.plot([min_e, max_e], [min_e, max_e], 'k--', label='Slope = 1 (Exact Agreement)')
plt.xlabel('Infinite Square Well Energy (J)')
plt.ylabel('Finite Square Well Energy (J)')
plt.title('Step 7: Finite vs. Infinite Well Energies')
plt.legend()
plt.grid(True)
plt.savefig("FiniteSquareWell_Three_graph.pdf")
plt.clf()


## This section here records the time it takes for the program to fully run its code
## since there's so much of it here. Make sure it records only one program execution.

"""-----Recording How Long The Program Runs For------"""
end_time = time.time()
print("\n")
print(f"Execution time: {(end_time - start_time)/60:.4f} minutes., or {(end_time - start_time):.4f} seconds.")
"""--------------------------------------------------"""