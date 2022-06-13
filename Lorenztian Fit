import numpy as np
import pandas as pd
from sympy import symbols as sym
from sympy import *
import matplotlib.pyplot as plt


### Function Defining:

def Average(lst):
  return sum(lst) / len(lst)

def R_Squared (RSS,TSS):
    return np.round(1-(RSS/TSS),2)

def Lorentzian_fit(x_lst, y_lst, x_0_guess, gamma_guess, I_guess,  n=50):
  A, B, I, x=sym("A B I x")
  f=I/(1+(((x-A)/B)**2))
  r=([])
  
  
  for i in x_lst:
    y_i=y_lst[np.where(x_lst==i)]
    r_i=y_i-f.subs(x, i)
    r=list(np.append(r,r_i))

  RSS=0

  for j in r:
    RSS=RSS+j**2
    
  RSS_sol = lambdify([A,B,I], RSS, "numpy")
  S_prime_A=RSS.diff(A)
  S_prime_B=RSS.diff(B)
  S_prime_I=RSS.diff(I)
  sol=nsolve([S_prime_A, S_prime_B,S_prime_I], [A, B, I], [x_0_guess, gamma_guess, I_guess])
  
  coeff=np.round(np.array([float(sol[0]),float(sol[1]),float(sol[2])]),3)
  A,B,I=coeff[0],coeff[1], coeff[2]
  f_set=I/(1+(((x-A)/B)**2))
  f_sol = lambdify(x, f_set)
  
  x_fit = np.linspace(0,x_lst[len(x_lst)-1]+3, n)
  y_fit = np.array([])
  
  for i in x_fit:
    y_i=f_sol(i)
    y_fit=np.append(y_fit, y_i)

  ### R-squared
 
  
  y_avg=Average(y_lst)
  total_var=0
  for i in y_lst:
    total_var=total_var+((i-y_avg)**2)

  RSS_tot=RSS_sol(A,B,I)
  
  R_squared=R_Squared(RSS_tot,total_var)



  return x_fit, y_fit, f, A, B, I, R_squared

### Execution Example

x=np.array([2.41, 2.7, 2.9, 3.1, 3.3, 3.41, 3.5, 3.59, 3.65, 3.9, 4.05])
y=np.array([33.5, 44.83, 65.27, 104.7, 203.2, 320.2, 746.3, 1325, 832.5, 261.1, 179.8])

fit=Lorentzian_fit(x,y,x[np.argmax(y)],x[(np.argmax(y))+1]-x[(np.argmax(y))-1], max(y), 500)
print ("I="+str(fit[5])+"\n"+
       "gamma="+str(fit[4])+"\n"+
       "x_0="+str(fit[3])+"\n"+
       "R^2="+str(fit[6])+"\n"+
       "f="+"I/(1+(((x-x_0)/gamma)**2))")

### Ploting

plt.plot(x, y,"bo")
plt.plot(fit[0], fit[1],"r")
plt.xticks(np.arange(0, x[len(x)-1]+3, 0.5))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Ampiltude [mV]")
plt.legend(["Obs","fit"])
plt.grid()
