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

def Exp_fit(x_lst, y_lst, n):
  K, T, t=sym("K T t")
  f=K*(1-(exp(-t/T)))
  r=([])
  
  for i in x_lst:
    y_i=y_lst[np.where(x_lst==i)]
    r_i=y_i-f.subs(t, i)
    r=list(np.append(r,r_i))

  RSS=0
  
  for j in r:
    RSS=RSS+j**2

  """S=simplify(S)"""
  RSS_sol = lambdify([K,T], RSS, "numpy")
  S_prime_K=RSS.diff(K)
  S_prime_T=RSS.diff(T)
  sol=nsolve([S_prime_K, S_prime_T], [K, T], [11, 1])
  coeff=np.array([np.round(float(sol[0]),2),np.round(float(sol[1]),2)])

  K,T=coeff[0],coeff[1]
  f=K*(1-exp(-t/T))
  f_sol = lambdify(t, f)
  
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

  RSS_tot=RSS_sol(K,T)
  


  R_squared=R_Squared(RSS_tot,total_var)

  return x_fit, y_fit, f, R_squared

### Data Read:

print ("Enter Path (without Quotation marks):")
path=input()
df=pd.read_excel(path)
df=df.to_numpy()

### Execution Example

x=np.array([1, 2, 3, 4, 5])
y=np.array([6, 5, 7, 10,9])


fit=Exp_fit(x,y,500)

### Ploting

plt.plot(x, y, "bo")
plt.plot(fit[0], fit[1], "r")


plt.xticks(np.arange(0, fit[0][len(fit[0])-1], 0.5))
label_1=str(fit[2])+"\n"+"R^2="+str(fit[3])
plt.legend(["Obs", label_1])
plt.grid()
