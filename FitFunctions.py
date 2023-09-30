"""
Created on Sun Jan 10 16:51:31 2021

@author: Narmin Ghaffari Laleh, modified by Frithjof Sy
"""
from scipy.optimize import differential_evolution
from scipy.integrate import odeint
import numpy as np
import warnings
from scipy.integrate import solve_ivp
from gekko import GEKKO

##############################################################################
# FUNCTIONS TO FIT THE DATA
##############################################################################

def Select_Function(functionName):
    if functionName == 'DoubleExponential':
        def fitfunc(t, f, d, g, v0):
            def my_diff(dim, t):
                return (d*(f-1)*dim**(-d) + f*g*dim**g)
            Ca0 = v0
            Casol = solve_ivp(my_diff, t, Ca0)
            return Casol[:,0]

def Select_Function(functionName):
    if functionName == 'DoubleExponential':
        def fitfunc(t, f, d, g):
            exp1 = np.vectorize(lambda x: np.exp(-d*x))
            exp2 = np.vectorize(lambda x: np.exp(g*x))
            Casol = (f-1)*exp1(t) + f*exp2(t)
            return Casol
def Select_Function(functionName):
    if functionName == 'Lottka-Volterra':
        def fitfunc(t, dim, r1, r2, K, d1, d2):
            T = np.sum(dim)
            y = dim[0]
            x = 0.02*dim[0]
            dydt = r1*y*(1-T/K)-d1*y
            dxdt = r2*x*(1-T/K)-d2*x
            return [dydt,dxdt]

# elif functionName == 'Lottka-Volterra':
#         def fitfunc(t, r1, r2, K, d1, d2, v0):
#             def my_diff(dim, t):
#                 T = np.sum(dim)
#                 dydt = np.zeros(2)
#                 dydt[0] = r1*dim[0]*(1-T/K)-d1*dim[0]
#                 dydt[1] = r2*0.02*dim[0]*(1-T/K)-d2*0.2*dim[0]
#                 return dydt
#             Ca0 = v0
#             Casol = odeint(my_diff, )
#             #Casol = solve_ivp(my_diff, (t[0],t[-1]), Ca0)
#             return Casol[:,0]

# def fitfunc(t, sigma, r, v0):
#     def my_diff(dim, t):
#         m = GEKKO() # create GEKKO model
#         k = 2
#         M = 1
#         d = 0.01
#         Kmax = 1
#         b = 1
#         g = 0.5
       
#         # create GEKKO variables
#         x = m.Var(dim[0])
#         y = m.Var(0.02*dim[0])

#         # create GEKKO equations
#         dxdt = x*r*(1-(x/Kmax*m.exp(-g*y)))-(M*x/(k+b*y))-d*x
#         dydt = sigma*(M*b/((k+b*y)**2) - (r*x*g/(Kmax*m.exp(-g*y))))
#         m.Equation(x.dt()==dxdt)
#         m.Equation(y.dt()==dydt)

#         # solve ODE
#         m.time = np.linspace(0,20) # time points
#         m.options.IMODE = 4 # dynamic simulation
#         m.options.NODES = 3 # collocation nodes
#         m.solve(disp=False, remote=False)

#         return [x.value, y.value]
   
#     Ca0 = [v0,v0]
#     Casol = my_diff(dim, t)
#     return Casol[:,0]


def Select_Fucntion(functionName):
    if functionName == 'Exponential':
        def fitfunc(t, alpha, beta, v0):     
            def myode(dim, t):
                return (alpha - beta)*dim        
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]

    elif functionName == 'DoubleExponential':
        def fitfunc(t, f, d, g):
            t_array=np.array(t)
            Casol = (1-f)*np.exp(-d*t_array) + f*np.exp(g*t_array)
            print("f=" + str(f),"d=" + str(d),"g=" + str(g))
            return Casol
        
    elif functionName == 'Lottka-Volterra':
        def fitfunc(t, r1, r2, K, d1, d2, v0):
            def my_diff(dim, t):
                T = np.sum(dim)
                y = dim[0]
                x = 0.02*dim[0]
                dydt = r1*y*(1-T/K)-d1*y
                dxdt = r2*x*(1-T/K)-d2*x
                return [dydt,dxdt]
            Ca0 = [v0,v0]
            Casol = odeint(my_diff, Ca0, t)
            print("r1=" + str(r1), "r2="+ str(r2), "K="+ str(K), "d1="+ str(d1), "d2="+ str(d2))
            #Casol = solve_ivp(my_diff, (t[0],t[-1]), Ca0)
            return Casol[:,0]
    
    elif functionName == 'ModKuznetsov':
        def fitfunc(t, sigma, rho, nu, mu, delta, alpha, v0):
            def my_diff(dim, t):
               x = dim[0]
               y = 0.02*dim[0]
               E0 = 10**7
               T0 = 10**9
               dxdt = sigma + (rho*x*y/(nu + y)) - delta*x - mu * x*y
               dydt =  alpha*y - E0/T0*x*y
               return [dxdt,dydt]
            Ca0 = [v0,v0]
            Casol = odeint(my_diff, Ca0, t)
            return Casol[:,0]
        
    # elif functionName == 'ModKuznetsov': # new version with only three free params, sigma, mu, delta
    #     def fitfunc(t, sigma, mu, delta, v0):
    #         def my_diff(dim, t):
    #             x = dim[0]
    #             y = 0.02*dim[0]
    #             alpha = 0.1
    #             rho = 90
    #             nu = 5
    #             E0 = 10**7
    #             T0 = 10**9
    #             dxdt = sigma + (rho*x*y/(nu + y)) - delta*x - mu * x*y
    #             dydt =  alpha*y - E0/T0*x*y
    #             return [dxdt,dydt]
    #         Ca0 = [v0,v0]
    #         Casol = odeint(my_diff, Ca0, t)
    #         return Casol[:,0]


    # elif functionName == 'ModKuznetsov':    # new implementation with only three free params, sigma, mu, delta
    #     #def fitfunc(t, sigma, mu, delta, v0):
    #     #def fitfunc(t, sigma, rho, nu, mu, delta, alpha, v0):        
    #     def fitfunc(t,sigma, mu, delta, v0):
    #         def my_diff(t):
    #             m = GEKKO(remote=False) # create GEKKO model
    #             #x = dim[0]
    #             #u = 0.02*dim[0]
    #             alpha = 0.1 # 1_0.1, 2_0.1, 3_0.1, 4_0.1, 5_0.1 (fünf stoppte nach 4), 6_0.1
    #             #sigma = 0.25
    #             rho = 9 # 1_0.1, 2_0.5, 3_1, 4_5, 5_10, 6_20
    #             nu = 4 # 1_0.1, 2_1, 3_1, 4_1, 5_2, 6_5
    #             E0 = 10**7
    #             T0 = 10**9

                                          
    #             # create GEKKO variables
    #             x = m.Var(Ca0[0],lb=1e-6)
    #             #x = m.Var(Ca0[0],lb=0)
    #             #y = m.Var(0.02*Ca0[0],lb=1e-6)
    #             u = m.Var(0.02*Ca0[0],lb=1e-6)
    #            # u = m.Var(0.02*Ca0[0],lb=0)

            
    #             # create GEKKO equations

                
    #             m.Equations([x.dt() ==  sigma + (rho*x*u/(nu + u)) - delta*x - mu*x*u, u.dt() == alpha*u - E0/T0*x*u])
    #             # solve ODE
    #             m.time = t # time points
    #             #m.options.IMODE = 5 # dynamic simulation
    #             m.options.IMODE = 5 # dynamic simulation
    #             m.options.NODES = 5 # collocation nodes
    #             #m.options.EV_TYPE = 2
    #             m.solve()

    #             #return [x.value, y.value]
    #             return [x.value, u.value]
        
    #         Ca0 = [v0,v0]            
    #         Casol = my_diff(t)
    #         return Casol[0]
        
    # elif functionName == 'GameTheory':
    #     def fitfunc(t, sigma, r, v0):
    #         def my_diff(dim, t):
    #             x = dim[0]
    #             y = 0.02*dim[0]
    #             m=0
    #             d=0.01
    #             Kmax = 1
    #             k=2
    #             b=1
    #             g=0.5
    #             dxdt = x*r*(1-(x/(Kmax*np.exp(-g*y))))-(m*x/(k+b*y))-d*x
    #             dydt = sigma*(m*b/(k+b*y)**2 - (r*x*g/(Kmax*np.exp(-g*y))))
    #             return [dxdt,dydt]
    #         Ca0 = [v0,v0]
    #         Casol = odeint(my_diff, Ca0, t)
    #         return Casol[:,0]

    elif functionName == 'GameTheory':    
        def fitfunc(t, sigma, r, v0):
            def my_diff(t):
                m = GEKKO(remote=False) # create GEKKO model
                # k = 2
                # M = 1 #Therapy
                # d = 0.01
                # Kmax = 1
                # b = 1
                # g = 0.5
                print("sigma=" + str(sigma), "r="+ str(r))
                k = 1.5
                M = 1 #Therapy
                d = 0.05
                Kmax = 1
                b = 2
                g = 0.5

                                          
                # create GEKKO variables
                x = m.Var(Ca0[0],lb=0)
                #y = m.Var(0.02*Ca0[0],lb=1e-6)
                u = m.Var(0.02*Ca0[0],lb=1e-6)

            
                # create GEKKO equations

                
                #dxdt = x*r*(1-(x/Kmax*m.exp(-g*y)))-(M*x/(k+b*y))-d*x
                #dydt = sigma*(M*b/((k+b*y)**2) - (r*x*g/(Kmax*m.exp(-g*y))))
                #dxdt = x*r1*(1-(x/Kmax1*m.exp(-g1*y)))-(M1*x/(k1+b1*y))-d1*x
                #dydt = sigma1*(M1*b1/((k1+b1*y)**2) - (r1*x*g1/(Kmax1*m.exp(-g1*y))))
                #m.Equation(x.dt()==dxdt)
                #m.Equation(y.dt()==dydt)

                #m.Equations([x.dt()==  x*(r*(m.exp(-g*u))*(1-x/(Kmax*(m.exp(-g*u))))-M/(k+b*u)-d), u.dt() == sigma*(-g*r*(m.exp(-g*u))*(1-x*(m.exp(g*u))/(Kmax))+(b*M)/((b*u+k)**2)-g*r*x/(Kmax))])
                m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(-g*u))))-M/(k+b*u)-d), u.dt() == sigma*((-g*r*x*(m.exp(g*u)))/(Kmax)+(b*M)/(b*u+k)**2)])
                # solve ODE
                m.time = t # time points
                m.options.IMODE = 5 # dynamic simulation
                m.options.NODES = 5 # collocation nodes
                #m.options.EV_TYPE = 2
                m.solve(disp=False)

                #return [x.value, y.value]
                return [x.value, u.value]
        
            Ca0 = [v0,v0]
            #Casol = odeint(my_diff, Ca0, t)
            Casol = my_diff(t)
            return Casol[0]


    elif functionName == 'Logistic' :
        def fitfunc(t, alpha, beta, v0):     
            def myode(dim, t):
                return alpha*dim * (1 - (dim/beta))     
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
                
    elif functionName == 'ClassicBertalanffy' :
        def fitfunc(t, alpha, beta, v0):     
            def myode(dim, t):
                return (alpha * (dim**2/3)) - (beta*dim)     # Classic Bertalanffy    
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]   
        
    elif functionName == 'GeneralBertalanffy' :
        def fitfunc(t, alpha, beta, lamda, v0):             
            def myode(dim, t):
                return alpha * (dim**lamda) - beta*dim     # General Bertalanffy   
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]        
    elif functionName == 'Gompertz' :
        def fitfunc(t, alpha, beta, v0):     
            def myode(dim, t):
                return  dim*(beta - alpha* np.log(dim))    # Gompertz
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
    elif functionName == 'GeneralGompertz' :
        def fitfunc(t, alpha, beta, lamda, v0):     
            def myode(dim, t):
                return (dim ** lamda)*(beta-(alpha*np.log(dim)) )   # General Gompertz
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
    return fitfunc

##############################################################################
    
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") 
    val = fitFunc(time, *parameterTuple)
    return np.sum((dimension - val) ** 2.0) 

##############################################################################
        
def generate_Initial_Parameters_genetic(ff, k, boundry, t, d, seed = 23, strategy = 'best1bin'):

    global fitFunc
    fitFunc= ff
    global time
    time = t
    global dimension
    dimension = d
    
    #boundry = [0,0.1]
    #strategy = 'best2bin'
    parameterBounds = []
    for i in range(k):
        parameterBounds.append(boundry)
    parameterBounds[0]=[0,0.01]
    parameterBounds[1]=[0.01,1]

    #parameterBounds[1]=[0,100] # DoubleExponential border for d
    #parameterBounds[2]=[0.01,1] # use for "r" in Function "GameTheory";
    #parameterBounds[1]=[0,0.1]
    #parameterBounds = [[10,100],[1,20],[0.01,5],[0.01,20],[0.01,30],[0.01,1],[0.01,100]] # for modKuznetsov GEKKO
    #parameterBounds = [[0.1,10],[0,1],[0,1],[0.01,5],[0.01,10],[0,1],[0.01,10]] # sigma, rho, nu, mu, delta, alpha
    #parameterBounds =([[10,0.01,0.01,0.01],[100,20,30,np.inf]])
    #parameterBounds = [[10,100],[0.01,20],[0.01,30],[0.01,1000]] # sigma, mu, delta
    #parameterBounds[2]=[0,0.1] # alpha, param für Modkuznetzsov with GEKKO three free params
    #parameterBounds[0]=[0.001,50] #sigma
    #parameterBounds[1]=[0.001,10]  #mu
    #parameterBounds[2]=[1,7]    #delta
    # parameterBounds[0]=[10,100] #sigma
    # parameterBounds[1]=[1,20]  #mu
    # parameterBounds[2]=[0.01,5]
    # parameterBounds[3]=[0.01,20]
    # parameterBounds[4]=[0.01,30]
    # parameterBounds[5]=[0.01,1]
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed = seed, strategy = strategy)
    
    return result.x  

##############################################################################    