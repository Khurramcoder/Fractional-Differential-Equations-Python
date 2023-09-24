import numpy as np 
from scipy.special import gamma, binom
from mittag_leffler import ml 
import time
class Relax_Osc: 
    """ 
    Initialize class given a dict "params" 
    """ 
    def __init__(self,params): 
        self.Acoeff = params["Acoeff"] 
        self.alpha  = params["alpha"] 
        self.f      = params["f"] 
        self.ICs    = params["ICs"] 
        self.m      = int(np.ceil(self.alpha)) 
        assert 0 < self.m <= 2, "ceil(alpha) must be 0 < ceil(alpha) <= 2" 
        assert self.m == len(self.ICs), "Must have ceil(alpha) number of initial conditions defined" 
        self.method  = params["method"]
        self.tmin    = params["tmin"] 
        self.tmax    = params["tmax"] 
        self.dt      = params["dt"] 
        self.tnum    = np.arange(self.tmin,self.tmax,self.dt) 
        self.ynum    = np.zeros(self.tnum.shape[0]) 
        self.wj = np.array([(-1)**j * binom(self.alpha,j) for j in range(len( self.tnum))])
        self.analytic = params["analytic"] 
        if self.analytic: 
            self.tanalytic = np.arange(self.tmin,self.tmax,(self.tmax-self. tmin)/1000.) 
            self.yanalytic = np.zeros(self.tanalytic.shape[0]) 
        self.exec_tanalyt = 0.; 
        self.exec_tnum = 0. 
        self.MAE = 0.; self.MSE = 0. 
        self.error = params["error"]
        self.write = params["write"]
    def analytical(self): 
        tanalyt0 = time.time() 
        A = self.Acoeff 
        start = min(self.tanalytic) 
        print("Analytical simulation size: %2d points" %self.tanalytic.shape [0]) 
        print("Evaluating analytical solution...") 
        # Define integrand for analytical solution 
        G2 = lambda t: t**(self.alpha-1) * ml(-A*t**self.alpha,self.alpha,self .alpha) 
        G2res = G2(self.tanalytic) 
        # Solve for analytical solution 
        f = self.f(self.tanalytic) 
        for i in range(1,len(self.tanalytic)): 
            self.yanalytic[i] = np.trapz(G2res[:i][::-1]*f[:i],self.tanalytic [:i]) 
        tanalyt1 = time.time() 
        self.exec_tanalyt = tanalyt1 - tanalyt0 
        print("DONE!") 
        return self.yanalytic
    
    
    def numerical(self): 
        tnum0 = time.time() 
        print("Numerical simulation size: %2d points" %self.tnum.shape[0]) 
        print("Evaluating numerical solution...") 
        T = len(self.tnum) 
        A = self.Acoeff 
        self.ynum[0] = self.ICs[0] 
        if self.m == 2: 
            self.ynum[1] = self.ICs[1] * self.dt + self.ynum[0] 
        if self.method == "GL": 
            for i in range(self.m,T): 
                term1 = A*self.dt**self.alpha * self.ynum[i-1] 
                term2 = np.dot(self.wj[1:i+1],self.ynum[i-1::-1]) # Main fractional derivative calculation 
                term3 = self.dt**self.alpha * self.f(self.tnum[i]) 
                self.ynum[i] = -term1 - term2 + term3 
        if self.method == "matrix": 
            from scipy.linalg import solve_triangular 
            # Generate LHS matrix 
            left = np.zeros((T-self.m,T-self.m)) 
            main_diag = np.diag_indices(T-self.m)
            # Lower triangular matrix with GL coefficients 
            for i in range(T-self.m): 
                left[i,:i+1] = self.wj[:i+1][::-1] 
            left *= self.dt**-self.alpha 
            left[main_diag] += A 
            # Generate RHS vector values 
            right = self.f(self.tnum[self.m:]) 
            # Solve for numerical solution 
            self.ynum[self.m:] = solve_triangular(left,right,lower=True, check_finite=True) 
        tnum1 = time.time() 
        self.exec_tnum = tnum1 - tnum0 
        print("DONE!") 
        return self.ynum
    def plot(self): 
        import matplotlib.pyplot as plt 
        plt.figure(figsize=(7,5)) 
        plt.plot(self.tnum,self.ynum,label="Numerical") 
        if self.analytic: 
            plt.plot(self.tanalytic,self.yanalytic,label="Analytical") 
        plt.xlabel("Time") 
        plt.ylabel("y(t)") 
        plt.title(r"Relaxation-Oscillation Model (h = %.3f)" %self.dt) 
        plt.xlim(min(self.tnum),max(self.tnum)) 
        plt.legend(loc=1) 
        plt.show() 
    def run(self): 
        self.get_params() 
        self.numerical() 
        print("Time to calculate numerical solution: %.6f" %self.exec_tnum) 
        print("--------")
        if self.analytic: 
            self.analytical() 
            print("Time to calculated analytical solution: %.6f" %self. exec_tanalyt) 
            if self.error: 
                self.get_error() 
                print("MAE: %.3e, MSE: %.3e" %(self.MAE,self.MSE)) 
        if (not self.analytic and self.error): 
            print("Cannot calculate error without analytical solution") 
        if self.write: 
            store_data = "RO_results_" + self.method + ".txt" 
            f = open(store_data,"a") 
            f.write("%.6f,%.6f,%.6f,%.6f,%.6e,%.6e\n" %(self.alpha,self.dt, self.exec_tnum,self.exec_tanalyt,self.MAE,self.MSE)) 
            f.close()
        print("================================================\n")
    def get_error(self): 
        """ 
        Since the number of points calculated for the analytical and numerical 
        solutions are not necessarily the same,
        we match the common time points of both analytical and numerical 
        solutions and average the error of all matched points. 
        """
        self.MAE = 0.; 
        self.MSE = 0. 
        dt_analyt = (self.tmax-self.tmin)/1000. # predetermined number of step size for analytical solution for sufficient resolution
        print("Calculating error...") 
        j = 0; count = 0  
        # Matching algorithm is faster if you for loop goes through fewer points!
        # Loop through analytical points and scan through numerical points only once for matches
        if dt_analyt >= self.dt:
            for i in range(len(self.tanalytic)):
                prev = j
                while not np.isclose(self.tnum[j], self.tanalytic[i]):
                    if self.tnum[j] > self.tanalytic[i] or j == len(self.tnum) -1:
                        j = prev
                        break
                    j += 1
                if j != prev:
                    self.MAE += np.abs(self.yanalytic[i] - self.ynum[j]) 
                    self.MSE += (self.yanalytic[i] - self.ynum[j])**2 
                    count += 1 

        # Loop through numerical points and scan through analytical points only once for matches
        if dt_analyt < self.dt:
            for i in range(len(self.tnum)):
                prev = j
                while not np.isclose(self.tnum[i], self.tanalytic[j]):
                    if self.tanalytic[j] > self.tnum[i] or j == len(self. tanalytic)-1:
                        j = prev
                        break
                    j += 1
                if j != prev:
                    self.MAE += np.abs(self.yanalytic[j] - self.ynum[i]) 
                    self.MSE += (self.yanalytic[j] - self.ynum[i])**2 
                    count += 1 
        # If no matches are made, set error to 0 to avoid divide by zero
        if count == 0:
            self.MAE = 0.; self.MSE = 0. 
        else: 
            self.MAE = self.MAE/count; self.MSE = self.MSE/count 
        print("DONE!")
        return self.MAE,self.MSE
    def output(self):
        return self.t, self.ynum, self.tanalytic, self.yanalytic
    def get_params(self):
        print("================================================")
        print("RELAXATION-OSCILLATION PROBLEM:")
        print("--------------------------------") 
        print("_0D_t^alpha y(t) + A*y(t) = f(t)") 
        print("--------------------------------") 
        print("PARAMS:") 
        print("alpha = %.2f" %self.alpha) 
        print("method = %s" %self.method) 
        print("t range = [%.2f,%2.f]" %(min(self.tnum),max(self.tnum))) 
        print(" y(0) = %.2f, y\'(0) = %.2f" %(self.ICs[0],self.ICs[1])) 
        print(" dt = %.3f" %self.dt) 
if __name__ == "__main__":
    """
    _0D_t^alpha y(t) + Ay(t) = f(t)
     
    """   
    params = {
        "Acoeff"   : 0.5, # A coefficient 
        "alpha"    : 1.5, # alpha-order derivative 
        "f"        : lambda t : np.heaviside(t,0), # Use lambda function to define your source function 
        "tmin"     : 0, # Start time 
        "tmax"     : 10, # End time 
        "dt"       : 0.001, # Time step 
        "ICs"      : [0,0], # Define initial conditions, must satsify existence (ceil(alpha) = # initial conditions) 
        "method"   : "matrix", # Pick numerical method, "GL" or "matrix" 
        "analytic" : True, # Solve for analytical solution? True or False 
        "error"    : True, # Evaluate both mean absolute error and mean square error (MAE and MSE) 
        "write"    : True # Write 6 values for: alpha, dt, numerical comp time, analytical comp time, MAE, MSE 
    }
    # How to initialize class and run code 
    # RO = Relax_Osc(params) 
    # RO.run() 
    # RO.plot() 
    # t,ynum,tanalytic,yanalytic = RO.output() 
    # How to do multiple runs 
    step = np.arange(0.001,0.011,0.001) 
    step = np.concatenate((step,np.arange(0.02,0.11,0.01))) 
    step = np.concatenate((step,np.arange(0.2,1.1,0.1))) 
    iterations = 1 
    for h in step:
        params["dt"] = h 
        RO = Relax_Osc(params) 
        for i in range(iterations): 
            RO.run() 
            RO.plot()              
                      














