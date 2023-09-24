import numpy as np 
from scipy.special import gamma, binom 
import time


class Bagley_Torvik: 
    def __init__(self,params): 
        self.A,self.B,self.C = params["coeffs"] 
        self.alpha = params["alpha"] 
        self.f = params["f"] 
        self.ICs = params["ICs"] 
        self.m = 2 
        assert self.m == len(self.ICs), "Must have %1d number of initial conditions defined" %self.m 
        self.method = params["method"]
        tmin = params["tmin"] 
        tmax = params["tmax"] 
        self.dt = params["dt"] 
        self.tnum = np.arange(tmin,tmax,self.dt) 
        self.ynum = np.zeros(self.tnum.shape[0]) 
        self.wj = np.array([(-1)**j * binom(self.alpha,j) for j in range(len( self.tnum))])
        self.analytic = params["analytic"] 
        if self.analytic: 
            self.tanalytic = np.arange(tmin,tmax,(tmax-tmin)/1000.) # 1000 steps is recommended for best accuracy and time efficiency 
            self.yanalytic = np.zeros(self.tanalytic.shape[0]) 
        self.tmin = tmin; self.tmax = tmax 
        self.exec_tanalyt = 0.; self.exec_tnum = 0. 
        self.MAE = 0; self.MSE = 0. 
        self.error = params["error"] 
        self.write = params["write"]
    def analytical(self): 
        tanalyt0 = time.time() 
        # Number of iterations for analytical solution to converge, adjust if needed for accuracy and/or time efficiency 
        iterations = 50 
        G3res = np.zeros(self.tanalytic.shape[0]) 
        print("Analytical simulation size: %2d points" %self.tanalytic.shape [0]) 
        print("Evaluating analytical solution...") 
        # Find Green’s function values at each point and store 
        for i in range(len(self.tanalytic)): 
            G3 = 0 
            for k in range(iterations): 
                mlk = 0 
                for j in range(iterations): 
                    mlk += (gamma(j+k+1) * ((-self.B/self.A)*self.tanalytic[i 
                        ]**0.5)**j) / \
                            (gamma(j+1) * gamma(0.5*j + 0.5*k + 2 + (3.*k/2))) 
                G3 += (1/self.A) * ((-1)**k)/gamma(k+1) * (self.C/self.A)**k *self.tanalytic[i]**(2*k+1) * mlk 
                G3res[i] = G3 
        # Get values for source function, f 
        f = self.f(self.tanalytic) 
        # Solve the convolution integral
        for i in range(1,len(self.tanalytic)): 
            self.yanalytic[i] = np.trapz(G3res[:i][::-1]*f[:i], self.tanalytic [:i]) 
        tanalyt1 = time.time() 
        self.exec_tanalyt = tanalyt1 - tanalyt0 
        print("DONE!") 
        return self.yanalytic
    def numerical(self): 
        print("Numerical simulation size: %2d points" %len(self.tnum)) 
        print("Evaluating numerical solution...") 
        tnum0 = time.time() 
        T = len(self.tnum) 
        A = self.A; B = self.B; C = self.C 
        self.ynum[0] = self.ICs[0] 
        self.ynum[1] = self.ICs[1] * self.dt + self.ynum[0] 
        if self.method == "GL": 
            for i in range(self.m,T): 
                term1 = (self.dt**2) * (self.f(self.tnum[i]) - C*self.ynum[i -1]) 
                term2 = A * (2*self.ynum[i-1] - self.ynum[i-2]) 
                term3 = B * (self.dt**0.5 * np.dot(self.wj[1:i],self.ynum[1:i ][::-1])) # Main fractional derivative calculation 
                numer = term1 + term2 - term3 
                denom = A + B*(self.dt**0.5) 
                self.ynum[i] = numer/denom
        if self.method == "matrix": 
            from scipy.linalg import solve_triangular 
            Tm = T-self.m 
            left = np.zeros((Tm,Tm)) 
            # Generate lower triangular matrix with GL coefficients for fractional derivative
            for i in range(T-self.m): 
                left[i,:i+1] = self.wj[:i+1][::-1] 
            left *= B * self.dt**-self.alpha 
             # Add backward difference matrix for second-order derivative
            main_diag = np.diag_indices(Tm) 
            lower1_diag = (main_diag[0][1:],(main_diag[1]-1)[1:]) 
            lower2_diag = (main_diag[0][2:],(main_diag[1]-2)[2:]) 
            left[main_diag] += 1. * A * self.dt**-2. 
            left[lower1_diag] += -2. * A * self.dt**-2. 
            left[lower2_diag] += 1. * A * self.dt**-2.
            #  Add identity matrix for term with no derivative
            left[main_diag] += C
            # Generate RHS vector 
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
        plt.title("Bagley-Torvik Model (h = %.3f)" %self.dt) 
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
            store_data = "BT_results_" + self.method + ".txt" 
            f = open(store_data,"a") 
            f.write("%.6f,%.6f,%.6f,%.6f,%.6e,%.6e\n" %(self.alpha,self.dt, self.exec_tnum,self.exec_tanalyt,self.MAE,self.MSE)) 
            f.close() 
        print("================================================\n")
    def get_error(self): 
        """ 
        Since the number of points calculated for the analytical and numerical solutions
        are not necessarily the same,
        we match the common time points of both analytical and numerical 
        solutions and average the error of all matched points. 
        """   
        self.MAE = 0.; self.MSE = 0. 
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
    def get_params(self): 
        print("BAGLEY-TORVIK PROBLEM:") 
        print("---------------------------------------------") 
        print("A*y\"(t) + B*_0D_t^(3/2) y(t) + C*y(t) = f(t)") 
        print("---------------------------------------------") 
        print("PARAMS:") 
        print("alpha = %.2f" %self.alpha) 
        print("method = %s" %self.method) 
        print("t range = [%.2f,%2.f]" %(min(self.tnum),max(self.tnum))) 
        print(" y(0) = %.2f, y\'(0) = %.2f" %(self.ICs[0],self.ICs[1])) 
        print(" dt = %.3f" %self.dt)
    def output(self): 
        return self.tnum, self.ynum, self.tanalytic, self.yanalytic
if __name__ == "__main__":
    params = { 
               "coeffs" : [1.,0.5,0.5], # A, B, C coefficients 
               "alpha" : 1.5, # For Bagley-Torvik, this should always be 1.5 
               "f" : lambda t : 8. * (t < 1), # Use lambda function to define your source function 
               "tmin" : 0, # Start time 
               "tmax" : 30, # End time 
               "dt" : 0.01, # Time step (for matrix, dt = 0.001 for (0,30) overloads scipy.solve_triangular) 
               "ICs" : [0,0], # Initial conditions, must have 2 defined for BT problem 
               "method" : "matrix", # Pick numerical method 
               "analytic" : True, # Solve for analytical solution? True or False 
               "error" : True, # Evaluate both mean absolute error and mean square error (MAE and MSE) 
               "write" : True # Write 6 values: alpha, dt, numerical comp time, analytical comp time, MAE, MSE 
    }         
    # How to initialize class and run code
#   BT = Bagley_Torvik(params) 
#   BT.run() 
#   BT.plot()
    # How to run multiple instances
    step = np.arange(0.005,0.011,0.001) 
    step = np.concatenate((step,np.arange(0.02,0.11,0.01))) 
    step = np.concatenate((step,np.arange(0.2,1.1,0.1))) 
    iterations = 1
    for h in step: 
        params["dt"] = h 
        BT = Bagley_Torvik(params) 
        for i in range(iterations): 
            BT.run() 
            BT.plot()                        
                                 
                                               
