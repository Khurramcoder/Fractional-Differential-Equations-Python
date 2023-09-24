import numpy as np 
from scipy.special import gamma, binom

class FPDE:
### Initialize the y matrix (stores results of numerical calculations) 
#  Contains N rows for the space dimension, contains M columns for time dimension 
#  y[space-index,time-index] is how to access y-values at a particular location 
#  y[space-index,:] is how to access y-values at a given location in space 
#  y[:,time-index] is how to access y-values at a given moment in time
    def __init__(self,params): 
        # Set alpha 
        self.alpha = params["alpha"]
        # Set up space and time grids 
        xmin = params["xmin"]; xmax = params["xmax"] 
        tmin = params["tmin"]; tmax = params["tmax"] 
        dx = params["dx"]; dt = params["dt"] 
        x = np.arange(xmin,xmax,dx); N = len(x) 
        t = np.arange(tmin,tmax,dt); M = len(t)
        # Define initial condition function and set to y0 
        self.y = np.zeros((N,M)) 
        self.y0 = params["ICs"] 
        self.y[:,0] = self.y0(x)
        # Define coefficient functions 
        self.f = params["f"] 
        self.K = params["K"]
        # Define boundary conditions 
        self.r1,self.r2 = params["BCs"] 
        self.y[0,:] = self.r1(t); 
        self.y[-1,:] = self.r2(t)
        # Allocate memory for Hm vector and Gm matrix 
        self.Hm = np.zeros(x.shape[0]) 
        self.Gm = np.zeros((N,N))
        # Redefine to self 
        self.x = x; self.t = t 
        self.N = N; self.M = M 
        self.dx = dx; self.dt = dt
    ## Calculates the B coefficient values given mu- and m-values
    def Bcoeff(self,mu,m): 
        if mu == 0: 
            B = (m-1)**(1-self.alpha) - (m-1+self.alpha)*m**(-self.alpha) 
        else: 
            B = (m-mu+1)**(1-self.alpha) - 2*(m-mu)**(1-self.alpha) + (m-mu-1) **(1-self.alpha) 
        return B
    ## Calculate the Hm vector given m-values 
    def Hmvec(self,m): 
        self.Hm[0] = self.r1(self.t[m]) 
        self.Hm[-1] = self.r2(self.t[m]) 
        for i in range(1,self.N-1): 
            # iterate through x values 
            Hmi = (1-self.alpha)*m**(-self.alpha)*self.y0(self.x[i]) + gamma(2-self.alpha)*self.dt**self.alpha * self.f(self.x[i], self.t[m]) 
            for mu in range(m): 
                Hmi -= self.Bcoeff(mu,m) * self.y[i,mu] 
                self.Hm[i] = Hmi 
                return self.Hm
    ## Calculates the Gm matrix given m-values 
    def Gmat(self,m): 
        # Find indices of upper, main, and lower diagonals 
        main_diag = np.diag_indices(self.N); main,_ = main_diag 
        lower_diag = (main_diag[0][1:],(main_diag[1]-1)[1:]); lower,_ = lower_diag 
        upper_diag = (main_diag[0][:-1],(main_diag[1]+1)[:-1]); upper,_ = upper_diag 
        # Set values of the upper, main, and lower diagonals 
        self.Gm[main_diag] = -2*self.K(self.x[main],self.t[m]) 
        self.Gm[lower_diag] = self.K(self.x[lower],self.t[m]) 
        self.Gm[upper_diag] = self.K(self.x[upper],self.t[m])
        # First and last rows are all zero to account for boundary conditions 
        self.Gm[0,0] = 0.; self.Gm[0,1] = 0. 
        self.Gm[-1,-1] = 0.; self.Gm[-1,-2] = 0. 
        return self.Gm
    ## Solve numerically
    def numerical(self): 
        self.get_params() 
        from scipy.linalg import solve 
        print("Numerical simulation size: %.2d x %.2d points" %(self.x.shape [0],self.t.shape[0])) 
        print("Evaluating numerical solution...") 
        I = np.eye(self.N) 
        for m in range(1,self.M): 
            self.Hmvec(m) 
            self.Gmat(m) 
            LHS = I - ((gamma(2-self.alpha)*self.dt**self.alpha)/self.dx**2)* self.Gm 
            self.y[:,m] = solve(LHS,self.Hm) 
            print("DONE!") 
            return self.y
      ## Plot results on 3D plot
    def plot(self): 
        import matplotlib.pyplot as plt 
        from mpl_toolkits.mplot3d import Axes3D 
        fig = plt.figure(figsize=(10,7)) 
        ax = fig.add_subplot(111,projection="3d")
        # Create (N x M) matrices corresponding to space and time for each y(x ,t) 
        X,T = np.meshgrid(self.x,self.t) 
        # Plot results 
        ax.plot_wireframe(X, T, self.y.T, color='green')
        ax.set_title("Time-Fractional Partial Differential Equation") 
        ax.set_xlabel("x") 
        ax.set_ylabel("t") 
        ax.set_zlabel("y(x,t)") 
        plt.show()
    ## Print out input parameters
    def get_params(self): 
        print("----------------------------------------------") 
        print("TIME-FRACTIONAL PARTIAL DIFFERENTIAL EQUATIONS") 
        print("----------------------------------------------") 
        print("PARAMS:") 
        print("alpha = %.2f" %self.alpha) 
        print("x range = [%.2f,%.2f]" %(min(self.x),max(self.x))) 
        print("t range = [%.2f,%.2f]" %(min(self.t),max(self.t))) 
        print("dx = %.2f, dt = %.2f" %(self.dx, self.dt))
    ## Return the result values 
    def output(self): 
        return self.x, self.t, self.y  
if __name__ == "__main__": 
    params = { 
              "alpha": 0.5, 
              "tmin" : 0., 
              "tmax" : 10., 
              "dt" : 0.1, 
              "xmin" : 0.,
              "xmax" : np.pi, 
              "dx" : 0.01, 
              "BCs" : [lambda t: 0., lambda t: 0.], #[lambda t: np.exp(-t) , lambda t: -np.exp(-t)], 
              "ICs" : lambda x: np.sin(x), 
              "f" : lambda x,t : 0., #lambda x,t: np.cos(x) * np.cos(t), 
              "K" : lambda x,t : 1. #lambda x,t: 2 + (1 + x*(2*np.pi - x))/(5+5*t ) 
    } 
    FPDE1 = FPDE(params) 
    FPDE1.numerical() 
    FPDE1.plot()      
    
    
          