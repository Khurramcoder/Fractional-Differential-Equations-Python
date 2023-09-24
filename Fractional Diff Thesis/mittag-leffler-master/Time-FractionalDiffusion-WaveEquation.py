import numpy as np 
from scipy.special import gamma, binom
""" Diethelm, K., J. M. Ford, N. J. Ford, and M. Weilbeer. "A comparison of 
backward differentiation approaches for ordinary and partial differential 
equations of fractional order." In Proceedings of the 1st IFAC Workshop on 
Fractional Differentiation and its Applications, Bourdeaux, France. 2004. 
"""
class Diffusion_Wave: 
    def __init__(self,params): 
        self.alpha = params["alpha"] 
        xmin = params["xmin"]; xmax = params["xmax"] 
        tmin = params["tmin"]; tmax = params["tmax"]
        dx = params["dx"]; dt = params["dt"] 
        x = np.arange(xmin,xmax,dx); N = len(x) 
        t = np.arange(tmin,tmax,dt); M = len(t)
        self.u = np.zeros((N,M)) 
        assert 0. <= self.alpha <= 2., "Please input 0 <= alpha <= 2" 
        assert len(params["ICs"]) == int(np.ceil(self.alpha)), "Please input the right number of initial conditions" 
        if int(np.ceil(self.alpha)) == 1: 
            self.u0 = params["ICs"][0] 
            self.u[:,0] = self.u0(x) 
        if int(np.ceil(self.alpha)) == 2: 
            self.u0,self.u1 = params["ICs"] 
            self.u[:,0] = self.u0(x) 
            self.u[:,1] = self.u1(x)*dt + self.u[:,0]
        self.g1,self.g2 = params["g"] 
        self.h1,self.h2 = params["h"] 
        self.r1,self.r2 = params["r"] 
        self.phi = params["phi"] 
        self.f = params["f"]
        self.x = x; self.t = t 
        self.N = N; self.M = M 
        self.dx = dx; self.dt = dt
    # Calculate the 
    def omega(self,j,k):
        # Use the Diethelm coefficients for 0 < alpha < 1 
        if 0. < self.alpha < 1.: 
            omg = 1. 
            if 1 <= j <= k-1: 
                omg = (j-1)**(1-self.alpha) - 2*j**(1-self.alpha) + (j+1)**(1-self.alpha) 
            if j == k and j >= 1: 
                omg = (k-1)**(1-self.alpha) + (1-self.alpha-k)*k**-self.alpha 
            if j == 0: 
                omg = 1.
                omg *= 1/gamma(2-self.alpha) 
                # Use the modified GL coefficients of 1 <= alpha <= 2 or alpha = 0 
        elif 1. <= self.alpha <= 2. or self.alpha == 0.: 
            omg = (-1)**j * binom(self.alpha,j) 
            if j > 0 and k-j == 0: 
                omg += j**(-self.alpha) / gamma(1-self.alpha) 
                omg -= (-1)**j * gamma(self.alpha)/(gamma(self.alpha-j)*gamma (j+1)) 
        return omg
    def dcoeff(self,k): 
        g1term = self.g1(self.t[k])/self.dx 
        g2term = self.g2(self.t[k])/self.dx 
        h1term = self.h1(self.t[k])*(-1)**1 
        h2term = self.h2(self.t[k])*(-1)**2 
        d1coeff = g1term + h1term \
            * ((1/self.dx**2) - self.omega(k,k)/(2*self.phi(min(self.x),k) *self.dt**self.alpha)) 
        d2coeff = g2term + h2term \
            * ((1/self.dx**2) - self.omega(k,k)/(2*self.phi(max(self.x),k) *self.dt**self.alpha)) 
        return d1coeff,d2coeff
    def ocoeff(self,k): 
        o1coeff = (-1)**(1+1)*self.h1(self.t[k])/self.dx**2 
        o2coeff = (-1)**(2+1)*self.h2(self.t[k])/self.dx**2 
        return o1coeff, o2coeff
    def rcoeff(self,k): 
        a = min(self.x); b = max(self.x) 
        r1term = self.r1(self.t[k])/self.dx 
        r2term = self.r2(self.t[k])/self.dx 
        h1term = ((-1)**(1+1) * self.h1(self.t[k])/(2*self.phi(a,self.t[k]))) 
        h2term = ((-1)**(2+1) * self.h2(self.t[k])/(2*self.phi(b,self.t[k]))) 
        sum1 = (1/self.t[k]**self.alpha)*np.sum([self.omega(mu,0)*self.u[0,k-mu] for mu in range(k+1)]) 
        sum2 = (1/self.t[k]**self.alpha)*np.sum([self.omega(mu,self.M)*self.u [-1,k-mu] for mu in range(k+1)]) 
        Dtalpha1 = ((1-self.alpha)/gamma(2-self.alpha))*self.t[k]*self.u0(self .x[0]) 
        Dtalpha2 = ((1-self.alpha)/gamma(2-self.alpha))*self.t[k]*self.u0(self .x[-1]) 
        if 1 < self.alpha <= 2 or self.alpha == 0: 
            Dtalpha1 += (1./gamma(2-self.alpha))*self.t[k]**(1-self.alpha)* self.u1(self.x[0]) 
            Dtalpha2 += (1./gamma(2-self.alpha))*self.t[k]**(1-self.alpha)* self.u1(self.x[-1]) 
            r1coeff = r1term + h1term*(self.f(a,self.t[k]) - sum1 + Dtalpha1) 
            r2coeff = r2term + h2term*(self.f(b,self.t[k]) - sum2 + Dtalpha2) 
        return r1coeff,r2coeff
    def numerical(self): 
        from scipy.linalg import solve 
        self.get_params()
        print("Numerical simulation size: %.2d x %.2d points" %(self.x.shape [0],self.t.shape[0])) 
        print("Evaluating numerical solution...") 
        # Generate central difference matrix 
        Cdiff = np.zeros((self.N,self.N)) 
        main_diag = np.diag_indices(self.N) 
        lower_diag = (main_diag[0][1:],(main_diag[1]-1)[1:]) 
        upper_diag = (main_diag[0][:-1],(main_diag[1]+1)[:-1]) 
        Cdiff[main_diag] = -2. 
        Cdiff[lower_diag] = 1. 
        Cdiff[upper_diag] = 1. 
        Cdiff *= 1./self.dx**2 
        for k in range(int(np.ceil(self.alpha)),self.M): 
            # Calculate and set d,o,r-coefficents for the central difference matrix 
            d1coeff,d2coeff = self.dcoeff(k) 
            o1coeff,o2coeff = self.ocoeff(k) 
            r1coeff,r2coeff = self.rcoeff(k) 
            Cdiff[0,0] = d1coeff/self.dx**2; Cdiff[-1,-1] = d2coeff/self.dx**2 
            Cdiff[0,1] = o1coeff/self.dx**2; Cdiff[-1,-2] = o2coeff/self.dx**2
            # Calculate the terms for the RHS vector 
            RHS1 = self.f(self.x,self.t[k]) 
            RHS1[0] = r1coeff; RHS1[-1] = r2coeff 
            RHS2 = self.u0(self.x)*(1-self.alpha) / gamma(2-self.alpha) / self .t[k]**self.alpha 
            RHS2[0] = 0.; RHS2[-1] = 0. 
            RHS3 = np.array([0.]+[np.sum([self.omega(mu,k)*self.u[j,k-mu] for mu in range(1,k+1)]) for j in range(1,self.N-1)]+[0.]) 
            RHS3 *= 1./self.dt**self.alpha 
            RHS = RHS1 + RHS2 - RHS3
            # Calculate the terms for the LHS matrix 
            LHS = np.multiply(self.phi(self.x,self.t[k]),Cdiff) 
            LHS[main_diag] += self.omega(0,k) / self.dt**self.alpha
            # Solve for the numerical solution 
            self.u[:,k] = solve(LHS,RHS) 
            print("DONE!") 
            return self.u 
    def plot(self): 
        import matplotlib.pyplot as plt 
        from mpl_toolkits.mplot3d import Axes3D 
        fig = plt.figure(figsize=(10,7)) 
        ax = fig.add_subplot(111,projection="3d") 
        X,T = np.meshgrid(self.x,self.t) 
        ax.plot_wireframe(X,T,self.u.T,alpha=0.5) 
        ax.set_title(r"Fractional Diffusion-Wave Equation ($\alpha$ = %.1f)" % self.alpha) 
        ax.set_xlabel("x") 
        ax.set_ylabel("t") 
        ax.set_zlabel("u(x,t)")
        plt.show()
    def get_params(self): 
        print("--------------------------------------------------") 
        print("TIME-FRACTIONAL DIFFUSION-WAVE EQUATION") 
        print("--------------------------------------------------") 
        print("D_t^alpha y(x,t) = phi(x,t)(d^2/dx^2)y(x,t)+f(x,t)") 
        print("--------------------------------------------------") 
        print("PARAMS:") 
        print("alpha = %.2f" %self.alpha) 
        print("x range = [%.2f,%.2f]" %(min(self.x),max(self.x))) 
        print("t range = [%.2f,%.2f]" %(min(self.t),max(self.t))) 
        print("dx = %.2f, dt = %.2f" %(self.dx, self.dt))
    def output(self): 
        return self.x, self.t, self.u
if __name__ == "__main__": 
    params = { 
        "alpha" : 1.6, 
        "tmin" : 0., 
        "tmax" : 10., 
        "dt" : 0.1, 
        "xmin" : 0.01,
        "xmax" : np.pi, 
        "dx" : 0.01, 
        "ICs" : [lambda x : np.sin(x),lambda x : 0.], # Define ICs with list of lambda functions 
        "g" : [lambda t : 1., lambda t : 1.], # Define g,h,r with list of lambda functions 
        "h" : [lambda t : t*0., lambda t : t*0.], 
        "r" : [lambda t : t*0., lambda t : t*0.], 
        "f" : lambda x,t : 0.*x*t, # Define f, phi with lambda functions 
        "phi" : lambda x,t : -1. 
    } 
DW = Diffusion_Wave(params) 
DW.numerical() 
DW.plot()        
                