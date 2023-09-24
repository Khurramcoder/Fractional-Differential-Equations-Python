import numpy as np 
from scipy.special import gamma, binom 
import matplotlib.pyplot as plt

class FTWE: 
    def __init__(self,params): 
        self.alpha = params["alpha"] 
        xmin = params["xmin"]; xmax = params["xmax"] 
        tmin = params["tmin"]; tmax = params["tmax"] 
        dx = params["dx"]; dt = params["dt"] 
        x = np.arange(xmin,xmax,dx); N = len(x) 
        t = np.arange(tmin,tmax,dt); M = len(t)
        self.u = np.zeros((N,M)) 
        self.nsrc = int(params["nsrc"]*N) 
        self.src = params["src"] 
        self.u[self.nsrc,0] = self.src(t[0]) 
        self.u[self.nsrc,1] = self.src(t[1])
        self.x = x; self.t = t 
        self.N = N; self.M = M 
        self.dx = dx; self.dt = dt 
        self.A0,self.A1,self.A2 = params["coeffs"] 
        self.mu0 = params["mu0"] 
        self.mur = params["mur"] 
        self.eps0 = params["eps0"]
        self.wj1 = np.array([(-1)**j * binom(self.alpha,j) for j in range(M)]) 
        self.wj2 = np.array([(-1)**j * binom(self.alpha+2,j) for j in range(M) ])
    def numerical(self): 
        from scipy.sparse import csr_matrix 
        self.get_params() 
        print("Evaluating numerical solution...") 
        Cdiff = np.zeros((self.N,self.N)) 
        main_diag = np.diag_indices(self.N) 
        lower_diag = (main_diag[0][1:],(main_diag[1]-1)[1:]) 
        upper_diag = (main_diag[0][:-1],(main_diag[1]+1)[:-1]) 
        Cdiff[upper_diag] = 1. 
        Cdiff[main_diag] = -2. 
        Cdiff[lower_diag] = 1. 
        Cdiff = csr_matrix(Cdiff) 
        for k in range(1,self.M-1): 
            term1 = Cdiff.dot(self.u[:,k]) 
            term1 *= self.A1 * (self.dt**2 / self.dx**2) / (self.mu0*self.mur* self.eps0)
            term2 = Cdiff.dot(self.u[:,k-1::-1]) 
            term2 = term2.dot(self.wj1[:k]) 
            term2 *= self.A2 * (self.dt**-self.alpha / self.dx**2) / (self.mu0 *self.mur*self.eps0)  
            term3 = np.matmul(self.wj2[:k],self.u[:,k-1::-1].T) 
            term3 *= self.A0 / self.dt**self.alpha
            self.u[self.nsrc,k] = self.src(self.t[k]) 
            self.u[:,k+1] = term1 + term2 - term3 + 2*self.u[:,k] - self.u[:,k -1] 
        print("DONE!") 
        return self.u 
    def plot(self): 
        import matplotlib.pyplot as plt 
        plt.figure(1,figsize=(10,7)) 
        plt.xlabel("x",fontsize=16) 
        plt.ylabel(r"$E_y$",fontsize=16) 
        plt.title("Four-Term 1D Fractional Wave Equation",fontsize=20) 
        plt.plot(self.x,self.u[:,0],label="t=%.3e" %self.t[0]) 
        plt.plot(self.x,self.u[:,int(self.N/4)],label="t=%.3e" %self.t[int( self.N/4)]) 
        plt.plot(self.x,self.u[:,int(self.N/2)],label="t=%.3e" %self.t[int( self.N/2)]) 
        plt.plot(self.x,self.u[:,int(3*self.N/4)],label="t=%.3e" %self.t[int (3*self.N/4)]) 
        plt.plot(self.x,self.u[:,-1],label="t=%.3e" %self.t[-1]) 
        plt.legend(loc=1,prop={"size":14}) 
        plt.show()
        
    def animate(self): 
        from matplotlib import rcParams 
        import matplotlib.animation as animation 
        print("Generating animation...") 
        rcParams['animation.convert_path'] = r'/usr/bin/convert' 
        fig = plt.figure(2,figsize=(10,7)) 
        ax = plt.axis([np.min(self.x),np.max(self.x),np.min(self.u),np.max( self.u)]) 
        curve, = plt.plot(self.x,self.u[:,0]) 
        def animate(time): 
            curve.set_data(self.x,self.u[:,time]) 
            plt.title(r"Four-Term 1D Fractional Wave Equation, $\alpha=%.2f$ ( timestep=%.1d)" %(self.alpha,time),fontsize=18) 
            return curve, 
        myAnimation = animation.FuncAnimation(fig,animate,frames=np.arange(0, len(self.t),int(len(self.t)/200)), interval=10, blit=True, repeat= False) 
        myAnimation.save("fractional_wave-"+str(self.alpha)+".gif", writer=" imagemagick",fps=30) 
        print("DONE!") 
    def get_params(self): 
        print("----------------------------------") 
        print("FOUR-TERM FRACTIONAL WAVE EQUATION") 
        print("----------------------------------") 
        print("PARAMS:" )
        print("alpha = %.2f" %self.alpha) 
        print("x range = [%.2f,%.2f]" %(min(self.x),max(self.x))) 
        print("t range = [%.2f,%.2f]" %(min(self.t),max(self.t))) 
        print("dx = %.2f, dt = %.2f" %(self.dx, self.dt)) 
        print("A0 = %.3e, A1 = %.3e, A2 = %.3e" %(self.A0,self.A1,self.A2)) 
        print("mu0 = %.3e, mur = %.3e, eps0 = %.3e" %(self.mu0,self.mur,self. eps0)) 
    def output(self): 
        return self.x,self.t,self.u 
if __name__ == "__main__": 
    freq = 1. # Frequency of the sine wave 
    tau = 5/freq # Duration of initial ramp function (5 periods) 
    params = { 
            "alpha" : 0.9, 
            "tmin" : 0., 
            "tmax" : 20., 
            "dt" : 0.01, 
            "xmin" : 0., 
            "xmax" : 20., 
            "dx" : 0.01, 
            "coeffs": [10e-4,0.1,10e-8], 
            "mu0" : 1., 
            "mur" : 1., 
            "eps0" : 1., 
            "nsrc" : .5, 
            "src" : lambda t : 0.5*(1+np.sin(np.pi/2/tau*(2*t-tau)))*np.sin(2*np .pi*freq*t) if t < tau else np.sin(2*np.pi*t) 
    } 
    FT = FTWE(params) 
    FT.numerical() 
    FT.plot() 
    FT.animate()       