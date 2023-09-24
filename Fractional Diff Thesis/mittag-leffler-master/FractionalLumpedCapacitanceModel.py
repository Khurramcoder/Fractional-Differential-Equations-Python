import numpy as np 
from scipy.special import gamma, binom 
from mittag_leffler import ml

class Frac_Lumped:
    def __init__(self,params): 
        data = np.loadtxt(params["exp"]) 
        ambient = np.where(data[:,1] < 30)[0][-1] 
        data = data[ambient:] 
        print("Ambient temperature: %.3f C" %min(data[:,1])) 
        data[:,0] -= min(data[:,0]) 
        data[:,1] -= min(data[:,1]) 
        sing = np.where(data[:,1] == max(data[:,1]))[0][0] 
        self.rise = np.concatenate((data[:sing+1,0][:,None],data[:sing+1,1][:, None]),axis=1) 
        self.fall = np.concatenate((data[sing:,0][:,None],data[sing:,1][:,None ]),axis=1) 
        self.fall -= min(self.fall[:,0]) 
        self.data = data 
        self.sing = sing 
        self.Eg = params["Eg"] 
        self.analytic = params["analytic"]
        bioheat = np.genfromtxt(params["bioheat"],delimiter=",") 
        bioheat[:,1] -= bioheat[:,1][0] 
        self.bioheat = bioheat 
        self.analyt_heat_coeff = params["analyt_heat_coeff"] 
        self.analyt_cool_coeff = params["analyt_cool_coeff"] 
        self.num_heat_coeff = params["num_heat_coeff"] 
        self.num_cool_coeff = params["num_cool_coeff"]
    def heating(self,t): 
        a,b,c,alpha = self.analyt_heat_coeff 
        return b * t * ml(-a*t**alpha,alpha,2)
    def cooling(self,t): 
        a,b,c,alpha = self.analyt_cool_coeff 
        return c * t**(alpha-1) * ml(-a*t**alpha,alpha,alpha)  
    def analytical_fit(self): 
        shift = self.rise[:,0][self.sing] 
        self.tfit = np.concatenate((self.rise[:,0],self.fall[:,0]+shift),axis =0) 
        lumpy = np.concatenate((self.rise[:,1],self.fall[:,1]),axis=0) 
        self.analyt_fit = np.concatenate((self.heating(self.rise[:,0]),self. cooling(self.fall[:,0])),axis=0) 
        return self.analyt_fit 
    def numerical_fit(self): 
        peak = self.data[:,0][self.sing] 
        start = 0.; end = 0.1; dt = 0.0001 
        self.tnum = np.arange(start,end,dt)
        A1,_,_,alpha1 = self.num_heat_coeff 
        A2,_,_,alpha2 = self.num_cool_coeff 
        laser = lambda t : self.Eg*t**(1-alpha1)/gamma(2-alpha1) * (0 < t < peak) 
        wj = [] 
        for j in range(1,len(self.tnum)): 
            if self.tnum[j] <= peak: 
                wj.append(((-1)**j)*binom(alpha1,j)) 
            elif self.tnum[j] > peak: 
                wj.append(((-1)**j)*binom(alpha2,j)) 
            wj = np.array(wj) 
            self.ynum = np.zeros(self.tnum.shape[0]) 
            self.ynum[0] = self.data[0][1]
        for i in range(1,len(self.tnum)): 
            if self.tnum[i] <= peak: 
                term1 = -A1*(dt**alpha1)*self.ynum[i-1] 
                term2 = -np.dot(wj[:i],self.ynum[i-1::-1]) 
                term3 = dt**alpha1 * laser(self.tnum[i]) 
            elif self.tnum[i] > peak: 
                term1 = -A2*(dt**alpha2)*self.ynum[i-1] 
                term2 = -np.dot(wj[:i],self.ynum[i-1::-1]) 
                term3 = dt**alpha2 * laser(self.tnum[i]) 
            self.ynum[i] = term1 + term2 + term3 
        return self.ynum 
    def plot(self): 
        import matplotlib.pyplot as plt 
        plt.figure(figsize=(10,7)) 
        plt.scatter(self.data[:,0],self.data[:,1],marker="+",label=" Experimental",color="k",alpha=0.5) 
        plt.plot(self.tnum,self.ynum,label="Fractional Model") 
        if self.analytic: 
            plt.plot(self.tfit,self.analyt_fit,label="Analytical Fit") 
            plt.plot(self.bioheat[:,0],self.bioheat[:,1],label="Pennes Bioheat") 
            plt.xlim(0,0.1) 
            plt.ylim(0,) 
            plt.xlabel("Time (s)") 
            plt.ylabel(r"dT ($^{\circ}$C)") 
            plt.title("Fractional Lumped Capacitance Model") 
            plt.legend(loc=1) 
            plt.show()
    def run(self): 
        self.get_params() 
        self.numerical_fit() 
        if self.analytic: 
            self.analytical_fit()
    def output(self): 
        return self.tnum, self.ynum, self.tfit, self.analyt_fit
    def get_params(self): 
        print("FRACTIONAL LUMPED CAPACITANCE MODEL") 
        print("-----------------------------------") 
        print("PARAMS:") 
        print("Numerical heating curve coeffs:") 
        print("a = %.3f, b = %.3f, c = %.3f, alpha = %.3f" %tuple(self. num_heat_coeff)) 
        print("Numerical cooling curve coeffs:") 
        print("a = %.3f, b = %.3f, c = %.3f, alpha = %.3f" %tuple(self. num_cool_coeff)) 
        print("Analytical heating curve coeffs:") 
        print("a = %.3f, b = %.3f, c = %.3f, alpha = %.3f" %tuple(self. analyt_heat_coeff)) 
        print("Analytical cooling curve coeffs:") 
        print("a = %.3f, b = %.3f, c = %.3f, alpha = %.3f" %tuple(self. analyt_cool_coeff)) 
        print("Eg = %.3f" %self.Eg )
if __name__ == "__main__":
    params = { 
            "exp" : "0.01s,1.037cm,1p,2780W.txt", 
            "bioheat" : "0.01s,1cm,2780W-SESE.csv", 
            "analyt_heat_coeff" : [10.9,3600,0,0.841], 
            "analyt_cool_coeff" :[2.625,0,17.5,0.906], 
            "num_heat_coeff" : [10.9,3600,0,0.841], 
            "num_cool_coeff" : [0.1,0,17.5,0.855], 
            "Eg" : 2780*1.28, "analytic" : True 
    } 
    FLC = Frac_Lumped(params) 
    FLC.run() 
    FLC.plot()                      













