import numpy as np
import GPy
import pylab as pb
from data_loader import BXL_DMO_Pk
import random
import pickle

mlp = GPy.kern.MLP(9)
linear = GPy.kern.Linear(9)
polynomial = GPy.kern.Poly(9)
RBF = GPy.kern.RBF(9)
Bias = GPy.kern.Bias(9)
ARD = GPy.kern.RBF(9, ARD=True)

class gpy_emulator:

    def __init__(self, P_k_data, kern, var_fix=False, single_k='nan', var=(1,10,.1), save=False, upload=False):
        self.test_models = P_k_data.test_models
        self.kern = kern

        if upload == True:
            self.model = pickle.load(open("gpy_model.pkl", "rb"))
        else:
            self.model = GPy.models.GPHeteroscedasticRegression(P_k_data.X_train, (np.reshape(P_k_data.Y_train[:, single_k], (len(P_k_data.Y_train),1)) if isinstance(single_k, int) else P_k_data.Y_train), self.kern)
            #self.model = GPy.models.GPHeteroscedasticRegression(P_k_data.X_train, np.reshape(P_k_data.Y_train[:, single_k], (149,1)), self.kern)
            if var_fix == True:
                if self.test_models in range(50):
                    index1 = 49
                    index2 = 99
                elif self.test_models in range(50, 100):
                    index1 = 50
                    index2 = 99
                else:
                    index1 = 50
                    index2 = 100
                self.model.het_Gauss.variance[:index1] = var[0]
                self.model.het_Gauss.variance[index1:index2] = var[1]
                self.model.het_Gauss.variance[index2:] = var[2]
                #self.model.het_Gauss.variance[49:99].fix()
            elif var_fix == False:
                pass
            self.model.optimize()
        self.y = 10**(self.model._raw_predict(P_k_data.X_test.reshape(1, -1))[0])
        if save == True:
            pickle.dump(self.model, open("gpy_model.pkl", "wb"))
        if P_k_data.Y_test.ndim > 1:
            for w in range(len(P_k_data.Y_test)):
                self.error = self.y/(P_k_data.Y_test[w, single_k] if isinstance(single_k, int) else P_k_data.Y_test[w, :])
        else:
            self.error = self.y/(P_k_data.Y_test[single_k] if isinstance(single_k, int) else P_k_data.Y_test)

        return    

    def plot(self, P_k_data, name):
        pb.plot(P_k_data.k_test, P_k_data.Y_test, color='b', label=('True Non-linear P(k) for model_' + f"{self.test_models:03d}"))
        pb.plot(P_k_data.k_test, self.y.reshape(-1,1), color='r', label=('Predicted Non-linear P(k) for model_' + f"{self.test_models:03d}"))

        pb.title('GPy test (kernel = ' + f"{name}" + ')')
        pb.xlabel('k (1/Mpc)')
        pb.ylabel('P(k) (Mpc^3)')
        pb.xscale('log')
        pb.yscale('log')
        pb.legend()
        pb.savefig(f'./Plots/gpy_test_{name}.pdf', dpi=800)
        pb.clf()

        
        pb.hlines(y=1.000, xmin=-1, xmax=max(P_k_data.k_test)+2, color='k', linestyles='solid', alpha=0.5, label=None)
        pb.hlines(y=0.990, xmin=-1, xmax=max(P_k_data.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        pb.hlines(y=1.010, xmin=-1, xmax=max(P_k_data.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label=None)
        pb.hlines(y=0.950, xmin=-1, xmax=max(P_k_data.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        pb.hlines(y=1.050, xmin=-1, xmax=max(P_k_data.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label=None)
        pb.plot(P_k_data.k_test, self.error.reshape(-1,1), label=('Residual error for model_' + f"{self.test_models:03d}"))
        pb.title('GPy error test (kernel = ' + f"{name}" + ')')
        pb.xlabel('k (1/Mpc)')
        pb.ylabel('Residual error')
        pb.xscale('log')
        pb.legend()
        pb.xlim(right=10)
        pb.savefig(f'./Plots/gpy_test_error_{name}.pdf', dpi=800)
        pb.clf()

        return

if __name__ == "__main__":
    #test_model = random.randint(0, 149)
    test_model = random.randint(0, 49)
    nbk_boost = BXL_DMO_Pk(test_model, 100, pk='nbk-rebin-std', lin='rebin', pad=True)
    nbk_boost.plot_k()
    print(nbk_boost.modes)
    gpy_rbf = gpy_emulator(nbk_boost, RBF)
    gpy_ard = gpy_emulator(nbk_boost, ARD)
    gpy_rbf_fix = gpy_emulator(nbk_boost, RBF, var_fix = True)
    gpy_ard_fix = gpy_emulator(nbk_boost, ARD, var_fix = True)
    #gpy_bias = gpy_emulator(nbk_boost, Bias)
    #gpy_poly = gpy_emulator(nbk_boost, polynomial)
    #gpy_lin = gpy_emulator(nbk_boost, linear)
    #gpy_mlp = gpy_emulator(nbk_boost, mlp)

    #gpy_rbf_bias_a = gpy_emulator(nbk_boost, RBF+Bias)
    #gpy_rbf_mlp_a = gpy_emulator(nbk_boost, RBF+mlp)
    #gpy_rbf_poly_a = gpy_emulator(nbk_boost, RBF+polynomial)
    #gpy_rbf_lin_a = gpy_emulator(nbk_boost, RBF+linear)

    #gpy_rbf_bias_m = gpy_emulator(nbk_boost, RBF*Bias)
    #gpy_rbf_mlp_m = gpy_emulator(nbk_boost, RBF*mlp)
    #gpy_rbf_poly_m = gpy_emulator(nbk_boost, RBF*polynomial)
    #gpy_rbf_lin_m = gpy_emulator(nbk_boost, RBF*linear)
    
    print(gpy_rbf.model.kern)
    print(gpy_rbf.error)
    gpy_rbf.plot(nbk_boost, 'RBF')
    print(gpy_ard.model.kern)
    print(gpy_ard.error)
    gpy_ard.plot(nbk_boost, 'RBF(ARD)')
    gpy_rbf_fix.plot(nbk_boost, 'RBF fixed var')
    gpy_ard_fix.plot(nbk_boost, 'RBF(ARD) fixed var')
    #gpy_bias.plot(nbk_boost, 'Bias')
    #gpy_poly.plot(nbk_boost, 'Polynomial')
    #gpy_lin.plot(nbk_boost, 'Linear')
    #gpy_mlp.plot(nbk_boost, 'MLP')
    
    #gpy_rbf_bias_a.plot(nbk_boost, 'RBF+Bias')
    #gpy_rbf_mlp_a.plot(nbk_boost, 'RBF+MLP')
    #gpy_rbf_poly_a.plot(nbk_boost, 'RBF+Polynomial')
    #gpy_rbf_lin_a.plot(nbk_boost, 'RBF+Linear')

    #gpy_rbf_bias_m.plot(nbk_boost, 'RBFxBias')
    #gpy_rbf_mlp_m.plot(nbk_boost, 'RBFxMLP')
    #gpy_rbf_poly_m.plot(nbk_boost, 'RBFxPolynomial')
    #gpy_rbf_lin_m.plot(nbk_boost, 'RBFxLinear')

    #mean_error0 = []
    #for i in range(5):
        #test_model = random.randint(0, 149)
        #boost = BXL_DMO_Pk(test_model, 100, pk='nbk-rebin', lin='rebin')
        #grbf = gpy_emulator(boost, RBF)
        #gbias = gpy_emulator(nbk_boost, Bias)
        #gpoly = gpy_emulator(nbk_boost, polynomial)
        #glin = gpy_emulator(nbk_boost, linear)
        #gmlp = gpy_emulator(nbk_boost, mlp)
        
        #grbf_bias_a = gpy_emulator(nbk_boost, RBF+Bias)
        #grbf_mlp_a = gpy_emulator(nbk_boost, RBF+mlp)
        #grbf_poly_a = gpy_emulator(nbk_boost, RBF+polynomial)
        #grbf_lin_a = gpy_emulator(nbk_boost, RBF+linear)
        
        #grbf_bias_m = gpy_emulator(nbk_boost, RBF*Bias)
        #grbf_mlp_m = gpy_emulator(nbk_boost, RBF*mlp)
        #grbf_poly_m = gpy_emulator(nbk_boost, RBF*polynomial)
        #grbf_lin_m = gpy_emulator(nbk_boost, RBF*linear)
        
        #mean_error0.append(grbf.error.reshape(-1,1))
        
        
    #pb.hlines(y=1.000, xmin=-1, xmax=nbk_boost.high_k_cut+1, color='k', linestyles='solid', alpha=0.5, label=None)
    #pb.hlines(y=0.990, xmin=-1, xmax=nbk_boost.high_k_cut+1, color='k', linestyles='dashed', alpha=0.5, label='1% error')
    #pb.hlines(y=1.010, xmin=-1, xmax=nbk_boost.high_k_cut+1, color='k', linestyles='dashed', alpha=0.5, label=None)
    #pb.hlines(y=0.950, xmin=-1, xmax=nbk_boost.high_k_cut+1, color='k', linestyles='dotted', alpha=0.5, label='5% error')
    #pb.hlines(y=1.050, xmin=-1, xmax=nbk_boost.high_k_cut+1, color='k', linestyles='dotted', alpha=0.5, label=None)
    #pb.plot(nbk_boost.k_test, gpy_rbf.error.reshape(-1,1), label=('kernel = RBF'), linestyle='solid', color='b')
    #pb.plot(nbk_boost.k_test, gpy_lin.error.reshape(-1,1), label=('kernel = Linear'), linestyle='solid', color='g')
    #pb.plot(nbk_boost.k_test, gpy_poly.error.reshape(-1,1), label=('kernel = Polynomial'), linestyle='solid', color='r')
    #pb.plot(nbk_boost.k_test, gpy_mlp.error.reshape(-1,1), label=('kernel = MLP'), linestyle='solid', color='m')
    #pb.plot(nbk_boost.k_test, gpy_bias.error.reshape(-1,1), label=('kernel = Bias'), linestyle='solid', color='c')
    #pb.plot(nbk_boost.k_test, gpy_rbf_bias_a.error.reshape(-1,1), label=('kernel = RBF+Bias'), linestyle='dashed', color='c')
    #pb.plot(nbk_boost.k_test, gpy_rbf_lin_a.error.reshape(-1,1), label=('kernel = RBF+Linear'), linestyle='dashed', color='g')
    #pb.plot(nbk_boost.k_test, gpy_rbf_poly_a.error.reshape(-1,1), label=('kernel = RBF+Polynomial'), linestyle='dashed', color='r')
    #pb.plot(nbk_boost.k_test, gpy_rbf_mlp_a.error.reshape(-1,1), label=('kernel = RBF+MLP'), linestyle='dashed', color='m')
    #pb.plot(nbk_boost.k_test, gpy_rbf_bias_m.error.reshape(-1,1), label=('kernel = RBF*Bias'), linestyle='dotted', color='c')
    #pb.plot(nbk_boost.k_test, gpy_rbf_lin_m.error.reshape(-1,1), label=('kernel = RBF*Linear'), linestyle='dotted', color='g')
    #pb.plot(nbk_boost.k_test, gpy_rbf_poly_m.error.reshape(-1,1), label=('kernel = RBF*Polynomial'), linestyle='dotted', color='r')
    #pb.plot(nbk_boost.k_test, gpy_rbf_mlp_m.error.reshape(-1,1), label=('kernel = RBF*MLP'), linestyle='dotted', color='m')
    #pb.title('GPy error test (model_' + f"{test_model}" + ')' )
    #pb.xlabel('k (1/Mpc)')
    #pb.ylabel('Residual error')
    #pb.xscale('log')
    #pb.legend(fontsize=5, ncols=4)
    #pb.savefig(f'./Plots/gpy_test_error_comparison.pdf', dpi=800)
    #pb.clf()
