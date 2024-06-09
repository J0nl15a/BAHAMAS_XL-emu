import pandas as pd
import numpy as np
import GPy
import pylab as pb
from data_loader import bahamasXLDMOData
from flamingo_data_loader import flamingoDMOData
import random
import pickle
import matplotlib.figure as fi

mlp = GPy.kern.MLP(9)
linear = GPy.kern.Linear(9)
polynomial = GPy.kern.Poly(9)
RBF = GPy.kern.RBF(9)
Bias = GPy.kern.Bias(9)
ARD = GPy.kern.RBF(9, ARD=True)

class gpyEmulator:

    def __init__(self, data, kern=GPy.kern.RBF(9), var_fix=False, single_k='nan', var=(1,10,.1), save=False, upload=False, ARD=False):
        
        self.kern = kern
        if ARD == True:
            self.kern = GPy.kern.RBF(9, ARD=True)

        self.data = data

        if single_k == 'nan':
            self.error = np.zeros((len(self.data.Y_test), self.data.Y_test.shape[0]))
        else:
            self.error = np.zeros((len(self.data.Y_test), 1))
            
        print(self.data.X_train.shape, self.data.Y_train.shape)
        if upload == True:
            self.model = pickle.load(open("gpy_model.pkl", "rb"))
        else:
            self.model = GPy.models.GPHeteroscedasticRegression(self.data.X_train, np.reshape(self.data.Y_train[:, single_k], (len(self.data.Y_train),1)) if isinstance(single_k, int) else self.data.Y_train, self.kern)
            if var_fix == True:
                if self.data.test_models in range(50):
                    index1 = 49
                    index2 = 99
                elif self.data.test_models in range(50, 100):
                    index1 = 50
                    index2 = 99
                else:
                    index1 = 50
                    index2 = 100
                self.model.het_Gauss.variance[:index1] = var[0]
                self.model.het_Gauss.variance[index1:index2] = var[1]
                self.model.het_Gauss.variance[index2:] = var[2]
                #if single_k == 'nan':
                    #self.kern.bias_variance[:index1] = bias_weight[0]
                    #self.kern.bias_variance[index1:index2] = bias_weight[1]
                    #self.kern.bias_variance[index2:] = bias_weight[2]
                #else:
                    #self.kern.bias_variance[:index1] = bias_weight[0][i]
                    #self.kern.bias_variance[index1:index2] = bias_weight[1][i]
                    #self.kern.bias_variance[index2:] = bias_weight[2][i]
                #self.kern.weight_variance[:index1] = var[0]
                #self.kern.weight_variance[index1:index2] = var[1]
                #self.kern.weight_variance[index2:] = var[2]
                #self.model.het_Gauss.variance[49:99].fix()
            elif var_fix == False:
                pass
            self.model.optimize()
        self.y = 10**(self.model._raw_predict(self.data.X_test.reshape(1, -1))[0])
        self.post_var = self.model._raw_predict(self.data.X_test.reshape(1, -1))[1][0]
        if save == True:
            pickle.dump(self.model, open("gpy_model.pkl", "wb"))
        if self.data.Y_test.ndim > 1:
            for w in range(len(self.data.Y_test)):
                self.error[w, :] = self.y/(self.data.Y_test[w, single_k] if isinstance(single_k, int) else self.data.Y_test[w, :])
        else:
            self.error = self.y/(self.data.Y_test[single_k] if isinstance(single_k, int) else self.data.Y_test)

        return    

    def plot(self, name, plot_file='png'):

        w,h=fi.figaspect(.3)
        fig, (ax1, ax2) = pb.subplots(1, 2, figsize=(w,h))
        ax1.plot(self.data.k_test, self.data.Y_test, color='b', label=('True P(k) for model_' + f"{self.data.test_models:03d}"))
        ax1.plot(self.data.k_test, self.y.reshape(-1,1), color='r', label=('Predicted P(k) for model_' + f"{self.data.test_models:03d}"))
        fig.suptitle('GPy test (kernel = '+f"{name}"+', '+("Intermediate" if self.data.test_models in range(50) else "Low" if self.data.test_models in range(50, 100) else "High")+' resolution)')
        ax1.set_xlabel('k (1/Mpc)')
        ax1.set_ylabel('P(k) (Mpc^3)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.set_xlim(right=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10)
              
        
        ax2.hlines(y=1.000, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='solid', alpha=0.5, label=None)
        ax2.hlines(y=0.990, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        ax2.hlines(y=1.010, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='dashed', alpha=0.5, label=None)
        ax2.hlines(y=0.950, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        ax2.hlines(y=1.050, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='dotted', alpha=0.5, label=None)
        ax2.plot(self.data.k_test, self.error.reshape(-1,1), label=('model_' + f"{self.data.test_models:03d}" + ' error'))
        ax2.set_xlabel('k (1/Mpc)')
        ax2.set_ylabel('Residual error')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.set_xlim(right=(int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10))
        if max(self.error.reshape(-1,1)) > 1.2 and min(self.error.reshape(-1,1)) < 0.8:
            ax2.set_ylim(top=1.2, bottom=0.8)
        elif min(self.error.reshape(-1,1)) < 0.8:
            ax2.set_ylim(bottom=0.8)
        elif max(self.error.reshape(-1,1)) > 1.2:
            ax2.set_ylim(top=1.2)
        fig.subplots_adjust(wspace=0.15)
        pb.savefig(f'./Plots/gpy_test_error_{name}.{plot_file}', dpi=1200)
        pb.clf()

        return

if __name__ == "__main__":

    test_model = random.randint(0, 149)
    bxl = bahamasXLDMOData(pk='nbk-rebin-std', lin='rebin', holdout=test_model)
    print(test_model)
    print(bxl.Y_test)
    bxl.extend_data(pad='emu')
    print(bxl.Y_test)
    
    weights = pd.read_csv('./BXL_data/weights.csv')
    emu = gpyEmulator(bxl, ARD=True)
    print(bxl.Y_train)
    print(emu.y)
    print(emu.error)
    emu.plot('ARD')
    quit()





    #test_model = random.randint(0, 149)
    b = pd.read_csv('./BXL_data/weights.csv')
    test_model = random.randint(100, 149)
    print('Model = ' + str(test_model))
    nbk_boost = BXL_DMO_Pk(test_model, 100, pk='nbk-rebin-std', lin='rebin', pad=True, holdout=True)
    flam = FLAMINGO_DMO_Pk(nbk_boost, 100, cutoff=(.01, 10), lin='camb')
    #nbk_boost.plot_k()
    print(nbk_boost.modes)
    #gpy_rbf = gpy_emulator(nbk_boost, RBF)
    #gpy_ard = gpy_emulator(nbk_boost, ARD*Bias, var_fix = True)
    #gpy_ard = gpy_emulator(flam, ARD*Bias, var_fix = True)
    #gpy_rbf_fix = gpy_emulator(nbk_boost, RBF, var_fix = True)
    pred = []
    err = []
    for i in range(len(flam.k_test)):
        HR_Bias = GPy.kern.Bias(9)
        IR_Bias = GPy.kern.Bias(9)
        LR_Bias = GPy.kern.Bias(9)
        HR_Bias.bias_variance = b['HR'][i]
        IR_Bias.bias_variance = b['IR'][i]
        LR_Bias.bias_variance = b['LR'][i]

        weights = np.linspace(.01, 10, len(flam.k_test))
        IR_weights = np.linspace(.7, 10, 7)

        #if i == 7:
            #gpy_ard_fix = gpy_emulator(flam, ARD*(HR_Bias+IR_Bias+LR_Bias), var_fix = True, single_k=i, var=(.01, weights[len(flam.k_test)-i-1], weights[i]))
        #elif i<7:
            #gpy_ard_fix = gpy_emulator(flam, ARD*(HR_Bias+IR_Bias+LR_Bias), var_fix = True, single_k=i, var=(IR_weights[6-i], weights[i], weights[len(flam.k_test)-i-1]))
        #else:
            #gpy_ard_fix = gpy_emulator(flam, ARD*(HR_Bias+IR_Bias+LR_Bias), var_fix = True, single_k=i, var=(IR_weights[i-8], weights[i], weights[len(flam.k_test)-i-1]))
        #pred.append(gpy_ard_fix.y[0])
        #err.append(gpy_ard_fix.error[0])
    gpy_ard_fix = gpy_emulator(nbk_boost, ARD*Bias, var_fix = True)
    w,h=fi.figaspect(.3)
    fig, (ax1, ax2) = pb.subplots(1, 2, figsize=(w,h))
    #ax1.plot(flam.k_test, flam.Y_test[0], color='b', label=('True P(k) for model_' + f"{test_model:03d}"))
    #ax1.plot(flam.k_test, gpy_ard.y[0], color='r', label=('Predicted P(k) for model_' + f"{test_model:03d}"))
    ax1.plot(nbk_boost.k_test, nbk_boost.Y_test, color='b', label=('True P(k) for model_' + f"{test_model:03d}"))
    ax1.plot(nbk_boost.k_test, gpy_ard_fix.y, color='r', label=('Predicted P(k) for model_' + f"{test_model:03d}"))
    
    fig.suptitle('HR model')
    ax1.set_xlabel('k (1/Mpc)')
    ax1.set_ylabel('P(k) (Mpc^3)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_xlim(right=10)
    
    
    ax2.hlines(y=1.000, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='solid', alpha=0.5, label=None)
    ax2.hlines(y=0.990, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label='1% error')
    ax2.hlines(y=1.010, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label=None)
    ax2.hlines(y=0.950, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label='5% error')
    ax2.hlines(y=1.050, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label=None)
    #ax2.plot(nbk_boost.k_test, gpy_ard.error[0], label=('model_' + f"{test_model:03d}" + ' error'))
    ax2.plot(nbk_boost.k_test, gpy_ard.error, label=('model_' + f"{test_model:03d}" + ' error'))
    ax2.set_xlabel('k (1/Mpc)')
    ax2.set_ylabel('Residual error')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.set_xlim(right=10)
    if max(gpy_ard.error) > 1.2 and min(gpy_ard.error) < 0.8:
        ax2.set_ylim(top=1.2, bottom=0.8)
    elif min(gpy_ard.error) < 0.8:
        ax2.set_ylim(bottom=0.8)
    elif max(gpy_ard.error) > 1.2:
        ax2.set_ylim(top=1.2)
    fig.subplots_adjust(wspace=0.15)
    pb.savefig(f'./Plots/gpy_test_error_ardxbias_HR.png', dpi=1200)
    pb.clf()
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
    
    #print(gpy_rbf.model.kern)
    #print(gpy_rbf.error)
    #gpy_rbf.plot(nbk_boost, 'RBF')
    print(gpy_ard.model.kern)
    print(gpy_ard.error)
    gpy_ard.plot(nbk_boost, 'RBF(ARD)', 'png')
    #gpy_rbf_fix.plot(nbk_boost, 'RBF fixed var')
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
