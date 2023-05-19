import numpy as np
import matplotlib.pyplot as plt
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
import sepia.SepiaPlot as SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
from data_loader import BXL_DMO_Pk
import random

#Some setup parameters for the model that I don't really touch now
pc = 3
save = True

class sepia_emulator:

    #def __init__(self, X_train, Y_train, X_test, Y_test, test_models, k_ind, method = 'sim-only', pc=pc, samp=50, step=20, mcmc=1000):
    def __init__(self, P_k_data, method = 'sim-only', pc=0.99, samp=50, step=20, mcmc=1000):
        self.test_models = P_k_data.test_models
        
        #SepiaData model building
        if method == 'sim-only':
            self.data = SepiaData(x_sim = P_k_data.X_train, y_sim = P_k_data.Y_train, y_ind_sim = P_k_data.k_ind)
        elif method == 'rag-obs':
            if self.test_models in range(50):
                self.data = SepiaData(y_obs = P_k_data.Y_train[50:, :], y_ind_obs = P_k_data.k_ind, y_sim = P_k_data.Y_train[:50, :], t_sim = P_k_data.X_train[:50, :], y_ind_sim = P_k_data.k_ind)
            elif self.test_models in range(50, 100):
                self.data = SepiaData(y_obs = np.concatenate((P_k_data.Y_train[:50, :], P_k_data.Y_train[100:, :])), y_ind_obs = P_k_data.k_ind, y_sim = P_k_data.Y_train[50:100, :], t_sim = P_k_data.X_train[50:100, :], y_ind_sim = P_k_data.k_ind)
            elif self.test_models in range(100, 150):
                self.data = SepiaData(y_obs = P_k_data.Y_train[:100, :], y_ind_obs = P_k_data.k_ind, y_sim = P_k_data.Y_train[100:, :], t_sim = P_k_data.X_train[100:, :], y_ind_sim = P_k_data.k_ind)

        self.data.transform_xt()
        self.data.standardize_y()
        self.data.create_K_basis(n_pc=pc)
        if method == 'rag-obs':
            self.data.create_D_basis(D_type = 'linear')
            
        self.model = SepiaModel(self.data)
        
        #model.print_prior_info()  # Print information about the priors
        #model.print_value_info()  # Print information about the starting parameter values for MCMC
        #model.print_mcmc_info()   # Print information about the MCMC step types and step sizes
        
        self.model.tune_step_sizes(samp, step) #use 50 samples over 20 different step sizes

        self.model.do_mcmc(mcmc)

        self.samples = self.model.get_samples()

        self.pred_samples = self.model.get_samples(numsamples=10)

        #Prediction of model000
        if method == 'sim-only':
            self.pred = SepiaEmulatorPrediction(samples = self.pred_samples, model=self.model, x_pred = P_k_data.X_test.reshape(len(self.test_models),-1) if isinstance(self.test_models, list) else P_k_data.X_test.reshape(1, -1))
        elif method == 'rag-obs':
            self.pred = SepiaEmulatorPrediction(samples = self.pred_samples, model=self.model, t_pred = P_k_data.X_test.reshape(len(self.test_models),-1) if isinstance(self.test_models, list) else P_k_data.X_test.reshape(1, -1))

        self.predy = self.pred.get_y()

        self.y_pred_mean = np.mean(self.predy,0)

        #Unlog the predicted ratio
        self.y_pred_mean_norm = 10**(self.y_pred_mean)
        
        self.error = self.y_pred_mean_norm/P_k_data.Y_test

                            
    def mcmc_params(self):
        #Plots of MCMC parameters
        fig = SepiaPlot.mcmc_trace(self.samples)
        plt.show()

        ps = SepiaPlot.param_stats(self.samples) # returns pandas DataFrame
        print(ps)

        fig = SepiaPlot.rho_box_plots(self.model)
        plt.show()

    def plots(self, P_k_data, save=True):

        #Plot of non-linear P(k), true and predicted
        #The predicted is multiplied by linear interpolated P(k) to get back the non-linear spectrum
        if isinstance(self.test_models, int):
            plt.plot(P_k_data.k_test, P_k_data.Y_test, color='b', label=('True Non-linear P(k) for model_' + f"{self.test_models:03d}"))
            plt.plot(P_k_data.k_test, self.y_pred_mean_norm.reshape(100, -1), color='r', label=('Predicted Non-linear P(k) for model_' + f"{self.test_models:03d}"))
            plt.title('Comparison of predicted and true P(k) for model_' + f"{self.test_models:03d}", fontsize=10, wrap=True)
        elif isinstance(self.test_models, list):
            for m in range(len(self.test_models)):
                plt.plot(P_k_data.k_test, P_k_data.Y_test[m, :], color='b', label=('True Non-linear P(k) for model_'+ f"{self.test_models[m]:03d}"))
                plt.plot(P_k_data.k_test, self.y_pred_mean_norm[m, :], color='r', label=('Predicted Non-linear P(k) for model_'+ f"{self.test_models[m]:03d}"))
                plt.title('Comparison of predicted and true P(k) for ' + f"{len(self.test_models)}" + ' models', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('P(k) (Mpc^3)')
        #plt.xlim(right = 1)
        #plt.ylim(bottom = 0)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        
        if save == False:
            plt.show()
        elif save == True:
            plt.savefig('./Plots/Classbased_EMUTRUEVPRED.pdf', dpi=800)
        plt.clf()

        #Error plot
        plt.hlines(y=1.000, xmin=-1, xmax=P_k_data.high_k_cut, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=P_k_data.high_k_cut, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=P_k_data.high_k_cut, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=P_k_data.high_k_cut, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=P_k_data.high_k_cut, color='k', linestyles='dotted', alpha=0.5, label=None)
        if isinstance(self.test_models, int):
            plt.plot(P_k_data.k_test, self.error.reshape(100, -1), label=('Residual error for model_' + f"{self.test_models:03d}"))
            if isinstance(pc, int):
                plt.title('Residual error on non-linear P(k) for prediction of model_' + f"{self.test_models}" + ' (' + f"{pc:01d}" + 'pcs)', fontsize=10, wrap=True)
            else:
                plt.title('Residual error on non-linear P(k) for prediction of model_' + f"{self.test_models}", fontsize=10, wrap=True)
        elif isinstance(self.test_models, list):
            for m in range(len(self.test_models)):
                plt.plot(P_k_data.k_test, self.error[m, :], label=('Residual error for model_' + f"{self.test_models[m]:03d}"))
                if isinstance(pc, int):
                    plt.title('Residual error on non-linear P(k) for prediction of ' + f"{len(self.test_models)}" + ' models (' + f"{pc:01d}" + 'pcs)', fontsize=10, wrap=True)
                else:
                    plt.title('Residual error on non-linear P(k) for prediction of ' + f"{len(self.test_models)}" + ' models', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('Predicted Non-linear P(k)/True Non-linear P(k)')
        plt.xscale('log')
        plt.legend()

        if save == False:
            plt.show()
        elif save == True:
            plt.savefig('./Plots/Classbased_EMUERROR.pdf', dpi=800)
        plt.clf()


if __name__ == "__main__":
    test_model = random.randint(0, 149)    
    nbk_boost = BXL_DMO_Pk(test_model, 100, pk = 'nbk-rebin')
    sepia = sepia_emulator(nbk_boost, method='rag-obs')

    print(sepia.data)
    print(sepia.error)
    sepia.plots(nbk_boost)
