import numpy as np
import matplotlib.pyplot as plt
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
import sepia.SepiaPlot as SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
import sepia.SepiaSensitivity as SepiaSensitivity
from data_loader import BXL_DMO_Pk
import random
from sepia_emu_extension import sepia_HR_emulator, sepia_LR_emulator

#Some setup parameters for the model that I don't really touch now
pc = 3
save = True

class sepia_emulator:

    def __init__(self, P_k_data, method = 'sim-only', pc=0.99, samp=50, step=20, mcmc=1000, extension=False):
        self.test_models = P_k_data.test_models

        self.X_train = P_k_data.X_train
        self.X_test = P_k_data.X_test
        
        if extension == False:
            self.Y_train = P_k_data.Y_train
            self.Y_test = P_k_data.Y_test
            self.k_ind = P_k_data.k_test
        else:
            self.P_k_extend = np.zeros([150, 23])
            self.k_ind = np.zeros([3, 23])
            self.k_ind[0, :2] = P_k_data.k[50, :2]
            self.k_ind[0, 2:21] = P_k_data.k[0, :]
            self.k_ind[0, 21:] = P_k_data.k[100, 17:]
            self.k_ind[1, :19] = P_k_data.k[50, :]
            self.k_ind[1, 19:] = P_k_data.k[100, 15:]
            self.k_ind[2, :4] = P_k_data.k[50, :4]
            self.k_ind[2, 4:] = P_k_data.k[100, :]
            for h in range(150):
                if h in range(50):
                    self.P_k_extend[h, :2] = extension[1].med_res_LR[h, :]
                    self.P_k_extend[h, 2:21] = P_k_data.P_k[h, :]
                    self.P_k_extend[h, 21:] = extension[0].med_res_HR[h, :]
                elif h in range(100):
                    self.P_k_extend[h, :19] = P_k_data.P_k[h, :]
                    self.P_k_extend[h, 19:] = extension[0].low_res_HR[h-50, :]
                elif h in range(150):
                    self.P_k_extend[h, :4] = extension[1].high_res_LR[h-100, :]
                    self.P_k_extend[h, 4:] = P_k_data.P_k[h, :]
            self.Y_test = self.P_k_extend[self.test_models, :]
            self.Y_train = np.log10(np.delete(self.P_k_extend, self.test_models, axis=0))
            
        
        #SepiaData model building
        if method == 'sim-only':
            self.data = SepiaData(x_sim = self.X_train, y_sim = self.Y_train, y_ind_sim = self.k_ind)
        elif method == 'rag-obs':
            print(list(np.hsplit(self.Y_train[49:52, :].reshape(-1,3), 3)), list( np.hsplit(self.Y_train[99:102, :].reshape(-1,3), 3)))
            print(np.shape([np.array(self.k_ind[1, :]).reshape(-1,1), np.array(self.k_ind[2, :]).reshape(-1,1)]))
            if self.test_models in range(50):
                self.data = SepiaData(y_obs = [self.Y_train[49:99, :], self.Y_train[99:, :]], y_ind_obs = [np.array(self.k_ind[1, :]).reshape(-1,1), np.array(self.k_ind[2, :]).reshape(-1,1)], y_sim = self.Y_train[:49, :], t_sim = self.X_train[:49, :], y_ind_sim = np.array(self.k_ind[0, :]).reshape(-1,1))
            elif self.test_models in range(50, 100):
                self.data = SepiaData(y_obs = np.concatenate((self.Y_train[:50, :], self.Y_train[99:, :])), y_ind_obs = [self.k_ind[0, :], self.k_ind[2, :]], y_sim = self.Y_train[50:99, :], t_sim = self.X_train[50:99, :], y_ind_sim = self.k_ind[1, :])
            elif self.test_models in range(100, 150):
                self.data = SepiaData(y_obs = self.Y_train[:100, :], y_ind_obs = [self.k_ind[0, :], self.k_ind[1, :]], y_sim = self.Y_train[100:, :], t_sim = self.X_train[100:, :], y_ind_sim = self.k_ind[2, :])

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
            self.pred = SepiaEmulatorPrediction(samples = self.pred_samples, model=self.model, x_pred = self.X_test.reshape(len(self.test_models),-1) if isinstance(self.test_models, list) else self.X_test.reshape(1, -1))
        elif method == 'rag-obs':
            self.pred = SepiaEmulatorPrediction(samples = self.pred_samples, model=self.model, t_pred = self.X_test.reshape(len(self.test_models),-1) if isinstance(self.test_models, list) else self.X_test.reshape(1, -1))

        self.predy = self.pred.get_y()

        self.y_pred_mean = np.mean(self.predy,0)

        #Unlog the predicted ratio
        self.y_pred_mean_norm = 10**(self.y_pred_mean)
        
        self.error = self.y_pred_mean_norm/self.Y_test

        self.sens = SepiaSensitivity.sensitivity(self.model, self.samples)
        
        #main_max = np.max(self.sens['smePm'])
        #main_min = np.min(self.sens['smePm'])
        #total_max = np.max(self.sens['stePm'])
        #total_min = np.min(self.sens['stePm'])
        main_effect = self.sens['smePm']
        total_effect = self.sens['stePm']
        self.main_sens = main_effect/np.sum(main_effect)
        self.total_sens = total_effect/np.sum(total_effect)

        return
        
    def plot_sens(self, cosmo=np.arange(9)):
        #Sensitivity plots
        print(self.sens)
        parameter_names = ['Omega_m', 'f_b', 'h0', 'ns', 'A_s', 'w0', 'wa', 'Omega_nuh^2', 'alpha_s']
        plt.rc('xtick', labelsize=6)
        plt.bar(np.arange(len(cosmo))-0.2, self.main_sens*100, label='Main effect', width=0.4)
        plt.bar(np.arange(len(cosmo))+0.2, self.total_sens*100, label='Total effect', width=0.4)
        plt.xticks(np.arange(len(cosmo)), parameter_names)
        plt.xlabel('Parameters')
        plt.ylabel('Sensitivity')
        plt.legend()
        plt.savefig('./Plots/Classbased_SENSITIVITY.pdf', dpi=800)
        plt.rcdefaults()
        plt.clf()
        return        
                            
    def mcmc_params(self):
        #Plots of MCMC parameters
        fig = SepiaPlot.mcmc_trace(self.samples)
        plt.show()

        ps = SepiaPlot.param_stats(self.samples) # returns pandas DataFrame
        print(ps)

        fig = SepiaPlot.rho_box_plots(self.model)
        plt.show()

        return

    def plots(self, P_k_data, save=True):

        #Plot of non-linear P(k), true and predicted
        #The predicted is multiplied by linear interpolated P(k) to get back the non-linear spectrum
        if isinstance(self.test_models, int):
            plt.plot(P_k_data.k_test, self.Y_test, color='b', label=('True Non-linear P(k) for model_' + f"{self.test_models:03d}"))
            plt.plot(P_k_data.k_test, self.y_pred_mean_norm.reshape(100, -1), color='r', label=('Predicted Non-linear P(k) for model_' + f"{self.test_models:03d}"))
            plt.title('Comparison of predicted and true P(k) for model_' + f"{self.test_models:03d}", fontsize=10, wrap=True)
        elif isinstance(self.test_models, list):
            for m in range(len(self.test_models)):
                plt.plot(P_k_data.k_test, self.Y_test[m, :], color='b', label=('True Non-linear P(k) for model_'+ f"{self.test_models[m]:03d}"))
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

        return

    def plot_extension(self):

        #Plotting extension
        for i in range(50):
            plt.figure(1)
            plt.plot(self.k_ind[0, :], self.P_k_extend[i, :], color = 'tab:blue', label=('Models 000-049' if i==0 else None))
            plt.plot(self.k_ind[1, :], self.P_k_extend[i+50, :], color = 'tab:orange', label=('Models 050-099' if i==0 else None))
            plt.plot(self.k_ind[2, :], self.P_k_extend[i+100, :], color = 'tab:green', label=('Models 100-149' if i==0 else None))
            plt.figure(2)
            plt.plot(self.k_ind[0, :], self.P_k_extend[i, :], color = 'tab:blue')
            plt.figure(3)
            plt.plot(self.k_ind[1, :], self.P_k_extend[i+50, :], color = 'tab:orange')
            plt.figure(4)
            plt.plot(self.k_ind[2, :], self.P_k_extend[i+100, :], color = 'tab:green')

        plt.figure(1)
        plt.title('Boost function with extended range vs k for all models', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('P(k) (Mpc^3)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('./Plots/Classbased_EXTENSION.pdf', dpi=800)
        plt.clf()
        
        plt.figure(2)
        plt.title('Boost function with extended range vs k for models 0-49', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('P(k) (Mpc^3)')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('./Plots/Classbased_EXTENSION_medium.pdf', dpi=800)
        plt.clf()

        plt.figure(3)
        plt.title('Boost function with extended range vs k for models 50-99', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('P(k) (Mpc^3)')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('./Plots/Classbased_EXTENSION_low.pdf', dpi=800)
        plt.clf()
        
        plt.figure(4)
        plt.title('Boost function with extended range vs k for models 100-149', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('P(k) (Mpc^3)')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('./Plots/Classbased_EXTENSION_high.pdf', dpi=800)
        plt.clf()

        return

if __name__ == "__main__":
    #test_model = random.randint(0, 149)    
    test_model = 5
    nbk_boost = BXL_DMO_Pk(test_model, 100, pk='nbk-rebin', lin='rebin')
    #HR_extend = HR_emulator(nbk_boost)
    #LR_extend = LR_emulator(nbk_boost)
    #sepia = sepia_emulator(nbk_boost, method='rag-obs', extension=(HR_extend, LR_extend))
    sepia = sepia_emulator(nbk_boost)

    print(sepia.data)
    print(sepia.error)
    sepia.plots(nbk_boost)
    sepia.plot_sens()
