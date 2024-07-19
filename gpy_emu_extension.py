import numpy as np
import GPy
import pylab as pb

class gpyHighResEmulator:

    def __init__(self, P_k_data, kern=GPy.kern.RBF(9), consist=False):

        self.data = P_k_data

        if self.data.log == True:
            HR_model = GPy.models.GPHeteroscedasticRegression(self.data.parameters_norm[100:, :], np.log10(self.data.P_k_nonlinear[100:, :]), kern)
            HR_model.optimize()
            y = 10**HR_model._raw_predict(self.data.parameters_norm.reshape(150, -1))[0]
        else:
            HR_model = GPy.models.GPHeteroscedasticRegression(self.data.parameters_norm[100:, :], self.data.P_k_nonlinear[100:, :], kern)
            HR_model.optimize()
            y = HR_model._raw_predict(self.data.parameters_norm.reshape(150, -1))[0]

        midpoint = int(len(self.data.k_test)/2)
        self.data.P_k_nonlinear[:,midpoint:][self.data.nan_mask[:,midpoint:]] = y[:,midpoint:][self.data.nan_mask[:,midpoint:]]

        if isinstance(self.data.holdout, bool):
            if self.data.log == True:
                self.data.Y_train = np.log10(self.data.P_k_nonlinear)
            else:
                self.data.Y_train = self.data.P_k_nonlinear
        else:
            self.data.Y_test = self.data.P_k_nonlinear[self.data.holdout, :]
            if self.data.log == True:
                self.data.Y_train = np.log10(np.delete(self.data.P_k_nonlinear, self.data.holdout, axis=0))
            else:
                self.data.Y_train = np.delete(self.data.P_k_nonlinear, self.data.holdout, axis=0)
        
        return

class gpyLowResEmulator:

    def __init__(self, P_k_data, kern=GPy.kern.RBF(9), consist=False):

        self.data = P_k_data

        if self.data.log == True:
            LR_model = GPy.models.GPHeteroscedasticRegression(self.data.parameters_norm[50:100, :], np.log10(self.data.P_k_nonlinear[50:100, :]), kern)
            LR_model.optimize()
            y = 10**LR_model._raw_predict(self.data.parameters_norm.reshape(150, -1))[0]
        else:
            LR_model = GPy.models.GPHeteroscedasticRegression(self.data.parameters_norm[50:100, :], self.data.P_k_nonlinear[50:100, :], kern)
            LR_model.optimize()
            y = LR_model._raw_predict(self.data.parameters_norm.reshape(150, -1))[0]

        midpoint = int(len(self.data.k_test)/2)
        self.data.P_k_nonlinear[:,:midpoint][self.data.nan_mask[:,:midpoint]] = y[:,:midpoint][self.data.nan_mask[:,:midpoint]]
        
        try:
            assert True not in np.isnan(self.data.P_k_nonlinear[:,:midpoint])
        except AssertionError:
            mask = np.isnan(self.data.P_k_nonlinear[:,:midpoint])
            self.data.P_k_nonlinear[:,:midpoint][mask] = 1
            
        if isinstance(self.data.holdout, bool):
            if self.data.log == True:
                self.data.Y_train = np.log10(self.data.P_k_nonlinear)
            else:
                self.data.Y_train = self.data.P_k_nonlinear
        else:
            self.data.Y_test = self.data.P_k_nonlinear[self.data.holdout, :]
            if self.data.log == True:
                self.data.Y_train = np.log10(np.delete(self.data.P_k_nonlinear, self.data.holdout, axis=0))
            else:
                self.data.Y_train = np.delete(self.data.P_k_nonlinear, self.data.holdout, axis=0)
            
        return

class gpyMedResStepEmulator:

    def __init__(self, P_k_data, HL=False, LH=False, pc=0.999, samp=50, step=20, mcmc=1000):
        self.test_models = P_k_data.test_models

        MR_X = P_k_data.parameters_norm[50:100, :]
        if HL == False:
            pass
        else:
            MR_Y = HL.P_k[50:100, :]
        #elif LH == True:
            #MR_Y = P_k_data.P_k[50:100, :]

        MR_data = SepiaData(x_sim = MR_X, y_sim = HR_Y, y_ind_sim = P_k_data.k[100, :])
        HR_data.transform_xt()
        HR_data.standardize_y()
        HR_data.create_K_basis(n_pc=pc)
        
        HR_model = SepiaModel(HR_data)
        HR_model.tune_step_sizes(samp, step)
        HR_model.do_mcmc(mcmc)
        HR_samples = HR_model.get_samples()
        
        low_res_max = max(P_k_data.k[50, :])
        med_res_max = max(P_k_data.k[0, :])
        self.med_res_HR = np.zeros([50, 2])
        self.low_res_HR = np.zeros([50, 4])
        
        for i in range(100):
            m=0
            HR_pred = SepiaEmulatorPrediction(samples = HR_samples, model=HR_model, x_pred = P_k_data.parameters_norm[i, :].reshape(1,-1))
            HR_predy = HR_pred.get_y()
            HR_pred_mean = np.mean(HR_predy,0)
            if i in range(50):
                for j in P_k_data.k[100, :]:
                    if j <= med_res_max:
                        m += 1
                self.med_res_HR[i, :] = HR_pred_mean[0,m:]
            else:
                for j in P_k_data.k[100, :]:
                    if j <= low_res_max:
                        m += 1
                self.low_res_HR[i-50, :] = HR_pred_mean[0,m:]

        return
                                
                                
if __name__ == "__main__":
    import random
    #from data_loader import bahamasXLDMOData
    from flamingo_data_loader import flamingoDMOData

    test = random.randint(100,149)
    boost = flamingoDMOData(pk='nbk-rebin-std', lin='rebin', log=False, holdout=test)
    #boost.weights(plot_weights=True)
    #emulator = gpyImprovedEmulator(boost, 'variance_weights', fix_variance=True, ARD=True, flamingo=False)
    #emulator.plot('ARD')

    HR_model = GPy.models.GPHeteroscedasticRegression(boost.parameters_norm[100:, :], boost.P_k_nonlinear[100:, :], GPy.kern.RBF(9))
    HR_model.optimize()
    y = HR_model._raw_predict(boost.parameters_norm[test, :].reshape(1,-1))[0]

    pb.plot(boost.k_test, boost.P_k_nonlinear[test, :], label='True Pk')
    pb.plot(boost.k_test, y.reshape(15), label='Predicted Pk', linestyle='dotted')
    pb.xscale('log')
    pb.yscale('log')
    pb.legend()
    pb.savefig('./Plots/HR_test.png', dpi=1200)
    pb.clf()
