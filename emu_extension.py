import numpy as np
from data_loader import BXL_DMO_Pk
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaPredict import SepiaEmulatorPrediction
import random

class HR_emulator:

    def __init__(self, P_k_data, pc=0.999, samp=50, step=20, mcmc=1000):
        self.test_models = P_k_data.test_models

        HR_X = P_k_data.parameters_norm[100:, :]
        HR_Y = P_k_data.P_k[100:, :]

        HR_data = SepiaData(x_sim = HR_X, y_sim = HR_Y, y_ind_sim = P_k_data.k[100, :])
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

class LR_emulator:

    def __init__(self, P_k_data, pc=0.999, samp=50, step=20, mcmc=1000):
        self.test_models = P_k_data.test_models
        
        LR_X = P_k_data.parameters_norm[50:100, :]
        LR_Y = P_k_data.P_k[50:100, :]
        
        LR_data = SepiaData(x_sim = LR_X, y_sim = LR_Y, y_ind_sim = P_k_data.k[50, :])
        LR_data.transform_xt()
        LR_data.standardize_y()
        LR_data.create_K_basis(n_pc=pc)
        
        LR_model = SepiaModel(LR_data)
        LR_model.tune_step_sizes(samp, step)
        LR_model.do_mcmc(mcmc)
        LR_samples = LR_model.get_samples()
        
        high_res_min = min(P_k_data.k[100, :])
        med_res_min = min(P_k_data.k[0, :])
        self.med_res_LR = np.zeros([50, 2])
        self.high_res_LR = np.zeros([50, 4])
        
        for i in range(100):
            m=0
            LR_pred = SepiaEmulatorPrediction(samples = LR_samples, model=LR_model, x_pred = P_k_data.parameters_norm[i, :].reshape(1,-1) if i in range(50) else P_k_data.parameters_norm[i+50, :].reshape(1,-1))
            LR_predy = LR_pred.get_y()
            LR_pred_mean = np.mean(LR_predy,0)
            if i in range(50):
                for j in P_k_data.k[50, :]:
                    if j <= med_res_min:
                        m += 1
                self.med_res_LR[i, :] = LR_pred_mean[0,1:m]
            else:
                for j in P_k_data.k[50, :]:
                    if j <= high_res_min:
                        m += 1
                self.high_res_LR[i-50, :] = LR_pred_mean[0,1:m]
            
        return

class MR_step_emulator:

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
    n = 105
    nbk_boost = BXL_DMO_Pk(n, 100, pk = 'nbk-rebin', lin = 'rebin')
    HR = HR_emulator(nbk_boost)
    LR = LR_emulator(nbk_boost)
    print(HR.low_res_HR)
    print(HR.med_res_HR)
    print(LR.high_res_LR)
    print(LR.med_res_LR)
