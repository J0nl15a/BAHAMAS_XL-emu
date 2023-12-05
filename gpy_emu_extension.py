import numpy as np
from data_loader import BXL_DMO_Pk
import GPy
import random
import pylab as pb
from gpy_emulator import gpy_emulator

class gpy_HR_emulator:

    def __init__(self, P_k_data, res, k, kern=GPy.kern.RBF(9), test=False, consist=False):
        self.data = P_k_data

        for i, t in enumerate(res):
            HR = 100
        
            if t == 'Low':
                start = 50
                end = 100
            elif t == 'Med':
                start = 0
                end = 50
            elif t == 'High':
                start = 100
                end = 150

            if test == False:
                pass
            else:
                if self.data.test_models in range(100):
                    HR -= 1
                if self.data.test_models in range(start, end):
                    end -= 1
                elif self.data.test_models < start:
                    start -= 1
                    end -= 1

            HR_model = GPy.models.GPHeteroscedasticRegression(self.data.X_train[HR:, :], self.data.Y_train[HR:, :], kern)
            HR_model.optimize()
            y = HR_model._raw_predict(self.data.X_train[start:end, :].reshape((end-start), -1))[0]
            #print(y[:, k:])
            
            self.data.Y_train[start:end, k[i]:] = y[:, k[i]:]

        if test == False:
            pass
        else:
            z = HR_model._raw_predict(self.data.X_test.reshape(1, -1))[0][0]
            if consist == True:
                print('HR Consistency test')
                print(self.data.Y_test)
                print(10**z)
                pb.plot(self.data.k_test, self.data.Y_test, label='Test data (Low res)')
                pb.plot(self.data.k_test, (10**z).reshape(-1,1), label='HR predicted test data')
                pb.title('HR Consistency test')
                pb.xscale('log')
                pb.yscale('log')
                pb.xlabel('k (1/Mpc)')
                pb.xlabel('P(k) (Mpc^3)')
                pb.legend()
                pb.savefig('./Plots/emu_ext_test_HR.pdf', dpi=800)
                pb.clf()
            for j in range(test, 15):
                self.data.Y_test[j] = 10**(z[j])
        
        return

class gpy_LR_emulator:

    def __init__(self, P_k_data, res, k, kern=GPy.kern.RBF(9), test=False, consist=False):
        self.data = P_k_data

        for i, t in enumerate(res):
            LR = [50, 100]

            if t == 'High':
                start = 100
                end = 150
            elif t == 'Med':
                start = 0
                end = 50
            elif t == 'Low':
                start = 50
                end = 100
            
            if test == False:
                pass
            else:
                if self.data.test_models in range(50, 100):
                    LR[1] -= 1
                elif self.data.test_models in range(50):
                    LR[0] -= 1
                    LR[1] -= 1

                if self.data.test_models in range(start, end):
                    end -= 1
                elif self.data.test_models < start:
                    start -= 1
                    end -= 1

            LR_model = GPy.models.GPHeteroscedasticRegression(self.data.X_train[LR[0]:LR[1], :], self.data.Y_train[LR[0]:LR[1], :], kern)
            LR_model.optimize()
            print(self.data.X_train[start:end, :].shape)
            y = LR_model._raw_predict(self.data.X_train[start:end, :].reshape((end-start), -1))[0]
            #print(y[:, :k])
        
            self.data.Y_train[start:end, :k[i]] = y[:, :k[i]]
            self.data.Y_train[start:end, 0] = 0

        if test == False:
            pass
        else:
            z = LR_model._raw_predict(self.data.X_test.reshape(1, -1))[0][0]
            if consist == True:
                print('LR Consistency test')
                print(self.data.Y_test)
                print(10**z)
                pb.plot(self.data.k_test, self.data.Y_test, label='Test data (High res)')
                pb.plot(self.data.k_test, (10**z).reshape(-1,1), label='LR predicted test data')
                pb.title('LR Consistency test')
                pb.xscale('log')
                pb.yscale('log')
                pb.xlabel('k (1/Mpc)')
                pb.xlabel('P(k) (Mpc^3)')
                pb.legend()
                pb.savefig('./Plots/emu_ext_test_LR.pdf', dpi=800)
                pb.clf()
            for j in range(test):
                self.data.Y_test[j] = 10**(z[j])
            self.data.Y_test[0] = 1
            
        return

class gpy_MR_step_emulator:

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
    n = random.randint(50,99)
    m = random.randint(100,149)
    nbk_boost = BXL_DMO_Pk(n, 100, pk='nbk-rebin-std', lin='rebin')
    nb = BXL_DMO_Pk(m, 100, pk='nbk-rebin-std', lin='rebin')
    print("Default Y_train:")
    print(nbk_boost.Y_train[24])
    print(nbk_boost.Y_train[79])
    print(nbk_boost.Y_train[124])
    lr = gpy_LR_emulator(nb, ['Med', 'Low', 'High'], [2, 1, 3], test=1, consist=True).data
    HR = gpy_HR_emulator(nbk_boost, ['Low'], [len(nbk_boost.k_test)-1], test=len(nbk_boost.k_test)-1, consist=True).data
    print("HR corrected Y_train:")
    print(HR.Y_train[24])
    print(HR.Y_train[79])
    print(HR.Y_train[124])
    LR = gpy_LR_emulator(nbk_boost, ['Med', 'Low', 'High'], [2, 1, 3], test=1).data
    print("LR corrected Y_train:")
    print(LR.Y_train[24])
    print(LR.Y_train[79])
    print(LR.Y_train[124])

    print("Default Y_train again:")
    print(nbk_boost.Y_train[24])
    print(nbk_boost.Y_train[79])
    print(nbk_boost.Y_train[124])

    gpy = gpy_emulator(nbk_boost, GPy.kern.RBF(9, ARD=True))

    pb.plot(nbk_boost.k_test, nbk_boost.Y_test, color='b')
    pb.plot(nbk_boost.k_test, gpy.y.reshape(-1,1), color='r')
    pb.xscale('log')
    pb.yscale('log')
    pb.savefig('./Plots/emu_ext_test.pdf', dpi=800)
    pb.clf()

    pb.hlines(y=1.000, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='solid', alpha=0.5, label=None)
    pb.hlines(y=0.990, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label='1% error')
    pb.hlines(y=1.010, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label=None)
    pb.hlines(y=0.950, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label='5% error')
    pb.hlines(y=1.050, xmin=-1, xmax=max(nbk_boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label=None)
    pb.plot(nbk_boost.k_test, gpy.error.reshape(-1,1))
    pb.xscale('log')
    pb.savefig('./Plots/emu_ext_test_err.pdf', dpi=800)
    pb.clf()
                        
    for i in range(50):
        pb.plot(nbk_boost.k_test, 10**nbk_boost.Y_train[i, :], color='tab:blue')
        pb.plot(nbk_boost.k_test, 10**nbk_boost.Y_train[i+49, :], color='tab:orange')
        pb.plot(nbk_boost.k_test, 10**nbk_boost.Y_train[i+99, :], color='tab:green')
    pb.plot(nbk_boost.k_test, nbk_boost.Y_test, color='tab:orange')
    pb.xscale('log')
    pb.yscale('log')
    pb.savefig('./Plots/emu_ext.pdf', dpi=800)
    pb.clf()
