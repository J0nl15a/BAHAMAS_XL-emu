import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import random

bins = 100
#cutoff = (np.pi*1260)/1400

#test_model = [3, 12, 18, 90, 138]
#test_models = random.randint(0, 149)
test_models = 50
#test_model = 11


class BXL_DMO_Pk:

    def __init__(self, test_models, bins, cutoff = (0.025, 4.5), cosmo = np.arange(9), pk = 'reg', lin = 'class', boost = True):
        self.test_models = test_models
        
        #The parameters file
        self.parameters = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/slhs_nested_3x50_w0_m0p6_m1p2_wa_m1p6_p0p5_with_running_and_fgas_and_As.txt', skiprows=1, max_rows=150, usecols=cosmo)

        #Normalize the design to be between 0,1
        self.design_max = np.max(self.parameters, 0)
        self.design_min = np.min(self.parameters, 0)
        self.parameters_norm = (self.parameters-self.design_min)/(self.design_max-self.design_min)
    
        #Split into test and train data
        self.test_parameters = self.parameters_norm[self.test_models, :]
        self.train_parameters = np.delete(self.parameters_norm, self.test_models, axis=0)
        
        if pk == 'reg':
            self.k = np.zeros([150, 435])
            self.P_k = np.zeros([150, 435])
        elif pk == 'nbk':
            self.k = np.zeros([150, 1023])
            self.P_k = np.zeros([150, 1023])

        self.P_k_interp = np.zeros([150, bins])
        self.P_k_nonlinear = np.zeros([150, bins])

        low_k_cut = cutoff[0]
        high_k_cut = cutoff[1]
        self.k_test = np.logspace(np.log10(low_k_cut), np.log10(high_k_cut), bins)

        if lin == 'camb':
            self.k_camb_linear = np.logspace(-3, np.log10(50), 300)
            self.P_k_camb_linear = np.loadtxt('/home/arijconl/BAHAMAS_XL/pk_lin_camb2022_slhs_nested_3x50_kmax_50_running_w0_m1p2_m0p6_wa_m1p6_p0p5.txt', skiprows=index, max_rows=1)
        elif lin == 'class':
            self.P_k_class_linear = np.zeros((150, 400, 2))
    
        #This is used to build up k, the full P(k), linear P(k) and non-linear P(k)
        for i in range(150):
        
            print('model = ' + str(i))
            if i in np.arange(0, 50):
                size = 700
            elif i in np.arange(50, 100):
                size = 1400
            elif i in np.arange(100, 150):
                size = 350

            if pk == 'reg':
                self.k[i, :] = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/power_spectra/power_matter_0122.txt', skiprows = 20, usecols = 1)

                self.P_k[i, :] = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/power_spectra/power_matter_0122.txt', skiprows = 20, usecols = 2)

            elif pk == 'nbk':
                self.k[i, :] = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/PS/PS_k_078.csv', skiprows = 1, usecols = 0)*(self.parameters[i, 2])
                
                self.P_k[i, :] = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/PS/PS_k_078.csv', skiprows = 1, usecols = 1)/(self.parameters[i, 2]**3)
                
            h = interpolate.interp1d(self.k[i, :], self.P_k[i, :], kind='cubic')#, fill_value="extrapolate") #if pk == 'nbk' else 'nan')
            self.P_k_interp[i, :] = h(self.k_test)
        
            if lin == 'camb':
                k_linear = k_linear*(self.parameters[i, 2])
                P_k_linear = P_k_linear/(self.parameters[i, 2]**3)
                
                f = interpolate.interp1d(k_linear, P_k_linear, kind='cubic')
                y = f(self.k_test)

            elif lin == 'class':
                self.P_k_class_linear[i, :, :] = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" +'_N1260_L' + f"{size}" + '_DMO/class_linear_spectra_z_0.txt', skiprows=2, usecols=(0, 2))

                f = interpolate.interp1d(self.P_k_class_linear[i, :, 0], self.P_k_class_linear[i, :, 1], kind='cubic')
                y = f(self.k_test)

            if boost == True:
                self.P_k_nonlinear[i, :] = self.P_k_interp[i, :] / y
            elif boost == False:
                self.P_k_nonlinear[i, :] = self.P_k_interp[i, :]

        self.P_k_nonlinear_test = self.P_k_nonlinear[self.test_models, :]
        self.P_k_nonlinear_train = np.delete(self.P_k_nonlinear, self.test_models, axis=0)
        
        #Log of non-linear boost
        self.P_k_nonlinear_train_log = np.log10(self.P_k_nonlinear_train)
        self.k_ind = np.linspace(0, 1, bins)
        #k_z0_test_array = np.repeat(k_z0_test, np.size(P_k_z0_nonlinear_train_log, 0), axis=0).reshape(-1, 100)
        #print(k_z0_test_array)
            

    def plot_k(self):

        for i in range(150):
            plt.figure(3)
            if i in range(50):
                plt.plot(self.k_test, self.P_k_interp[i, :], color = 'tab:blue', label=('Models 000-049' if i==0 else None))
            elif i in range(50, 100):
                plt.plot(self.k_test, self.P_k_interp[i, :], color = 'tab:orange', label=('Models 050-099' if i==50 else None))
            elif i in range(100, 150):
                plt.plot(self.k_test, self.P_k_interp[i, :], color = 'tab:green', label=('Models 100-149' if i==100 else None))

            u=0
            for t in self.k[i, :]:
                if t <= max(self.P_k_class_linear[i, :, 0]):
                    u+=1
                    k_plot = self.k[i, :u]
            e = interpolate.interp1d(self.P_k_class_linear[i, :, 0], self.P_k_class_linear[i, :, 1], kind='cubic')
            p = f(k_plot)

            plt.figure(1)
            if i in range(50):
                plt.plot(k_plot, self.P_k[i, :u]/p, color='tab:blue', label=('Models 000-049' if i==0 else None))
            elif i in range(50, 100):
                plt.plot(k_plot, self.P_k[i, :u]/p, color='tab:orange', label=('Models 050-099' if i==50 else None))
            elif i in range(100, 150):
                plt.plot(k_plot, self.P_k[i, :u]/p, color='tab:green', label=('Models 100-149' if i==100 else None))

            plt.figure(2)
            if i in range(50):
                k_fund = (2*np.pi)/700
                plt.plot(k_plot/k_fund, P_k[i, :u]/p, color='tab:blue', label=('Models 000-049' if i==0 else None))
            elif i in range(50, 100):
                k_fund = (2*np.pi)/1400
                plt.plot(k_plot/k_fund, P_k_z0[:u]/p, color='tab:orange', label=('Models 050-099' if i==50 else None))
            elif i in range(100, 150):
                k_fund = (2*np.pi)/350
                plt.plot(k_plot/k_fund, P_k_z0[:u]/p, color='tab:green', label=('Models 100-149' if i==100 else None))

        plt.figure(1)
        plt.title('Boost function vs k for all models', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('P(k) (Mpc^3)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('./Plots/Power_spectra_z0_predicted_boostvk.pdf', dpi=800)
        plt.clf()
        
        plt.figure(2)
        plt.title('Boost function vs k/k_fundamental for all models', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('P(k) (Mpc^3)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('./Plots/Power_spectra_z0_predicted_boostvkfund.pdf', dpi=800)
        plt.clf()
        
        plt.figure(3)
        plt.title('P(k) vs test_k for all models', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('P(k) (Mpc^3)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('./Plots/Power_spectra_z0_predicted_testcase.pdf', dpi=800)
        plt.clf()

                
    def rebin(self, k_z0, P_k_z0, h):
        P_k_z0_nonlinear = np.zeros([150, 8])
        k_rebin = np.arange(2, 257, 2)*(np.pi/size)
        P_k_z0_interp = P_k_z0
        k_z0_test = k_rebin
        k_eff = np.sum(P_k_z0 * k_z0) /np.sum(P_k_z0)
        print(k_eff)
        low_k = 0.0352 * k_eff
        high_k = 8 * k_eff
        k_z0_test = np.arange(low_k, high_k, k_eff)
        print(k_z0_test)
        P_k_z0_interp = h(k_z0_test)
        y = f(k_rebin)
        k_plot = k_z0_test
            
        
if __name__ == "__main__":
    nbk_boost = BXL_DMO_Pk(test_models, bins, pk = 'nbk')
    print(nbk_boost.test_models)
    print(nbk_boost.test_parameters)
    print(nbk_boost.k_test)
    print(nbk_boost.P_k_nonlinear_test)
