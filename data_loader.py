import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import random
import pandas as pd
from pade import pade
from pade_func import pade_func
from hypercube import hypercube_plot
from gpy_emu_extension import gpy_HR_emulator
import pdb
import glob


class BXL_DMO_Pk:

    def __init__(self, test_models, bins, cutoff = (-1, 10), cosmo = np.arange(9), pk = 'nbk-rebin', lin = 'class', boost = True, extrap = False, add_pade = False, pad = False, holdout = True, sigma8=False, plot_hypercube=False):
        self.test_models = test_models
        self.bins = bins
        self.pk = pk
        self.pad = pad
        self.lin = lin
        self.modes = {}
        self.low_k_cut = cutoff[0]
        self.high_k_cut = cutoff[1]
        
        #The parameters file
        self.parameters = np.loadtxt('./BXL_data/slhs_nested_3x50_w0_m0p6_m1p2_wa_m1p6_p0p5_with_running_and_fgas_and_As.txt', skiprows=1, max_rows=150, usecols=cosmo)
        if sigma8 == True:
            sig8 = np.loadtxt('./BXL_data/slhs_nested_3x_Om_fb_h_ns_sigma8_w0_wa_Omnuh2_alphas_fgasfb.txt', max_rows=150, usecols=4)
            self.parameters[:, 4] = sig8

        if plot_hypercube == True:
            labels = {r'$\Omega_m$':[.20,.25,.30,.35], r'$f_b$':[.14,.15,.16,.17], r'$h_0$':[.64,.7,.76], r'$n_s$':[.95,.97,.99], r'$\sigma_8$':[0.72,0.80,0.88], r'$w_0$':[-.7,-.9,-1.1], r'$w_a$':[.2,-.5,-1.2], r'$\Omega_{\nu}h^2$':[.005,.003,.001], r'$\alpha_s$':[.025,0,-.025]}
            
            hypercube = hypercube_plot(data=[self.parameters[:50, :], self.parameters[50:100, :], self.parameters[100:, :]], parameter_labels=labels, save_to='./Plots/BXL_hypercube.png', title='BAHAMAS XL Latin hypercube design')
            
        #Normalize the design to be between 0,1
        self.design_max = np.max(self.parameters, 0)
        self.design_min = np.min(self.parameters, 0)
        self.parameters_norm = (self.parameters-self.design_min)/(self.design_max-self.design_min)
    
        #Split into test and train data
        if holdout == False:
            self.X_train = self.parameters_norm
        else:
            self.X_test = self.parameters_norm[self.test_models, :]
            self.X_train = np.delete(self.parameters_norm, self.test_models, axis=0)

        def extract_series(group):
            return {'k': group['k'].values, 'Pk': group['boost'].values, 'model': group['model'].iloc[0], 'modes': group['N_modes'].values if pk == 'nbk-rebin-std' else None}

        if pk == 'nbk-rebin' or pk == 'nbk-rebin-std':
            if pk == 'nbk-rebin':
                df = pd.read_csv("./BXL_data/boost_rebinned_21.csv")
            elif pk == 'nbk-rebin-std':
                df = pd.read_csv("./BXL_data/boost_rebinned_standardized.csv")
                
            grouped = df.groupby('model')
            arrays_per_model = grouped.apply(extract_series)
            array_size = len(arrays_per_model[0]['k'])
            self.bins = array_size

            if pk == 'nbk-rebin-std':
                self.modes['Med res'] = arrays_per_model[0]['modes'].tolist()
                self.modes['Low res'] = arrays_per_model[50]['modes'].tolist()
                self.modes['High res'] = arrays_per_model[100]['modes'].tolist()

        else:
            random_int = random.randint(0, 149)
            if pk == 'nbk':
                random_model = np.loadtxt(glob.glob('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{random_int:03d}" + '_N1260_L*_DMO/PS/PS_k_078.csv')[0], skiprows = 1, usecols = (0,1), delimiter=' ')
                array_size = len(random_model)
            else:
                random_model = np.loadtxt(glob.glob('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{random_int:03d}" + '_N1260_L*_DMO/power_spectra/power_matter_0122.txt')[0], skiprows = 20, usecols = (1,2,3))
                for x in range(len(random_model)):
                    if random_model[x,1] < random_model[x,2]:
                        array_size = x-1
                        break

        if pk == 'nbk-rebin-std':
            self.k_test = arrays_per_model[0]['k']
        else:
            if pk == 'nbk-rebin':
                minimum_k = min(arrays_per_model[100]['k'])
                maximum_k = max(arrays_per_model[50]['k'])
            elif pk == 'nbk':
                minimum_k = min(random_model[0] if random_int in range(100, 150) else np.loadtxt(glob.glob('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{100:03d}" + '_N1260_L*_DMO/PS/PS_k_078.csv')[0], skiprows = 1, usecols= 0, delimiter=' '))
                maximum_k = max(random_model[0] if random_int in range(50, 100) else np.loadtxt(glob.glob('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{50:03d}" + '_N1260_L*_DMO/PS/PS_k_078.csv')[0], skiprows = 1, usecols= 0, delimiter=' '))
            elif pk == 'powmes':
                minimum_k = min(random_model[0] if random_int in range(100, 150) else np.loadtxt(glob.glob('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{100:03d}" + '_N1260_L*_DMO/power_spectra/power_matter_0122.txt')[0], skiprows = 20, usecols = 1))
                maximum_k = max(random_model[0] if random_int in range(50, 100) else np.loadtxt(glob.glob('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{50:03d}" + '_N1260_L*_DMO/power_spectra/power_matter_0122.txt')[0], skiprows = 20, usecols = 1))

            if self.low_k_cut == -1:
                self.low_k_cut = minimum_k

            try:
                print(self.low_k_cut, minimum_k)
                self.low_k_cut >= minimum_k
            except ValueError:
                print('self.low_k_cut > minimum_k == False')
                self.low_k_cut = minimum_k

            try:
                print(self.high_k_cut, maximum_k)
                self.high_k_cut <= maximum_k
            except ValueError:
                print('self.high_k_cut < maximum_k == False')
                self.high_k_cut = maximum_k
                
            self.k_test = np.logspace(np.log10(self.low_k_cut), np.log10(self.high_k_cut), self.bins)
            quit()
            
        if self.lin == 'camb':
            self.k_camb_linear = np.tile(np.logspace(-3, np.log10(50), 300), (150,1))
            self.P_k_camb_linear = np.loadtxt('./BXL_data/pk_lin_camb2022_slhs_nested_3x50_kmax_50_running_w0_m1p2_m0p6_wa_m1p6_p0p5.txt')
        elif self.lin == 'class':
            class_directory = glob.glob('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_*_N1260_L*_DMO/class_linear_spectra_z_0.txt')
            for index, directory in enumerate(class_directory):
                if index == 0:
                    linear_spectra = np.loadtxt(directory, skiprows=2, usecols=(0,2))
                    k_lin_size = P_k_lin_size = len(linear_spectra)
                    self.k_class_linear = np.zeros([150, k_lin_size])
                    self.P_k_class_linear = np.zeros([150, P_k_lin_size])
                self.k_class_linear[index, :] = np.loadtxt(directory, skiprows=2, usecols=0)
                self.P_k_class_linear[index, :] = np.loadtxt(directory, skiprows=2, usecols=2)
        elif self.lin == 'rebin':
            pass

        #This is used to build up k, the full P(k), linear P(k) and non-linear P(k)

        self.k = np.zeros([150, array_size])
        self.P_k = np.zeros([150, array_size])
        self.P_k_interp = np.zeros([150, len(self.k_test)])
        self.P_k_nonlinear = np.zeros([150, len(self.k_test)])
                                
        if pk == 'powmes':
            training_directory = glob.glob('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_*_N1260_L*_DMO/power_spectra/power_matter_0122.txt')

            for index, directory in enumerate(training_directory):
                self.k[index, :] = np.loadtxt(directory, skiprows = 20, usecols = 1)[:array_size]
                self.P_k[index, :] = np.loadtxt(directory, skiprows = 20, usecols = 2)[:array_size]

        elif pk == 'nbk':
            training_directory = glob.glob('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_*_N1260_L*_DMO/PS/PS_k_078.csv')

            for index, directory in enumerate(training_directory):
                self.k[index, :] = np.loadtxt(directory, skiprows = 1, usecols = 0, delimiter=' ')*(self.parameters[index, 2])
                self.P_k[index, :] = np.loadtxt(directory, skiprows = 1, usecols = 1, delimiter=' ')/(self.parameters[index, 2]**3)
                self.modes[index] = np.loadtxt(directory, skiprows = 1, usecols = 2, delimiter=' ')

        elif pk == 'nbk-rebin' or 'nbk-rebin-std':
            for model in arrays_per_model:
                self.k[model['model'], :] = model['k']
                self.P_k[model['model'], :] = model['Pk']

        print(self.k.shape)
        print(self.P_k.shape)
        print(self.P_k)
                
        #if extrap == True:
            #self.k_test = np.logspace(np.log10(self.low_k_cut), np.log10(10), self.bins)

        if pk == 'nbk-rebin-std':
            self.P_k_interp = self.P_k
        else:
            interpolation_function = {}
            for i in range(len(self.k)):
                if add_pade == False:
                    interpolation_function[i] = interpolate.interp1d(self.k[i, :], self.P_k[i, :], kind='cubic', fill_value="nan")
                elif add_pade == True:
                    print(self.k[i, -4:-1], self.P_k[i, -4:-1])
                    interpolation_function[i] = pade.fit(self.k[i, -4:-1], self.P_k[i, -4:-1])
                    #print(self.k_test[-4:-1], h(self.k_test))
                
                self.P_k_interp[i, :] = interpolation_function[i](self.k_test)

        linear_interpolation_function = {}
        if self.lin == 'camb':
            k_linear = self.k_camb_linear*np.reshape(self.parameters[:, 2], (150,1))
            P_k_linear = self.P_k_camb_linear/(np.reshape(self.parameters[:, 2], (150,1))**3)
        elif self.lin == 'class':
            k_linear = self.k_class_linear
            P_k_linear = self.P_k_class_linear

        if self.lin == 'rebin':
            self.P_k_linear_interp = np.ones([150, self.P_k_interp.shape[1]])
        else:
            self.P_k_linear_interp = np.zeros([150, self.P_k_interp.shape[1]])
            for l in range(len(k_linear)):
                linear_interpolation_function[l] = interpolate.interp1d(k_linear[l], P_k_linear[l], kind='cubic')
                self.P_k_linear_interp[l, :] = linear_interpolation_function[l](self.k_test)

        if pk == 'nbk-rebin' or pk == 'nbk-rebin-std':
            self.P_k_nonlinear = self.P_k_interp
        else:
            if boost == True:
                self.P_k_nonlinear = self.P_k_interp / self.P_k_linear_interp
            elif boost == False:
                self.P_k_nonlinear = self.P_k_interp

        print(self.P_k_nonlinear)
        #quit()
        
        if pad == True:
            for f in range(50):
                #q = interpolate.interp1d(self.k_test[1:-1], self.P_k_nonlinear[50:100, 1:-1], kind='cubic', fill_value="extrapolate")
                #q = pade.fit(self.k_test[3:5], np.log10(self.P_k_nonlinear[f, 3:5]))
                pade_func((3, 8, 13), self.k_test, self.P_k_nonlinear[f, :], (2, 15, 15), self.k_test, self.P_k_nonlinear[f, :])
                #self.P_k_nonlinear[f, :] = [10**s for s in self.P_k_nonlinear[f, :]]
                #y = q(self.k_test[:2])
                #self.P_k_nonlinear[:50, :2] = [10**s for s in y]
                #q = pade.fit(self.k_test[2:4], np.log10(self.P_k_nonlinear[f+50, 2:4]))
                pade_func((2, 7, 12), self.k_test, self.P_k_nonlinear[f+50, :], (1, 14, 15), self.k_test, self.P_k_nonlinear[f+50, :])
                #self.P_k_nonlinear[f+50, :] = [10**s for s in self.P_k_nonlinear[f+50, :]]
                #self.P_k_nonlinear[50:100, 0] = 10**(q(self.k_test[0]))
                #self.P_k_nonlinear[50:100, -1] = 10**(q(self.k_test[-1]))
                #y = q(self.k_test[:3])
                #self.P_k_nonlinear[100:, :3] = [10**s for s in y]
                #q = pade.fit(self.k_test[-4:-2], np.log10(self.P_k_nonlinear[f+50, -4:-2]))
                pade_func((4, 8, 13), self.k_test, self.P_k_nonlinear[f+100, :], (3, 15, 15), self.k_test, self.P_k_nonlinear[f+100, :])
                #self.P_k_nonlinear[f+100, :] = [10**s for s in self.P_k_nonlinear[f+100, :]]
                #self.P_k_nonlinear[50:100, -1] = 10**(q(self.k_test[-1]))
                #q = interpolate.interp1d(self.k_test[2:], self.P_k_nonlinear[:50, 2:], kind='cubic', fill_value="extrapolate")
                #y = q(self.k_test[:2])
                #self.P_k_nonlinear[:50, :2] = [10**s for s in y]
                #q = interpolate.interp1d(self.k_test[3:], self.P_k_nonlinear[100:, 3:], kind='cubic', fill_value="extrapolate")
                #q = pade.fit(self.k_test[4:6], self.P_k_nonlinear[f+100, 4:6])
                #self.P_k_nonlinear[50:100, -1] = 10**q(self.k_test[-1])
                #y = q(self.k_test[:3])
                #self.P_k_nonlinear[100:, :3] = [10**s for s in y]

        try:
            logged_array = np.log10(self.P_k_nonlinear)
        except ValueError:
            nan_mask = np.isnan(self.P_k_nonlinear)
            self.P_k_nonlinear[nan_mask] = 0
            #extended_data = gpy_HR_emulator(nbk_boost, ['Low'], [len(nbk_boost.k_test)-1], test=len(nbk_boost.k_test)-1, consist=True).data
            
        if holdout == False:
            self.Y_train = np.log10(self.P_k_nonlinear)
        else:
            self.Y_test = self.P_k_nonlinear[self.test_models, :]
            self.Y_train = np.log10(np.delete(self.P_k_nonlinear, self.test_models, axis=0))

        print(self.Y_train)

    def plot_k(self):

        for i in range(50):
            plt.figure(3)
            plt.plot(self.k_test, self.P_k_interp[i, :] if self.pad==False else self.P_k_nonlinear[i, :], color = 'tab:blue', label=('Models 000-049' if i==0 else None))
            plt.plot(self.k_test, self.P_k_interp[i+50, :] if self.pad==False else self.P_k_nonlinear[i+50, :], color = 'tab:orange', label=('Models 050-099' if i==0 else None))
            plt.plot(self.k_test, self.P_k_interp[i+100, :] if self.pad==False else self.P_k_nonlinear[i+100, :], color = 'tab:green', label=('Models 100-149' if i==0 else None))

            ###Plotting function broken for now###
            #if self.extrap == False:
            #print(max(self.P_k_class_linear[i, :, 0]), max(self.P_k_class_linear[i+50, :, 0]), max(self.P_k_class_linear[i+100, :, 0]))
            u_0=0
            u_50=0
            u_100=0
            for t in self.k[i, :]:
                if t <= 30:
                    u_0+=1
            for t in self.k[i+50, :]:
                if t <= 30:
                    u_50+=1
            for t in self.k[i+100, :]:
                if t <= 30:
                    u_100+=1
                        #k_plot = self.k[i, :u]
                #e = interpolate.interp1d(self.P_k_class_linear[i, :, 0], self.P_k_class_linear[i, :, 1], kind='cubic')
                #p = e(k_plot)
                ######################################
            #elif self.extrap == True:
                #u=self.bins

            plt.figure(1)
            if self.lin != 'class':
                plt.plot(self.k[i, :], self.P_k[i, :], color='tab:blue', label=('Models 000-049' if i==0 else None))
                plt.plot(self.k[i+50, :], self.P_k[i+50, :], color='tab:orange', label=('Models 050-099' if i==0 else None))
                plt.plot(self.k[i+100, :], self.P_k[i+100, :], color='tab:green', label=('Models 100-149' if i==0 else None))
            else:
                l_0 = interpolate.interp1d(self.P_k_class_linear[i, :, 0], self.P_k_class_linear[i, :, 1], kind='cubic')
                l_50 = interpolate.interp1d(self.P_k_class_linear[i+50, :, 0], self.P_k_class_linear[i+50, :, 1], kind='cubic')
                l_100 = interpolate.interp1d(self.P_k_class_linear[i+100, :, 0], self.P_k_class_linear[i+100, :, 1], kind='cubic')
                lin_0 = l_0(self.k[i, :u_0])
                lin_50 = l_50(self.k[i+50, :u_50])
                lin_100 = l_100(self.k[i+100, :u_100])
                plt.plot(self.k[i, :], self.P_k[i, :]/lin_0, color='tab:blue', label=('Models 000-049' if i==0 else None))
                plt.plot(self.k[i+50, :], self.P_k[i+50, :]/lin_50, color='tab:orange', label=('Models 050-099' if i==0 else None))
                plt.plot(self.k[i+100, :], self.P_k[i+100, :]/lin_100, color='tab:green', label=('Models 100-149' if i==0 else None))

            plt.figure(2)
            k_fund_low = (2*np.pi)/1400
            if self.lin != 'class':
                plt.plot(self.k[i, :]/(k_fund_low*2), self.P_k[i, :], color='tab:blue', label=('Models 000-049' if i==0 else None))
                plt.plot(self.k[i+50, :]/k_fund_low, self.P_k[i+50, :], color='tab:orange', label=('Models 050-099' if i==0 else None))
                plt.plot(self.k[i+100, :]/(k_fund_low*4), self.P_k[i+100, :], color='tab:green', label=('Models 100-149' if i==0 else None))
            else:
                l_0 = interpolate.interp1d(self.P_k_class_linear[i, :, 0], self.P_k_class_linear[i, :, 1], kind='cubic')
                l_50 = interpolate.interp1d(self.P_k_class_linear[i+50, :, 0], self.P_k_class_linear[i+50, :, 1], kind='cubic')
                l_100 = interpolate.interp1d(self.P_k_class_linear[i+100, :, 0], self.P_k_class_linear[i+100, :, 1],kind='cubic')
                lin_0 = l_0(self.k[i, :u_0])
                lin_50 = l_50(self.k[i+50, :u_50])
                lin_100 = l_100(self.k[i+100, :u_100])
                plt.plot(self.k[i, :]/(k_fund_low*2), self.P_k[i, :]/lin_0, color='tab:blue', label=('Models 000-049' if i==0 else None))
                plt.plot(self.k[i+50, :]/k_fund_low, self.P_k[i+50, :]/lin_50, color='tab:orange', label=('Models 050-099' if i==0 else None))
                plt.plot(self.k[i+100, :]/(k_fund_low*4), self.P_k[i+100, :]/lin_100, color='tab:green', label=('Models 100-149' if i==0 else None))

        plt.figure(1)
        plt.title(r'Boost function vs $k$ for all models', fontsize=10, wrap=True)
        plt.xlabel(r'$k \: [1/Mpc]$')
        plt.ylabel(r'$P(k) \: [Mpc^3]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('./Plots/Classbased_BOOSTVK.pdf', dpi=800)
        plt.clf()
        
        plt.figure(2)
        plt.title(r'Boost function vs $k$/$k_{fundamental}$ for all models', fontsize=10, wrap=True)
        plt.xlabel(r'$k \: [1/Mpc]$')
        plt.ylabel(r'$P(k) \: [Mpc^3]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('./Plots/Classbased_BOOSTVKFUND.pdf', dpi=800)
        plt.clf()
        
        plt.figure(3)
        plt.title(r'$P(k)$ vs test_k for all models', fontsize=10, wrap=True)
        plt.xlabel(r'$k \: [1/Mpc]$')
        plt.ylabel(r'$P(k) \: [Mpc^3]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('./Plots/Classbased_TESTCASE.pdf', dpi=800)
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

        
class FLAMINGO_DMO_Pk:

    def __init__(self, BXL_data, bins, cutoff = (0.025, 2.45), cosmo = np.arange(9), lin = 'class', boost = True, extrap = False, add_pade = False, pad = False):
        self.test_models = 'nan'
        self.bins = bins
        self.extrap = extrap
        self.pad = pad
        
        #The parameters file
        self.parameters = np.loadtxt('./BXL_data/FLAMINGO_params.txt', skiprows=1, usecols=cosmo)

        #Normalize the design to be between 0,1
        self.design_max = BXL_data.design_max
        self.design_min = BXL_data.design_min
        self.parameters_norm = (self.parameters-self.design_min)/(self.design_max-self.design_min)

        #array_size = 504

        #self.k = np.zeros([8, array_size])
        #self.P_k = np.zeros([8, array_size])
        self.k = []
        self.P_k = []
        self.P_k_interp = np.zeros([8, self.bins])
        self.P_k_nonlinear = np.zeros([8, self.bins])

        self.low_k_cut = cutoff[0]
        self.high_k_cut = cutoff[1]
        if self.low_k_cut == self.high_k_cut:
            self.k_test = BXL_data.k_test
            self.P_k_interp = np.zeros([8, len(self.k_test)])
            self.P_k_nonlinear = np.zeros([8, len(self.k_test)])
        else:
            self.k_test = np.logspace(np.log10(self.low_k_cut), np.log10(self.high_k_cut), self.bins)
            self.P_k_boost = np.zeros([8, len(BXL_data.k_test)])

        if lin == 'camb':
            self.k_camb_linear = np.loadtxt('./BXL_data/FLAMINGO_camb_pk.txt', usecols=0)
            self.Pk_camb_linear = np.loadtxt('./BXL_data/FLAMINGO_camb_pk.txt', usecols=1)
        #elif lin == 'class':
            #self.Pk_class_linear = np.zeros((150, 400, 2))
        elif lin == 'rebin':
            pass
        
        #This is used to build up k, the full P(k), linear P(k) and non-linear P(k)
        for i in range(8):
            
            #print('model = ' + str(i))
            if i == 0:
                boxsize = 1000
                n = 3600
            elif i == 1:
                boxsize = 2800
                n = 5040
            elif i == 2:
                boxsize = 5600
                n = 5040
            elif i == 3:
                boxsize = 11200
                n = 5040
            elif i == 4:
                boxsize = 1000
                n = 1800
            elif i == 5:
                boxsize = 400
                n = 720
            elif i == 6:
                boxsize = 200
                n = 720
            elif i == 7:
                boxsize = 1000
                n = 900

            #self.k[i, :] = np.loadtxt('./BXL_data/power_matter_L' + f"{boxsize:04d}" + 'N' + f"{n:04d}" + '_DMO_z0.txt', skiprows = 16, usecols = 1)
            k = np.loadtxt('./BXL_data/power_matter_L' + f"{boxsize:04d}" + 'N' + f"{n:04d}" + '_DMO_z0.txt', skiprows = 16, usecols = 1)
            self.k.append(k)
            if self.low_k_cut == -1:
                self.k_test = self.k[i, :]
                self.P_k_interp = np.zeros([8, len(self.k_test)])
                self.P_k_nonlinear = np.zeros([8, len(self.k_test)])
                
            #self.P_k[i, :] = np.loadtxt('./BXL_data/power_matter_L' + f"{boxsize:04d}" + 'N' + f"{n:04d}" + '_DMO_z0.txt', skiprows = 16, usecols = 2)
            P_k = np.loadtxt('./BXL_data/power_matter_L' + f"{boxsize:04d}" + 'N' + f"{n:04d}" + '_DMO_z0.txt', skiprows = 16, usecols = 2)
            self.P_k.append(P_k)
            
            if add_pade == False:
                h = interpolate.interp1d(k, P_k, kind='cubic', fill_value="extrapolate" if extrap == True else 'nan')
            elif add_pade == True:
                print(k[-4:-1], P_k[-4:-1])
                h = pade.fit(k[-4:-1], P_k[-4:-1])
                print(self.k_test[-4:-1], h(self.k_test))

            for u,g in enumerate(self.k_test):
                if g < min(k):
                    self.P_k_interp[i, u] = -1
                else:
                    self.P_k_interp[i, u] = h(self.k_test[u])

            if lin == 'camb':
                k_linear = self.k_camb_linear*(self.parameters[2])
                Pk_linear = self.Pk_camb_linear/(self.parameters[2]**3)
                
                f = interpolate.interp1d(k_linear, Pk_linear, kind='cubic')
                y = f(self.k_test)

            elif lin == 'class':
                self.Pk_class_linear = np.loadtxt('/mnt/data1/users/arijsalc/BAHAMAS_XL/DMO/model_000_N1260_L700_DMO/class_linear_spectra_z_0.txt', skiprows=2, usecols=(0, 2))
                
                f = interpolate.interp1d(self.Pk_class_linear[0], self.Pk_class_linear[1], kind='cubic')
                y = f(self.k_test)
                
            
            if boost == True:
                if self.low_k_cut == -1:
                    for q,p in enumerate(self.P_k_interp[i, :]):
                        if p == -1:
                            self.P_k_nonlinear[i, q] = 1
                        else:
                            self.P_k_nonlinear[i, q] = self.P_k_interp[i, q] / y[q]
                else:
                    self.P_k_nonlinear[i, :] = self.P_k_interp[i, :] / y
            elif boost == False:
                self.P_k_nonlinear[i, :] = self.P_k_interp[i, :]
                    
            if pad == True:
                for f in range(8):
                    pade_func((3, 8, 13), self.k_test, self.P_k_nonlinear[f, :], (2, 15, 15), self.k_test, self.P_k_nonlinear[f, :])
                    pade_func((2, 7, 12), self.k_test, self.P_k_nonlinear[f+50, :], (1, 14, 15), self.k_test, self.P_k_nonlinear[f+50, :])
                    pade_func((4, 8, 13), self.k_test, self.P_k_nonlinear[f+100, :], (3, 15, 15), self.k_test, self.P_k_nonlinear[f+100, :])

            v = interpolate.interp1d(self.k_test, self.P_k_nonlinear[i, :], kind='cubic')
            for h,j in enumerate(BXL_data.k_test):
                if j < min(self.k_test):
                    self.P_k_boost[i, h] = 1
                else:
                    self.P_k_boost[i, h] = v(BXL_data.k_test[h])

            for a,b in enumerate(self.P_k_boost[i, :]):
                if b < 0:
                    self.P_k_boost[i, a] = 1
            
        #print(self.P_k_nonlinear[50, :])
        self.k_test = BXL_data.k_test
        self.X_train = BXL_data.X_train
        self.X_test = self.parameters_norm
        self.Y_train = BXL_data.Y_train
        if self.low_k_cut == -1:
            self.Y_test = self.P_k_nonlinear
        else:
            self.Y_test = self.P_k_boost


if __name__ == "__main__":
    bins = 100
    #cutoff = (np.pi*1260)/1400

    #test_model = [3, 12, 18, 90, 138]
    test_models = random.randint(0, 149)
    #test_models = 50
    #test_model = 11
    
    nbk_boost = BXL_DMO_Pk(test_models, bins, pk = 'nbk-rebin', lin = 'class', holdout=False)
    quit()
    flamingo = FLAMINGO_DMO_Pk(nbk_boost, bins, cutoff=(.01,10), lin='camb')
    print(nbk_boost.k)
    print(nbk_boost.P_k)
    print(nbk_boost.k_test)
    print(nbk_boost.test_models)
    print(nbk_boost.X_train)
    print(nbk_boost.Y_train)
    #print(nbk_boost.X_test)
    #print(nbk_boost.Y_test)
    nbk_boost.plot_k()

    print(flamingo.X_test)
    print(flamingo.Y_test)

    print(flamingo.k[6])
    print(flamingo.P_k[6])
    print(flamingo.P_k_interp[6, :])
    print(flamingo.P_k_nonlinear[6, :])
    print(flamingo.P_k_boost[6, :])
    
    plt.plot(flamingo.k_test, flamingo.Y_test[0, :], label='1Gpc (High res)')
    plt.plot(flamingo.k_test, flamingo.Y_test[1, :], label='2.8Gpc')
    plt.plot(flamingo.k_test, flamingo.Y_test[2, :], label='5.6Gpc')
    plt.plot(flamingo.k_test, flamingo.Y_test[3, :], label='11.2Gpc')
    plt.plot(flamingo.k_test, flamingo.Y_test[4, :], label='1Gpc (Intermediate res)')
    plt.plot(flamingo.k_test, flamingo.Y_test[5, :], label='400Mpc (Intermediate res)')
    plt.plot(flamingo.k_test, flamingo.Y_test[6, :], label='200Mpc (High res)')
    plt.plot(flamingo.k_test, flamingo.Y_test[7, :], label='1Gpc (Low res)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'Boost function vs $k$ for FLAMINGO models', fontsize=10, wrap=True)
    plt.xlabel(r'$k \: [1/Mpc]$')
    plt.ylabel(r'$P(k) \: [Mpc^3]$')
    plt.legend()
    plt.savefig('./Plots/flam.pdf', dpi=800)
    plt.clf()

    print('done')

    plt.plot(flamingo.k_test, (flamingo.Y_test[3, :]/flamingo.Y_test[0, :]), label='1Gpc (High res)')
    plt.plot(flamingo.k_test, (flamingo.Y_test[3, :]/flamingo.Y_test[1, :]), label='2.8Gpc')
    plt.plot(flamingo.k_test, (flamingo.Y_test[3, :]/flamingo.Y_test[2, :]), label='5.6Gpc')
    plt.plot(flamingo.k_test, (flamingo.Y_test[3, :]/flamingo.Y_test[3, :]), label='11.2Gpc')
    plt.plot(flamingo.k_test, (flamingo.Y_test[3, :]/flamingo.Y_test[4, :]), label='1Gpc (Intermediate res)')
    plt.plot(flamingo.k_test, (flamingo.Y_test[3, :]/flamingo.Y_test[5, :]), label='400Mpc (Intermediate res)')
    plt.plot(flamingo.k_test, (flamingo.Y_test[3, :]/flamingo.Y_test[6, :]), label='200Mpc (High res)')
    plt.plot(flamingo.k_test, (flamingo.Y_test[3, :]/flamingo.Y_test[7, :]), label='1Gpc (Low res)')
    plt.xscale('log')
    #plt.yscale('log')
    plt.title('FLAMINGO models relative to the lowest resolution run (11.2Gpc)', fontsize=10, wrap=True)
    plt.xlabel(r'$k \: [1/Mpc]$')
    plt.ylabel(r'$P_{11.2}(k)/P_{i}(k)$')
    plt.legend()
    plt.savefig('./Plots/flam_comp.pdf', dpi=800)
    plt.clf()
                                                    
    plt.plot(flamingo.k_test, (flamingo.Y_test[4, :]/flamingo.Y_test[0, :]), label='1Gpc (High res)')
    plt.plot(flamingo.k_test, (flamingo.Y_test[4, :]/flamingo.Y_test[1, :]), label='2.8Gpc')
    plt.plot(flamingo.k_test, (flamingo.Y_test[4, :]/flamingo.Y_test[2, :]), label='5.6Gpc')
    #plt.plot(flamingo.k_test, (flamingo.Y_test[4, :]/flamingo.Y_test[3, :]), label='11.2Gpc')
    plt.plot(flamingo.k_test, (flamingo.Y_test[4, :]/flamingo.Y_test[4, :]), label='1Gpc (Intermediate res)')
    plt.plot(flamingo.k_test, (flamingo.Y_test[4, :]/flamingo.Y_test[5, :]), label='400Mpc (Intermediate res)')
    plt.plot(flamingo.k_test, (flamingo.Y_test[4, :]/flamingo.Y_test[6, :]), label='200Mpc (High res)')
    plt.plot(flamingo.k_test, (flamingo.Y_test[4, :]/flamingo.Y_test[7, :]), label='1Gpc (Low res)')
    plt.xscale('log')
    #plt.yscale('log')
    plt.title('FLAMINGO models relative to the fiducial run (1Gpc med res)', fontsize=10, wrap=True)
    plt.xlabel(r'$k \: [1/Mpc]$')
    plt.ylabel(r'$P_{fiducial}(k)/P_{i}(k)$')
    plt.legend()
    plt.savefig('./Plots/flam_comp_2.pdf', dpi=800)
    plt.clf()

    plt.plot(nbk_boost.k_test, nbk_boost.Y_test)
    plt.xscale('log')
    #plt.yscale('log')
    plt.title('FLAMINGO models relative to the fiducial run (1Gpc med res)', fontsize=10, wrap=True)
    plt.xlabel(r'$k \: [1/Mpc]$')
    plt.ylabel(r'$P_{fiducial}(k)/P_{i}(k)$')
    plt.legend()
    plt.savefig('./Plots/test.pdf', dpi=800)
    plt.clf()
