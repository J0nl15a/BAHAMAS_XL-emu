import numpy as np
import pylab as pb
from scipy import interpolate
import random
import pandas as pd
from pade_func import pade_func
from hypercube import hypercube_plot
from gpy_emu_extension import gpyHighResEmulator, gpyLowResEmulator
import glob
import re


class bahamasXLDMOData:

    def __init__(self, test_cosmology=False, bins=100, cutoff=(-1, 10), cosmo_params=np.arange(9), resolution='HR', pk='powmes', lin='camb', boost=True, log=True, holdout=False, sigma8=True, plot_hypercube=False, weight_k=True):

        self.test_cosmology = test_cosmology
        self.bins = bins
        self.pk = pk
        self.lin = lin
        self.modes = {}
        self.low_k_cut = cutoff[0]
        self.high_k_cut = cutoff[1]
        self.holdout = holdout
        self.log = log

        
        #The parameters file
        self.parameters = np.loadtxt('./BXL_data/slhs_nested_3x50_w0_m0p6_m1p2_wa_m1p6_p0p5_with_running_and_fgas_and_As.txt', skiprows=1, max_rows=150, usecols=cosmo_params)
        if sigma8 == True:
            self.As = self.parameters[:,4].copy()
            sig8 = np.loadtxt('./BXL_data/slhs_nested_3x_Om_fb_h_ns_sigma8_w0_wa_Omnuh2_alphas_fgasfb.txt', max_rows=150, usecols=4)
            self.parameters[:,4] = sig8

        self.particle_masses = np.loadtxt('./BXL_data/DM_masses.txt').reshape(-1,1)
        self.parameters = np.hstack((self.parameters, self.particle_masses))

        
        if plot_hypercube == True:
            labels = {r'$\Omega_m$':[.20,.25,.30,.35], r'$f_b$':[.14,.15,.16,.17], r'$h_0$':[.64,.7,.76], r'$n_s$':[.95,.97,.99], r'$\sigma_8$':[0.72,0.80,0.88], r'$w_0$':[-.7,-.9,-1.1], r'$w_a$':[.2,-.5,-1.2], r'$\Omega_{\nu}h^2$':[.005,.003,.001], r'$\alpha_s$':[.025,0,-.025]}#, r'$M_{\odot}$':[1e+9, 1e+10, 1e+11]}
            
            hypercube = hypercube_plot(data=[self.parameters[:50, :], self.parameters[50:100, :], self.parameters[100:, :]], parameter_labels=labels, save_to='./Plots/BXL_hypercube.png', title='BAHAMAS XL Latin hypercube design')


        self.correction_factor = [1.2761424, 1.1154015, 1.0447197, 1.0151449, 1.0195166, 1.0112120]

                    
        def extract_series(group):
            return {'k': group['k'].values, 'Pk': group['boost'].values, 'model': group['model'].iloc[0], 'modes': group['N_modes'].values if pk == 'nbk-rebin-std' else None}

        def extract_model_number(path):
            # Use regex to extract the number between 'model_' and '_N1260'
            match = re.search(r'model_(\d+)_N1260', path)
            if match:
                return int(match.group(1))  # Return the number as an integer
            return float('inf')  # If no match, return a large number to push it to the end (or handle as needed)

                
        if pk == 'nbk-rebin' or pk == 'nbk-rebin-std':
            if pk == 'nbk-rebin':
                df = pd.read_csv("./BXL_data/boost_rebinned_21.csv")
            elif pk == 'nbk-rebin-std':
                df = pd.read_csv("./BXL_data/boost_rebinned_standardized.csv")
                
            grouped = df.groupby('model')
            arrays_per_model = grouped.apply(extract_series)
            self.array_size = len(arrays_per_model[0]['k'])
            self.bins = self.array_size

            if pk == 'nbk-rebin-std':
                self.modes['Med res'] = arrays_per_model[0]['modes'].tolist()
                self.modes['Low res'] = arrays_per_model[50]['modes'].tolist()
                self.modes['High res'] = arrays_per_model[100]['modes'].tolist()

        else:
            random_high_res_int = random.randint(100, 149)
            if pk == 'nbk':
                random_high_res_model = np.loadtxt(glob.glob(f'/mnt/aridata1/users/arijsalc/BAHAMAS_XL/DMO/model_{random_high_res_int:03d}_N1260_L*_DMO/PS/PS_k_078.csv')[0], skiprows = 1, usecols = 0, delimiter=' ')*self.parameters[random_high_res_int, 2]
                self.array_size = len(random_high_res_model)
            elif pk == 'powmes':
                random_high_res_model = np.loadtxt(glob.glob(f'/mnt/aridata1/users/arijsalc/BAHAMAS_XL/DMO/model_{random_high_res_int:03d}_N1260_L*_DMO/power_spectra/power_matter_0122.txt')[0], skiprows = 20, usecols = (1,2,3))
                random_low_res_int = random.randint(50, 99)
                random_low_res_model = np.loadtxt(glob.glob(f'/mnt/aridata1/users/arijsalc/BAHAMAS_XL/DMO/model_{random_low_res_int:03d}_N1260_L*_DMO/power_spectra/power_matter_0122.txt')[0], skiprows = 20, usecols = 1)
                for x in range(len(random_high_res_model)):
                    if random_high_res_model[x,1] < random_high_res_model[x,2]:
                        self.array_size = x-1
                        break

                    
        if pk == 'nbk-rebin-std':
            self.k_test = arrays_per_model[0]['k']
        else:
            if pk == 'nbk-rebin':
                minimum_k = min(arrays_per_model[100]['k'])
                maximum_k = max(arrays_per_model[50]['k'])
            elif pk == 'nbk':
                minimum_k = min(random_high_res_model if random_high_res_int in range(100, 150) else (np.loadtxt(glob.glob(f'/mnt/aridata1/users/arijsalc/BAHAMAS_XL/DMO/model_{100:03d}_N1260_L*_DMO/PS/PS_k_078.csv')[0], skiprows = 1, usecols= 0, delimiter=' ')*self.parameters[100, 2]))*1.01
                maximum_k = max(random_high_res_model if random_high_res_int in range(50, 100) else (np.loadtxt(glob.glob(f'/mnt/aridata1/users/arijsalc/BAHAMAS_XL/DMO/model_{50:03d}_N1260_L*_DMO/PS/PS_k_078.csv')[0], skiprows = 1, usecols= 0, delimiter=' ')*self.parameters[50, 2]))*0.99
            elif pk == 'powmes':
                if weight_k == True:
                    random_high_res_model[:6, 0] = random_high_res_model[:6, 0]*self.correction_factor
                    random_low_res_model[:6] = random_low_res_model[:6]*self.correction_factor
                minimum_k = min(random_high_res_model[:,0])
                maximum_k = max(random_low_res_model)

            if self.low_k_cut == -1:
                self.low_k_cut = minimum_k

            try:
                print(self.low_k_cut, minimum_k)
                assert self.low_k_cut >= minimum_k
            except AssertionError:
                print('self.low_k_cut > minimum_k == False')
                self.low_k_cut = minimum_k

            try:
                print(self.high_k_cut, maximum_k)
                assert self.high_k_cut <= maximum_k
            except AssertionError:
                print('self.high_k_cut < maximum_k == False')
                self.high_k_cut = maximum_k

            print(self.low_k_cut, self.high_k_cut)
            self.k_test = np.logspace(np.log10(self.low_k_cut), np.log10(self.high_k_cut), self.bins)

            
        if self.lin == 'camb':
            self.k_camb_linear = np.tile(np.logspace(-3, np.log10(50), 300), (150,1))
            self.P_k_camb_linear = np.loadtxt('./BXL_data/pk_lin_camb2022_slhs_nested_3x50_kmax_50_running_w0_m1p2_m0p6_wa_m1p6_p0p5.txt')

        elif self.lin == 'class':
            class_directory = glob.glob('/mnt/aridata1/users/arijsalc/BAHAMAS_XL/DMO/model_*_N1260_L*_DMO/class_linear_spectra_z_0.txt')
            sorted_class_directory = sorted(class_directory, key=extract_model_number)
            for index, directory in enumerate(sorted_class_directory):
                if index == 0:
                    linear_spectra = np.loadtxt(directory, skiprows=2, usecols=(0,2))
                    linear_size = len(linear_spectra)
                    self.k_class_linear = np.zeros([150, linear_size])
                    self.P_k_class_linear = np.zeros([150, linear_size])
                self.k_class_linear[index, :] = np.loadtxt(directory, skiprows=2, usecols=0)
                self.P_k_class_linear[index, :] = np.loadtxt(directory, skiprows=2, usecols=2)

        elif self.lin == 'rebin':
            pass

        
        #This is used to build up k, the full P(k), linear P(k) and non-linear P(k)
        self.k = np.zeros([150, self.array_size])
        self.P_k = np.zeros([150, self.array_size])
        self.P_k_interp = np.zeros([150, len(self.k_test)])
        self.P_k_nonlinear = np.zeros([150, len(self.k_test)])

                                
        if pk == 'powmes':
            self.noise = np.zeros([150, self.array_size])
            training_directory = glob.glob('/mnt/aridata1/users/arijsalc/BAHAMAS_XL/DMO/model_*_N1260_L*_DMO/power_spectra/power_matter_0122.txt')
            sorted_training_directory = sorted(training_directory, key=extract_model_number)
            for index, directory in enumerate(sorted_training_directory):
                self.k[index, :] = np.loadtxt(directory, skiprows = 20, usecols = 1)[:self.array_size]
                self.P_k[index, :] = np.loadtxt(directory, skiprows = 20, usecols = 2)[:self.array_size]
                self.noise[index, :] = np.loadtxt(directory, skiprows = 20, usecols = 3)[:self.array_size]
                if weight_k == True:
                    self.k[index, :6] = self.k[index, :6]*self.correction_factor

        elif pk == 'nbk':
            training_directory = glob.glob('/mnt/aridata1/users/arijsalc/BAHAMAS_XL/DMO/model_*_N1260_L*_DMO/PS/PS_k_078.csv')
            sorted_training_directory = sorted(training_directory, key=extract_model_number)
            for index, directory in enumerate(sorted_training_directory):
                self.k[index, :] = np.loadtxt(directory, skiprows = 1, usecols = 0, delimiter=' ')*(self.parameters[index, 2])
                self.P_k[index, :] = np.loadtxt(directory, skiprows = 1, usecols = 1, delimiter=' ')/(self.parameters[index, 2]**3)
                self.modes[index] = np.loadtxt(directory, skiprows = 1, usecols = 2, delimiter=' ')

        elif pk == 'nbk-rebin' or 'nbk-rebin-std':
            for model in arrays_per_model:
                self.k[model['model'], :] = model['k']
                self.P_k[model['model'], :] = model['Pk']
                

        if pk == 'nbk-rebin-std':
            self.P_k_interp = self.P_k.copy()
        else:
            interpolation_function = {}
            for i in range(len(self.k)):
                interpolation_function[i] = interpolate.interp1d(self.k[i,:].copy(), self.P_k[i,:].copy(), kind='cubic')
                self.P_k_interp[i, :] = interpolation_function[i](self.k_test)


        self.linear_interpolation_function = {}
        if self.lin == 'camb':
            self.k_linear = self.k_camb_linear*(self.parameters[:, 2].reshape(-1,1))
            self.P_k_linear = self.P_k_camb_linear/(self.parameters[:, 2].reshape(-1,1))**3
        elif self.lin == 'class':
            self.k_linear = self.k_class_linear#*(self.parameters[:, 2].reshape(-1,1))
            self.P_k_linear = self.P_k_class_linear#/(self.parameters[:, 2].reshape(-1,1))**3

        if self.lin == 'rebin':
            self.P_k_linear_interp = np.ones([150, self.P_k_interp.shape[1]])
        else:
            self.P_k_linear_interp = np.zeros([150, self.P_k_interp.shape[1]])
            for l in range(len(self.k_linear)):
                self.linear_interpolation_function[l] = interpolate.interp1d(self.k_linear[l,:], self.P_k_linear[l,:], kind='cubic', bounds_error=True)
                self.P_k_linear_interp[l,:] = self.linear_interpolation_function[l](self.k_test)


        if pk == 'nbk-rebin' or pk == 'nbk-rebin-std':
            self.P_k_nonlinear = self.P_k_interp.copy()
        else:
            if boost == True:
                self.P_k_nonlinear = self.P_k_interp.copy() / self.P_k_linear_interp.copy()
            elif boost == False:
                self.P_k_nonlinear = self.P_k_interp.copy()

        self.nan_mask = np.isnan(self.P_k_nonlinear)
        '''try:
            logged_array = np.log10(self.P_k_nonlinear)
        except ValueError:
            self.P_k_nonlinear[self.nan_mask] = 0'''

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.normalise()
        print(self.X_train, self.X_test, self.Y_train)
            

        return
            

    def normalise(self, X=True, Y=True, undo=False, k=False):
        print('NORMALISE!')

        if isinstance(undo, bool):
            if X == True:
                #Normalize the design to be between 0,1
                self.design_max = np.max(self.parameters, axis=0)
                self.design_min = np.min(self.parameters, axis=0)
                self.parameters_norm = (self.parameters.copy()-self.design_min)/(self.design_max-self.design_min)
            
                #Split into test and train data
                if isinstance(self.holdout, bool):
                    self.X_train = self.parameters_norm.copy()
                    if not isinstance(self.test_cosmology, bool):
                        self.X_test = (self.test_cosmology.copy()-self.design_min)/(self.design_max-self.design_min)
                    else:
                        self.X_test = (self.flamingo_parameters.copy()-self.design_min)/(self.design_max-self.design_min)
                else:
                    self.X_test = self.parameters_norm[self.holdout, :].copy()
                    self.X_train = np.delete(self.parameters_norm.copy(), self.holdout, axis=0)

            if Y == True:

                if self.log == True:
                    self.P_k_nonlinear_norm = np.log10(self.P_k_nonlinear.copy())
                else:
                    self.P_k_nonlinear_norm = self.P_k_nonlinear.copy()
                
                #Normalise to 0 mean and variance of 1
                print(self.P_k_nonlinear_norm)
                self.mean_Pk = np.mean(self.P_k_nonlinear_norm, axis=0)
                print(self.mean_Pk)
                self.P_k_nonlinear_norm -= self.mean_Pk
                print(self.P_k_nonlinear_norm)
                self.std_Pk = np.std(self.P_k_nonlinear_norm, axis=0)
                print(self.std_Pk)
                self.P_k_nonlinear_norm /= self.std_Pk
                print(self.P_k_nonlinear_norm)
                
                if isinstance(self.holdout, bool):
                    self.Y_test = np.ones((len(self.k_test)))
                    self.Y_train = self.P_k_nonlinear_norm.copy()
                else:
                    self.Y_test = self.P_k_nonlinear[self.holdout, :].copy()
                    self.Y_train = np.delete(self.P_k_nonlinear_norm.copy(), self.holdout, axis=0)

            return self.X_train, self.X_test, self.Y_train, self.Y_test

                
        elif not isinstance(undo, bool):
            if isinstance(k, bool):
                return (undo*self.std_Pk)+self.mean_Pk
            else:
                return (undo*self.std_Pk[k])+self.mean_Pk[k]
        
        return

    def plot_k(self, file_type='png'):

        
        if self.lin == 'camb' or self.lin == 'class':
            linear_limit = []
            k_limit = 30            
            for i in range(150):
                for index, k in enumerate(self.k[i, :]):
                    if k >= k_limit:
                        linear_limit.append(index)
                        break
        else:
            linear_limit = [self.array_size+1 for _ in range(150)]

        for i in range(50):
            pb.figure(3)
            pb.plot(self.k_test, self.P_k_nonlinear[i, :], color ='tab:blue', label=('Padded data' if i==0 else None), linestyle='dashed')
            pb.plot(self.k_test, self.P_k_nonlinear[i+50, :], color = 'tab:orange', linestyle='dashed')
            pb.plot(self.k_test, self.P_k_nonlinear[i+100, :], color = 'tab:green', linestyle='dashed')
            
            pb.plot(self.k_test[~self.nan_mask[i, :]], self.P_k_nonlinear[i, :] if np.any(self.nan_mask)==False else self.P_k_nonlinear[i, :][~self.nan_mask[i, :]], color ='tab:blue', label=('Models 000-049' if i==0 else None))
            pb.plot(self.k_test[~self.nan_mask[i+50, :]], self.P_k_nonlinear[i+50, :] if np.any(self.nan_mask)==False else self.P_k_nonlinear[i+50, :][~self.nan_mask[i+50, :]], color = 'tab:orange', label=('Models 050-099' if i==0 else None))
            pb.plot(self.k_test[~self.nan_mask[i+100, :]], self.P_k_nonlinear[i+100, :] if np.any(self.nan_mask)==False else self.P_k_nonlinear[i+100, :][~self.nan_mask[i+100, :]], color = 'tab:green', label=('Models 100-149' if i==0 else None))
 
            pb.figure(1)
            pb.plot(self.k[i, :linear_limit[i]], self.P_k[i, :linear_limit[i]]/self.linear_interpolation_function[i](self.k[i, :linear_limit[i]]) if self.lin!='rebin' else self.P_k[i, :linear_limit[i]], color='tab:blue', label=('Models 000-049' if i==0 else None))
            pb.plot(self.k[i+50, :linear_limit[i+50]], self.P_k[i+50, :linear_limit[i+50]]/self.linear_interpolation_function[i+50](self.k[i+50, :linear_limit[i+50]]) if self.lin!='rebin' else self.P_k[i+50, :linear_limit[i+50]], color='tab:orange', label=('Models 050-099' if i==0 else None))
            pb.plot(self.k[i+100, :linear_limit[i+100]], self.P_k[i+100, :linear_limit[i+100]]/self.linear_interpolation_function[i+100](self.k[i+100, :linear_limit[i+100]]) if self.lin!='rebin' else self.P_k[i+100, :linear_limit[i+100]], color='tab:green', label=('Models 100-149' if i==0 else None))
            
            pb.figure(2)
            k_fund_low = (2*np.pi)/1400
            pb.plot(self.k[i, :linear_limit[i]]/(k_fund_low*2), self.P_k[i, :linear_limit[i]]/self.linear_interpolation_function[i](self.k[i, :linear_limit[i]]) if self.lin!='rebin' else self.P_k[i, :linear_limit[i]], color='tab:blue', label=('Models 000-049' if i==0 else None))
            pb.plot(self.k[i+50, :linear_limit[i+50]]/k_fund_low, self.P_k[i+50, :linear_limit[i+50]]/self.linear_interpolation_function[i+50](self.k[i+50, :linear_limit[i+50]]) if self.lin!='rebin' else self.P_k[i+50, :linear_limit[i+50]], color='tab:orange', label=('Models 050-099' if i==0 else None))
            pb.plot(self.k[i+100, :linear_limit[i+100]]/(k_fund_low*4), self.P_k[i+100, :linear_limit[i+100]]/self.linear_interpolation_function[i+100](self.k[i+100, :linear_limit[i+100]]) if self.lin!='rebin' else self.P_k[i+100, :linear_limit[i+100]], color='tab:green', label=('Models 100-149' if i==0 else None))

        pb.figure(1)
        pb.title(r'Boost function vs $k$ for all models', fontsize=10, wrap=True)
        pb.xlabel(r'$k \: [1/Mpc]$')
        pb.ylabel(r'$P(k) \: [Mpc^3]$')
        pb.xscale('log')
        pb.yscale('log')
        pb.legend()
        pb.savefig(f'./Plots/Classbased_BOOSTVK.{file_type}', dpi=800)
        pb.clf()
        
        pb.figure(2)
        pb.title(r'Boost function vs $k$/$k_{fundamental}$ for all models', fontsize=10, wrap=True)
        pb.xlabel(r'$k \: [1/Mpc]$')
        pb.ylabel(r'$P(k) \: [Mpc^3]$')
        pb.xscale('log')
        pb.yscale('log')
        pb.legend()
        pb.savefig(f'./Plots/Classbased_BOOSTVKFUND.{file_type}', dpi=800)
        pb.clf()
        
        pb.figure(3)
        pb.title(r'$P(k)$ vs test_k for all models', fontsize=10, wrap=True)
        pb.xlabel(r'$k \: [1/Mpc]$')
        pb.ylabel(r'$P(k) \: [Mpc^3]$')
        pb.xscale('log')
        pb.yscale('log')
        pb.legend()
        pb.savefig(f'./Plots/Classbased_TESTCASE.{file_type}', dpi=800)
        pb.clf()
        return

    
    def extend_data(self, pad=False):

        #print(self.nan_mask)
        #print(self.nan_mask[0,:],self.nan_mask[50,:],self.nan_mask[100,:])

        if pad == 'emu':
            #self = gpyLowResEmulator(self).data
            #self = gpyHighResEmulator(self).data
            gpyLowResEmulator(self)
            gpyHighResEmulator(self)
            print('PADDED')
            
        elif pad == 'pade':
            for f in range(50):
                pade_func((3, 8, 13), self.k_test, self.P_k_nonlinear[f, :], (2, 15, 15), self.k_test, self.P_k_nonlinear[f, :])
                pade_func((2, 7, 12), self.k_test, self.P_k_nonlinear[f+50, :], (1, 14, 15), self.k_test, self.P_k_nonlinear[f+50, :])
                pade_func((4, 8, 13), self.k_test, self.P_k_nonlinear[f+100, :], (3, 15, 15), self.k_test, self.P_k_nonlinear[f+100, :])

        elif pad == False:
            pass
        
        return

    

if __name__ == "__main__":
    import pdb

    test_models = random.randint(0, 149)
    
    nbk_boost = bahamasXLDMOData(pk='powmes', lin='class', holdout=test_models, log=True)
    print(nbk_boost.P_k_nonlinear)
    nbk_boost.extend_data('emu')
    print(nbk_boost.P_k_nonlinear)
    print(nbk_boost.k)
    print(nbk_boost.P_k)
    print(nbk_boost.k_test)
    print(nbk_boost.holdout)
    print(nbk_boost.X_train)
    print(nbk_boost.Y_train)
    print(nbk_boost.X_test)
    print(nbk_boost.Y_test)
    nbk_boost.plot_k()
    quit()

    hr = random.randint(100, 149)
    ir = random.randint(0, 49)
    lr = random.randint(50, 99)

    for x in range(50):
        pb.plot(nbk_boost.k[x], nbk_boost.P_k[x], color='tab:blue', label='Intermediate Res' if x==0 else None)
        pb.plot(nbk_boost.k[x+50], nbk_boost.P_k[x+50], color='tab:orange', label='Low Res' if x==0 else None)
        pb.plot(nbk_boost.k[x+100], nbk_boost.P_k[x+100], color='tab:green', label='High Res' if x==0 else None)
        pb.plot(nbk_boost.k[x], nbk_boost.noise[x], color='tab:red', linestyle='dashed', label='Shot noise (IR)' if x==0 else None)
        pb.plot(nbk_boost.k[x+50], nbk_boost.noise[x+50], color='tab:purple', linestyle='dashed', label='Shot noise (LR)' if x==0 else None)
        pb.plot(nbk_boost.k[x+100], nbk_boost.noise[x+100], color='tab:brown', linestyle='dashed', label='Shot noise (HR)' if x==0 else None)

    pb.title(r'BAHAMAS XL non-linear spectra', fontsize=10, wrap=True)
    pb.xlabel(r'$k \: [1/Mpc]$')
    pb.ylabel(r'$P(k) \: [Mpc^3]$')
    pb.xscale('log')
    pb.yscale('log')
    pb.legend()
    pb.savefig(f'./Plots/BXL_raw_data.png', dpi=1200)
    pb.clf()

    for i in range(len(nbk_boost.X_train)):
        print(nbk_boost.X_train[i,:])
    print(nbk_boost.X_test)
