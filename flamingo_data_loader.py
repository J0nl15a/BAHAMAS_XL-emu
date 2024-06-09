import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import random
import pandas as pd
from pade_func import pade_func
from hypercube import hypercube_plot
from gpy_emu_extension import gpyHighResEmulator, gpyLowResEmulator
import glob
from data_loader import bahamasXLDMOData
import matplotlib.figure as fi

    
class flamingoDMOData(bahamasXLDMOData):

    def __init__(self, bins=100, test_cosmology=False, cutoff=(-1, 10), cosmo_params=np.arange(9), pk='nbk-rebin-std', lin='rebin', flamingo_lin='camb', boost=True, holdout=False, sigma8=False, plot_hypercube=False, weight_k=True):
        
        super().__init__(test_cosmology=test_cosmology, bins=bins, cutoff=cutoff, cosmo_params=cosmo_params, pk=pk, lin=lin, boost=boost, holdout=holdout, sigma8=sigma8, plot_hypercube=plot_hypercube, weight_k=weight_k)

        if pk=='nbk-rebin-std' or pk=='nbk-rebin':
            super().extend_data(pad='emu')
        else:
            pass
                
        #The parameters file
        self.flamingo_parameters = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_params.txt', skiprows=1, usecols=cosmo_params)

        #Normalize the design to be between 0,1
        self.flamingo_parameters_norm = (self.flamingo_parameters-self.design_min)/(self.design_max-self.design_min)
        self.X_test = self.flamingo_parameters_norm
        
        self.flamingo_k = []
        self.flamingo_P_k = []
        self.flamingo_P_k_interp = np.zeros([14, self.bins])
        self.flamingo_P_k_nonlinear = np.zeros([14, self.bins])

        if flamingo_lin == 'camb':
            self.flamingo_camb_linear = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_camb_pk.txt', usecols=(0,1))
            k_linear = self.flamingo_camb_linear[:,0]*(self.flamingo_parameters[2])
            Pk_linear = self.flamingo_camb_linear[:,1]/(self.flamingo_parameters[2]**3)
            
            flamingo_linear_interpolation_function = interpolate.interp1d(k_linear, Pk_linear, kind='cubic')
            self.flamingo_linear_spectra = flamingo_linear_interpolation_function(self.k_test)

        elif flamingo_lin == 'class':
            self.flamingo_class_linear = np.loadtxt('/mnt/aridata1/users/arijsalc/BAHAMAS_XL/DMO/model_000_N1260_L700_DMO/class_linear_spectra_z_0.txt', skiprows=2, usecols=(0,2))
        
            flamingo_linear_interpolation_function = interpolate.interp1d(self.flamingo_class_linear[:,0], self.flamingo_class_linear[:,1], kind='cubic')
            self.flamingo_linear_spectra = flamingo_linear_interpolation_function(self.k_test)
            
        elif lin == 'rebin':
            pass

        self.flamingo_sims = [(200,720), (350,315), (350,630), (350,1260), (350,2520), (400,720), (700,1260), (1000,900), (1000,1800), (1000,3600), (1400,2520), (2800,5040), (5600,5040), (11200,5040)]

        #This is used to build up k, the full P(k), linear P(k) and non-linear P(k)
        for s in range(len(self.flamingo_sims)):
            flamingo_directory = './BXL_data/FLAMINGO_data/power_matter_L'+f"{self.flamingo_sims[s][0]:04d}"+'N'+f"{self.flamingo_sims[s][1]:04d}"+'_DMO_z0.txt'

            try:
                flamingo_model = np.loadtxt(flamingo_directory, skiprows = 20, usecols = (1,2,3))
                for x in range(len(flamingo_model)):
                    if flamingo_model[x,1] < flamingo_model[x,2]:
                        array_size = x-1
                        break
            except ValueError:
                flamingo_model = np.loadtxt(flamingo_directory, skiprows = 20, usecols = (1,2))
                for x in range(len(flamingo_model)):
                    if flamingo_model[x,0] > 100:
                        array_size = x-1
                        break
            
            self.flamingo_k.append(flamingo_model[:array_size, 0])                
            self.flamingo_P_k.append(flamingo_model[:array_size, 1])
                        
            h = interpolate.interp1d(self.flamingo_k[s], self.flamingo_P_k[s], kind='cubic')

            try:
                self.flamingo_P_k_interp[s, :] = h(self.k_test)
            except ValueError as v:
                if "below the interpolation range's minimum value" in str(v):
                    minimum_index = min(filter(lambda k: k[1] > min(flamingo_model[:array_size, 0]), enumerate(self.k_test)))[0]

                    self.flamingo_P_k_interp[s, :minimum_index] = 0
                    self.flamingo_P_k_interp[s, minimum_index:] = h(self.k_test[minimum_index:])
                else:
                    pass

                #elif "above the interpolation range's maximum value" in str(v):
                    #self.flamingo_P_k_interp[s, :] = h(max(filter(lambda k: k < max(flamingo_model[:array_size, 0]), self.k_test)))
            
        if boost == True:
            self.flamingo_P_k_nonlinear = self.flamingo_P_k_interp / self.flamingo_linear_spectra
        else:
            self.flamingo_P_k_nonlinear = self.flamingo_P_k_interp

        self.flamingo_mask = (self.flamingo_P_k_nonlinear == 0)
        ones = np.ones(self.flamingo_P_k_nonlinear.shape)
        self.flamingo_P_k_nonlinear[self.flamingo_mask] = ones[self.flamingo_mask]
            
        self.Y_test = self.flamingo_P_k_nonlinear

        return

    def plot_k_flamingo(self, file_type='png'):

        model_lines = [('tab:pink','dashed'), ('tab:green','dotted'), ('tab:green','solid'), ('tab:green','dashed'), ('tab:green','^'), ('tab:brown','solid'), ('tab:blue','solid'), ('tab:purple','dotted'), ('tab:purple','solid'), ('tab:purple','dashed'), ('tab:orange','solid'), ('tab:red','solid'), ('tab:olive','dotted'), ('tab:grey','D')]
        print(self.flamingo_mask)
        for i in range(len(self.flamingo_P_k_nonlinear)):
            unmasked_indices = np.arange(self.k_test.shape[0])[~self.flamingo_mask[i,:]]
            #print(unmasked_indices)
            dash_pattern = [1,5]
            try:
                faded_data, = plt.plot(self.k_test, self.P_k_nonlinear[i, :], color=model_lines[i][0], linestyle=model_lines[i][1], alpha=0.3)
                full_data, = plt.plot(self.k_test[unmasked_indices], self.P_k_nonlinear[i, unmasked_indices], color=model_lines[i][0], linestyle=model_lines[i][1], label='L'+f"{self.flamingo_sims[i][0]}"+'N'+f"{self.flamingo_sims[i][1]}")
            except ValueError as v:
                if 'is not a valid value for ls' in str(v):
                    faded_data, = plt.plot(self.k_test, self.P_k_nonlinear[i, :], color=model_lines[i][0], linestyle=None, marker=model_lines[i][1], markersize=3, alpha=0.3)
                    full_data, = plt.plot(self.k_test[unmasked_indices], self.P_k_nonlinear[i, unmasked_indices], color=model_lines[i][0], linestyle=None, marker=model_lines[i][1], markersize=3, label='L'+f"{self.flamingo_sims[i][0]}"+'N'+f"{self.flamingo_sims[i][1]}")
                    
            #if model_lines[i][1] == 'dashed' or 'dotted':
                #faded_data.set_dashes(dash_pattern)
                #full_data.set_dashes(dash_pattern)
            #else:
                #faded_data.set_linestyle('-')
                #full_data.set_linestyle('-')
                              
            #try:
                #plt.plot(self.k_test, self.P_k_nonlinear[i, :], color=model_lines[i][0], linestyle=model_lines[i][1], label='L'+f"{self.flamingo_sims[i][0]}"+'N'+f"{self.flamingo_sims[i][1]}")
            #except ValueError as v:
                #if 'is not a valid value for ls' in str(v):
                    #plt.plot(self.k_test, self.P_k_nonlinear[i, :], color=model_lines[i][0], linestyle=None, marker=model_lines[i][1], markersize=3, label='L'+f"{self.flamingo_sims[i][0]}"+'N'+f"{self.flamingo_sims[i][1]}")
        plt.title(r'$P(k)$ vs test_k for all models', fontsize=10, wrap=True)
        plt.xlabel(r'$k \: [1/Mpc]$')
        plt.ylabel(r'$P(k) \: [Mpc^3]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'./Plots/flamingo_data.{file_type}', dpi=800)
        plt.clf()

        return

    def weights(self, export_csv=False, plot_weights=False, file_type='png'):

        self.HR_finite_vol_weights = self.flamingo_P_k_nonlinear[10,:]/self.flamingo_P_k_nonlinear[2,:]
        self.IR_finite_vol_weights = self.flamingo_P_k_nonlinear[10,:]/self.flamingo_P_k_nonlinear[6,:]
        self.LR_finite_vol_weights = self.flamingo_P_k_nonlinear[10,:]/self.flamingo_P_k_nonlinear[10,:]
        
        self.HR_mass_res_weights = self.flamingo_P_k_nonlinear[3,:]/self.flamingo_P_k_nonlinear[3,:] 
        self.IR_mass_res_weights = self.flamingo_P_k_nonlinear[3,:]/self.flamingo_P_k_nonlinear[2,:] 
        self.LR_mass_res_weights = self.flamingo_P_k_nonlinear[3,:]/self.flamingo_P_k_nonlinear[1,:] 

        if export_csv == True:
            finite_vol_df = pd.DataFrame(np.transpose(self.HR_finite_vol_weights), columns=['HR'])
            finite_vol_df['IR'] = np.transpose(self.IR_finite_vol_weights)
            finite_vol_df['LR'] = np.transpose(self.LR_finite_vol_weights)
            finite_vol_df.to_csv('./BXL_data/finite_vol_weights.csv')
                                
            mass_res_df = pd.DataFrame(np.transpose(self.HR_mass_res_weights), columns=['HR'])
            mass_res_df['IR'] = np.transpose(self.IR_mass_res_weights)
            mass_res_df['LR'] = np.transpose(self.LR_mass_res_weights)
            mass_res_df.to_csv('./BXL_data/mass_res_weights.csv')

        plt.hlines(y=1.000, xmin=-1, xmax=100, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=1.010, xmin=-1, xmax=100, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.050, xmin=-1, xmax=100, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        
        if plot_weights == 'both':
            
            w,h=fi.figaspect(.3)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w,h))
            ax1.plot(self.k_test, self.HR_finite_vol_weights, color='tab:green', label='High Res')
            ax1.plot(self.k_test, self.IR_finite_vol_weights, color='tab:blue', label='Intermediate Res')
            ax1.plot(self.k_test, self.LR_finite_vol_weights, color='tab:orange', label='Low Res')
            fig.suptitle('Weights for the emulator for finite volume (left) and mass resolution (right) effects')
            ax1.set_xlabel('k (1/Mpc)')
            ax1.set_ylabel('Variance')
            ax1.set_xscale('log')
            ax1.set_xlim(right=20)
            ax1.legend()
            

            ax2.plot(self.k_test, self.HR_mass_res_weights, color='tab:green', label='High Res')
            ax2.plot(self.k_test, self.IR_mass_res_weights, color='tab:blue', label='Intermediate Res')
            ax2.plot(self.k_test, self.LR_mass_res_weights, color='tab:orange', label='Low Res')
            ax2.set_xlabel('k (1/Mpc)')
            ax2.set_xscale('log')
            ax2.set_xlim(right=20)
            ax2.legend()
            
            fig.subplots_adjust(wspace=0.15)
            plt.savefig(f'./Plots/emulator_weights.{file_type}', dpi=1200)
            plt.clf()

        elif plot_weights == 'finite_vol':

            plt.plot(self.k_test, self.HR_finite_vol_weights, color='tab:green', label='High Res')
            plt.plot(self.k_test, self.IR_finite_vol_weights, color='tab:blue', label='Intermediate Res')
            plt.plot(self.k_test, self.LR_finite_vol_weights, color='tab:orange', label='Low Res')
            plt.title('Weights for the emulator for finite volume effects')
            plt.xlabel('k (1/Mpc)')
            plt.ylabel('Variance')
            plt.xscale('log')
            plt.xlim(right=20)
            plt.legend()
            plt.savefig(f'./Plots/emulator_weights_finite_vol.{file_type}', dpi=1200)
            plt.clf()

        elif plot_weights == 'mass_res':

            plt.plot(self.k_test, self.HR_mass_res_weights, color='tab:green', label='High Res')
            plt.plot(self.k_test, self.IR_mass_res_weights, color='tab:blue', label='Intermediate Res')
            plt.plot(self.k_test, self.LR_mass_res_weights, color='tab:orange', label='Low Res')
            plt.title('Weights for the emulator for mass resolution effects')
            plt.xlabel('k (1/Mpc)')
            plt.ylabel('Variance')
            plt.xscale('log')
            plt.xlim(right=20)
            plt.legend()
            plt.savefig(f'./Plots/emulator_weights_mass_res.{file_type}', dpi=1200)
            plt.clf()

        return


if __name__ == "__main__":
    import pdb
    
    flamingo = flamingoDMOData(pk='powmes', lin='class', flamingo_lin='camb')
    print(flamingo.P_k_nonlinear)
    flamingo.extend_data('emu')
    print(flamingo.P_k_nonlinear)
    print(flamingo.k)
    print(flamingo.P_k)
    print(flamingo.k_test)
    #print(flamingo.test_models)
    print(flamingo.X_train)
    print(flamingo.Y_train)
    print(flamingo.X_test)
    print(flamingo.Y_test)
    flamingo.plot_k_flamingo()
    flamingo.weights(plot_weights='mass_res')
    print(flamingo.LR_mass_res_weights)
    quit()
    
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
