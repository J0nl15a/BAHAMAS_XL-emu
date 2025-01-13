import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from data_loader import bahamasXLDMOData
import matplotlib.figure as fi
import matplotlib.colors as mcolors

    
class flamingoDMOData(bahamasXLDMOData):

    def __init__(self, bins=100, test_cosmology=False, cutoff=(-1, 10), cosmo_params=np.arange(9), resolution='HR', pk='powmes', lin='camb', flamingo_lin='camb', boost=True, log=True, holdout=False, sigma8=True, plot_hypercube=False, weight_k=True):

        #The parameters file
        self.flamingo_parameters = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_params.txt', skiprows=1, usecols=cosmo_params)
        if sigma8==True:
            self.flamingo_As = self.flamingo_parameters[4].copy()
            sig8 = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_params.txt', skiprows=1, usecols=10)
            self.flamingo_parameters[4] = sig8

        self.flamingo_particle_masses = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_DM_masses.txt').reshape(-1,1)
        self.flamingo_parameters = np.tile(self.flamingo_parameters, (14,1))
        self.flamingo_parameters = np.hstack((self.flamingo_parameters, self.flamingo_particle_masses))
        
        super().__init__(test_cosmology=test_cosmology, bins=bins, cutoff=cutoff, cosmo_params=cosmo_params, resolution=resolution, pk=pk, lin=lin, boost=boost, log=log, holdout=holdout, sigma8=sigma8, plot_hypercube=plot_hypercube, weight_k=weight_k)
        
        super().extend_data(pad='emu')

        
        self.flamingo_k = []
        self.flamingo_P_k = []
        self.flamingo_P_k_interp = np.zeros([14, self.bins])
        self.flamingo_P_k_nonlinear = np.zeros([14, self.bins])
        
        if flamingo_lin == 'camb':
            self.flamingo_camb_linear = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_camb_pk.txt', usecols=(0,1))
            self.flamingo_k_linear = self.flamingo_camb_linear[:,0]*(self.flamingo_parameters[0,2])
            self.flamingo_Pk_linear = self.flamingo_camb_linear[:,1]/(self.flamingo_parameters[0,2]**3)
            
            flamingo_linear_interpolation_function = interpolate.interp1d(self.flamingo_k_linear, self.flamingo_Pk_linear, kind='cubic', bounds_error=False, fill_value=np.nan)
            #self.flamingo_linear_spectra = flamingo_linear_interpolation_function(self.k_test)
        elif flamingo_lin == 'class':
            self.flamingo_class_linear = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_CLASS_linear_pk.txt', usecols=(0,1))
            self.flamingo_k_linear = self.flamingo_class_linear[:,0]
            self.flamingo_Pk_linear = self.flamingo_class_linear[:,1]
        
            flamingo_linear_interpolation_function = interpolate.interp1d(self.flamingo_k_linear, self.flamingo_Pk_linear, kind='cubic', bounds_error=False, fill_value=np.nan)
            #self.flamingo_linear_spectra = flamingo_linear_interpolation_function(self.k_test)
            
        elif flamingo_lin == 'rebin':
            pass

        self.flamingo_sims = [(200,720), (350,315), (350,630), (350,1260), (350,2520), (400,720), (700,1260), (1000,900), (1000,1800), (1000,3600), (1400,2520), (2800,5040), (5600,5040), (11200,5040)]

        #This is used to build up k, the full P(k), linear P(k) and non-linear P(k)
        for s in range(len(self.flamingo_sims)):
            flamingo_directory = f'./BXL_data/FLAMINGO_data/power_matter_L{self.flamingo_sims[s][0]:04d}N{self.flamingo_sims[s][1]:04d}_DMO_z0.txt'

            flamingo_model = np.loadtxt(flamingo_directory, skiprows = 20, usecols = (1,2))
            flamingo_model[:6,0] = flamingo_model[:6,0]*self.correction_factor
            self.flamingo_k.append(flamingo_model[:, 0])
            self.flamingo_P_k.append(flamingo_model[:, 1])

            self.flamingo_linear_spectra = flamingo_linear_interpolation_function(self.flamingo_k[s])
            self.flamingo_boost = self.flamingo_P_k[s]/self.flamingo_linear_spectra

            mask = ~np.isnan(self.flamingo_boost)
            self.flamingo_boost = self.flamingo_boost[mask]
            self.flamingo_k[s] = self.flamingo_k[s][mask]
            self.flamingo_P_k[s] = self.flamingo_P_k[s][mask]
            
            h = interpolate.interp1d(self.flamingo_k[s], self.flamingo_boost, kind='cubic', bounds_error=False, fill_value=np.nan)

            self.flamingo_P_k_interp[s, :] = h(self.k_test)
            
        self.flamingo_P_k_nonlinear = self.flamingo_P_k_interp.copy()

        self.flamingo_mask = np.isnan(self.flamingo_P_k_nonlinear)
        ones = np.ones(self.flamingo_P_k_nonlinear.shape)
        self.flamingo_P_k_nonlinear[self.flamingo_mask] = ones[self.flamingo_mask]
        
        if not isinstance(self.holdout, bool):
            self.Y_test = self.Y_test
        elif not isinstance(self.test_cosmology, bool):
            self.Y_test = self.Y_test
        else:
            self.Y_test = self.flamingo_P_k_nonlinear.copy()

        if isinstance(test_cosmology, bool) and isinstance(holdout, bool):
            self.X_test = np.delete(self.X_test, [4,-1], axis=0)
            self.Y_test = np.delete(self.Y_test, [4,-1], axis=0)

        return

    def plot_k_flamingo(self, file_type='png'):

        model_lines = ['tab:pink', '#637939', 'tab:green', '#b5cf6b', '#a1d99b', 'tab:purple', 'tab:blue', '#e7ba52', 'tab:cyan', 'tab:olive', 'tab:orange', 'tab:red', 'tab:brown', 'tab:grey']
        print(self.flamingo_mask)
        for f in range(len(self.flamingo_P_k_nonlinear)):
            plt.plot(self.flamingo_k[f], self.flamingo_P_k[f], color=model_lines[f], label='L'+f"{self.flamingo_sims[f][0]}"+'N'+f"{self.flamingo_sims[f][1]}")
        plt.plot(self.flamingo_class_linear[:,0], self.flamingo_class_linear[:,1], color='k', alpha=.7, label='CLASS Linear spectra')
        #plt.plot(self.flamingo_camb_linear[:,0]*(self.flamingo_parameters[2]), self.flamingo_camb_linear[:,1]/(self.flamingo_parameters[2]**3), color='k', alpha=.7, label='CAMB Linear spectra')
        #plt.plot(self.flamingo_k[f], self.flamingo_linear_spectra, color='k', alpha=.7, label='Linear spectra')
        plt.title(r'FLAMINGO non-linear spectra', fontsize=10, wrap=True)
        plt.xlabel(r'$k \: [1/Mpc]$')
        plt.ylabel(r'$P(k) \: [Mpc^3]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(ncol=2)
        plt.savefig(f'./Plots/flamingo_raw_data.{file_type}', dpi=1200)
        plt.clf()

        for f in range(len(self.flamingo_P_k_nonlinear)):
            plt.plot(self.k_test, self.flamingo_P_k_interp[f, :], color=model_lines[f], label='L'+f"{self.flamingo_sims[f][0]}"+'N'+f"{self.flamingo_sims[f][1]}")
        #plt.plot(self.k_test, self.flamingo_linear_spectra, color='k', alpha=.7, label='Linear spectra')
        plt.title(r'FLAMINGO non-linear spectra rebinnned', fontsize=10, wrap=True)
        plt.xlabel(r'$k \: [1/Mpc]$')
        plt.ylabel(r'$P(k) \: [Mpc^3]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(ncol=2)
        plt.savefig(f'./Plots/flamingo_data_interp.{file_type}', dpi=1200)
        plt.clf()
        
        for i in range(len(self.flamingo_P_k_nonlinear)):
            #unmasked_indices = np.arange(self.k_test.shape[0])[~self.flamingo_mask[i,:]]

            #plt.plot(self.k_test, self.P_k_nonlinear[i, :], color=model_lines[i], linestyle='dotted', label='Extended data' if i==0 else None)
            plt.plot(self.k_test, self.flamingo_P_k_nonlinear[i, :], color=model_lines[i], label='L'+f"{self.flamingo_sims[i][0]}"+'N'+f"{self.flamingo_sims[i][1]}")
            #plt.plot(self.k_test[unmasked_indices], self.P_k_nonlinear[i, unmasked_indices], color=model_lines[i], label='L'+f"{self.flamingo_sims[i][0]}"+'N'+f"{self.flamingo_sims[i][1]}")

        plt.title(r'$P(k)$ vs test_k for all models', fontsize=10, wrap=True)
        plt.xlabel(r'$k \: [1/Mpc]$')
        plt.ylabel(r'$P(k) \: [Mpc^3]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(ncol=2)
        plt.savefig(f'./Plots/flamingo_data.{file_type}', dpi=1200)
        plt.clf()

        uhr = interpolate.interp1d(self.flamingo_k[4], self.flamingo_k[4], kind='cubic', bounds_error=False, fill_value=np.nan)
        class_lin = interpolate.interp1d(self.flamingo_class_linear[:,0], self.flamingo_class_linear[:,1], kind='cubic', bounds_error=False, fill_value=np.nan)
        #class_lin = interpolate.interp1d(self.flamingo_camb_linear[:,0]*(self.flamingo_parameters[2]), self.flamingo_camb_linear[:,1]/(self.flamingo_parameters[2]**3), kind='cubic', bounds_error=False, fill_value=np.nan)

        for f in range(len(self.flamingo_P_k_nonlinear)):
            plt.plot(self.flamingo_k[f], ((self.flamingo_P_k[f]/class_lin(self.flamingo_k[f]))/(uhr(self.flamingo_k[f])/class_lin(self.flamingo_k[f]))), color=model_lines[f], label='L'+f"{self.flamingo_sims[f][0]}"+'N'+f"{self.flamingo_sims[f][1]}")
        plt.title(r'FLAMINGO non-linear spectra ratioed with the L0350N2520 run', fontsize=10, wrap=True)
        plt.xlabel(r'$k \: [1/Mpc]$')
        plt.ylabel(r'$P(k) \: [Mpc^3]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(ncol=2)
        plt.savefig(f'./Plots/flamingo_ratio.{file_type}', dpi=1200)
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

        if plot_weights != False:

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

                print(self.LR_mass_res_weights)
                print([self.LR_mass_res_weights[k]-min(self.HR_mass_res_weights[k],self.IR_mass_res_weights[k],self.LR_mass_res_weights[k]) for k in range(len(self.k_test))])
                weights_HR_norm = [((self.HR_mass_res_weights[k]-min(self.HR_mass_res_weights[k],self.IR_mass_res_weights[k],self.LR_mass_res_weights[k]))*self.Y_train[100:,k])**2 for k in range(len(self.k_test))]
                weights_IR_norm = [((self.IR_mass_res_weights[k]-min(self.HR_mass_res_weights[k],self.IR_mass_res_weights[k],self.LR_mass_res_weights[k]))*self.Y_train[:50,k])**2 for k in range(len(self.k_test))]
                weights_LR_norm = [((self.LR_mass_res_weights[k]-min(self.HR_mass_res_weights[k],self.IR_mass_res_weights[k],self.LR_mass_res_weights[k]))*self.Y_train[50:100,k])**2 for k in range(len(self.k_test))]
                print(weights_LR_norm)
                #for i in range(50):
                plt.plot(self.k_test, weights_HR_norm, color='tab:green')#, label='High Res')
                plt.plot(self.k_test, weights_IR_norm, color='tab:blue')#, label='Intermediate Res')
                plt.plot(self.k_test, weights_LR_norm, color='tab:orange')#, label='Low Res')
                plt.title('Weights calculated internally for the emulator, for mass resolution effects', wrap=True)
                plt.xlabel('k (1/Mpc)')
                plt.ylabel('Variance')
                plt.xscale('log')
                plt.xlim(right=20)
                plt.legend()
                plt.savefig(f'./Plots/emulator_weights_mass_res_emulator.{file_type}', dpi=1200)
                plt.clf()

        return


if __name__ == "__main__":
    import pdb
    
    flamingo = flamingoDMOData(pk='powmes', lin='class', flamingo_lin='class', log=False)
    print(flamingo.P_k_nonlinear)
    print(flamingo.k)
    print(flamingo.P_k)
    print(flamingo.k_test)
    #print(flamingo.test_models)
    print(flamingo.X_train)
    print(flamingo.Y_train)
    print(flamingo.X_test)
    print(flamingo.Y_test)
    quit()
    flamingo.plot_k_flamingo()
    flamingo.weights(plot_weights='mass_res')
    print(flamingo.LR_mass_res_weights)
    print(flamingo.X_test*(flamingo.design_max-flamingo.design_min)+flamingo.design_min)
