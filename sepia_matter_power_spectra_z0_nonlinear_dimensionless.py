import numpy as np
import matplotlib.pyplot as plt
from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
import sepia.SepiaPlot as SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
import sepia.SepiaSensitivity as SepiaSensitivity
from scipy import interpolate
import random

bins = 100
#(2*np.pi)/700

#test_model = random.randint(0, 49)
test_model = 1
print(test_model)

#Some setup parameters for the model that I don't really touch now
pc = 3
samp = 50
step = 20
mcmc = 1000

mcmc_params = False
plots = True
save = False
Sens = True
info_model = True
info_pred = False

def emulator_dimensionless(bins, test_model, pc, samp, step, mcmc, mcmc_params = False, plots = False, save = False, Sens = False, info_model = False, info_pred = False, boost = True, cosmo = np.arange(9), lin = 'class'):

        #The parameters file
        parameters_z0 = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/slhs_nested_3x50_w0_m0p6_m1p2_wa_m1p6_p0p5_with_running_and_fgas_and_As.txt', skiprows=1, max_rows=150, usecols=cosmo)

        #Normalize the design to be between 0,1
        design_max = np.max(parameters_z0,0)
        design_min = np.min(parameters_z0,0)
        #print(design_max)
        #print(design_min)
        parameters_z0_norm = (parameters_z0-design_min)/(design_max-design_min)

        if test_model == 1:
                test_start = 0
                test_end = 49
                test_size = 700
        elif test_model == 2:
                test_start = 50
                test_end = 99
                test_size = 1400
        elif test_model == 3:
                test_start = 100
                test_end = 149
                test_size = 350
        
        #Split into test and train data
        test_parameters = parameters_z0_norm[test_start:test_end+1, :]
        #test_parameters = test_parameters[:, np.newaxis]
        #test_parameters.reshape(1,-1)
        #print(test_parameters)
        train_parameters = np.delete(parameters_z0_norm, np.arange(test_start, test_end+1), axis=0)

        k_z0 = np.zeros([150, 435])
        P_k_z0 = np.zeros([150, 435])
        P_k_z0_nonlinear = np.zeros([150, 435])
        P_k_z0_nonlinear_interp = np.zeros([150, bins])
        
        #This is used to build up k, the full P(k), linear P(k) and non-linear P(k)
        index = -1
        for i in range(0, 150):

                low_lim = 0
                upp_lim = 1
                low_k_cut = 0.017952   #(2*np.pi)/350
                high_k_cut = 5.654867   #(2*np.pi*1260)/1400
                k_z0_test = np.logspace(np.log10(low_k_cut), np.log10(high_k_cut), bins)

                model_000_k = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_000_N1260_L700_DMO/power_spectra/power_matter_0122.txt', skiprows = 20, usecols = 1)
                model_050_k = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_050_N1260_L1400_DMO/power_spectra/power_matter_0122.txt', skiprows = 20, usecols = 1)
                model_100_k = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_100_N1260_L350_DMO/power_spectra/power_matter_0122.txt', skiprows = 20, usecols = 1)
                
                if i in np.arange(0, 50):
                        size = 700
                        for l in model_000_k:
                                if low_k_cut > l:
                                        low_lim += 1
                                if l <= high_k_cut:
                                        upp_lim += 1
                elif i in np.arange(50, 100):
                        size = 1400
                        for l in model_050_k:
                                if low_k_cut > l:
                                        low_lim += 1
                                if l <= high_k_cut:
                                        upp_lim += 1
                elif i in np.arange(100, 150):
                        size = 350
                        for l in model_100_k:
                                if low_k_cut > l:
                                        low_lim += 1
                                if l <= high_k_cut:
                                        upp_lim += 1

                index += 1
		
                count = 0
                values = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/power_spectra/power_matter_0122.txt', skiprows = 20, usecols = 1)
	
                for k in values:
                        if k <= high_k_cut:
                                count += 1

                print("model_" + str(i) + " count = " + str(count))
	
                k_z0[index, :(upp_lim-low_lim)] = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/power_spectra/power_matter_0122.txt', skiprows = 20+low_lim, usecols = 1, max_rows = upp_lim-low_lim)
 
                P_k_z0[index, :(upp_lim-low_lim)] = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/power_spectra/power_matter_0122.txt', skiprows = 20+low_lim, usecols = 2, max_rows = upp_lim-low_lim)

                if lin == 'camb':
                        h0 = parameters_z0[index, 2]
                        k_z0_linear = np.logspace(-3, np.log10(50), 300)*h0

                        P_k_camb_linear = np.loadtxt('/home/arijconl/BAHAMAS_XL/pk_lin_camb2022_slhs_nested_3x50_kmax_50_running_w0_m1p2_m0p6_wa_m1p6_p0p5.txt', skiprows=index, max_rows=1)/(h0**3)
                        
                        f = interpolate.interp1d(k_z0_linear, P_k_camb_linear, kind='cubic')
                        y = f(k_z0[index, :(upp_lim-low_lim)])
                elif lin == 'class':
                        P_k_class_linear = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/class_linear_spectra_z_0.txt', skiprows=2, usecols=(0, 2))

                        f = interpolate.interp1d(P_k_class_linear[:, 0], P_k_class_linear[:, 1], kind='cubic')
                        y = f(k_z0[index, :(upp_lim-low_lim)])

                if boost == True:
                        P_k_z0_nonlinear[index, :(upp_lim-low_lim)] = P_k_z0[index, :(upp_lim-low_lim)] / y
                elif boost == False:
                        P_k_z0_nonlinear[index, :(upp_lim-low_lim)] = P_k_z0[index, :(upp_lim-low_lim)]

                g = interpolate.interp1d(k_z0[index, :(upp_lim-low_lim)], P_k_z0_nonlinear[index, :(upp_lim-low_lim)], kind='cubic')
                P_k_z0_nonlinear_interp[index, :] = g(k_z0_test)

        P_k_z0_nonlinear_test = P_k_z0_nonlinear_interp[test_start:test_end+1, :]
        P_k_z0_nonlinear_train = np.delete(P_k_z0_nonlinear_interp, np.arange(test_start, test_end+1), axis=0)

        #Log of non-linear boost
        P_k_z0_nonlinear_transformed = np.log10((P_k_z0_nonlinear_train*(k_z0_test**1.5))/(2*(np.pi**2)))

        k = np.logspace(0, 1, bins)

        #SepiaData model building
        data = SepiaData(x_sim = train_parameters, y_sim = P_k_z0_nonlinear_transformed, y_ind_sim = k)

        data.transform_xt()
        data.standardize_y()
        data.create_K_basis(n_pc=pc)

        model = SepiaModel(data)

        if info_model == False:
                pass
        elif info_model == True:
                print(data)
                model.print_prior_info()  # Print information about the priors
                model.print_value_info()  # Print information about the starting parameter values for MCMC
                model.print_mcmc_info()   # Print information about the MCMC step types and step sizes

        model.tune_step_sizes(samp, step) #use 50 samples over 20 different step sizes

        model.do_mcmc(mcmc)

        samples = model.get_samples()

        if mcmc_params == False:
                pass

        elif mcmc_params == True:
                #Plots of MCMC parameters
                fig = SepiaPlot.mcmc_trace(samples)
                plt.show()

                ps = SepiaPlot.param_stats(samples) # returns pandas DataFrame
                print(ps)

                fig = SepiaPlot.rho_box_plots(model)
                plt.show()
                plt.clf()

        pred_samples = model.get_samples(numsamples=10)

        #Prediction of model000
        pred = SepiaEmulatorPrediction(samples = pred_samples, model=model, x_pred = test_parameters.reshape(50,-1))

        predy = pred.get_y()
	
        y_pred_mean = np.mean(predy,0)

        if info_pred == False:
                pass
        elif info_pred == True:
                print("pred_samples = " + str(pred_samples))
                print(predy)
                print(y_pred_mean)

        sens = SepiaSensitivity.sensitivity(model, samples)
        print('sensitivity = ' + str(sens))
	
        #Main effect index
        main_max = np.max(sens['smePm'])
        main_min = np.min(sens['smePm'])
        #print(main_max)
        #print(main_min)
        main_effect = sens['smePm']
        main_effect_norm = (main_effect-main_min)/(main_max-main_min)
        #print(main_effect_norm)
        main_sens = main_effect/np.sum(main_effect)

        #Total effect index
        total_max = np.max(sens['stePm'])
        total_min = np.min(sens['stePm'])
        #print(total_max)
        #print(total_min)
        total_effect = sens['stePm']
        total_effect_norm = (total_effect-total_min)/(total_max-total_min)
        total_sens = total_effect/np.sum(total_effect)
	
        if Sens == False:
                pass
        elif Sens == True:
                #Sensitivity plots
                parameter_names = ['Omega_m', 'f_b', 'h0', 'ns', 'A_s', 'w0', 'wa', 'Omega_nuh^2', 'alpha_s']
                print(parameter_names)
                plt.rc('xtick', labelsize=6)
                plt.bar(np.arange(9)-0.2, main_sens*100, label='Main effect', width=0.4)
                plt.bar(np.arange(9)+0.2, total_sens*100, label='Total effect', width=0.4)
                plt.xticks(np.arange(9), parameter_names)
                plt.xlabel('Parameters')
                plt.ylabel('Sensitivity')
                plt.legend()
                plt.savefig('Power_spectra_z0_predicted_sensitivity.pdf', dpi=800)
                plt.rcdefaults()
                plt.clf()

        #Unlog the predicted ratio
        y_pred_mean_norm = (10**(y_pred_mean))*(2*(np.pi**2))/(k_z0_test**1.5)

        error = y_pred_mean_norm/P_k_z0_nonlinear_test

        if plots == False:
                pass
        
        elif plots == True:
                #Plot of non-linear P(k), true and predicted
                #The predicted is multiplied by linear interpolated P(k) to get back the non-linear spectrum
                for m in range(50):
                        plt.plot(k_z0_test, P_k_z0_nonlinear_test[m, :], color='b', label=('True Non-linear P(k)' if m==0 else None))
                        plt.plot(k_z0_test, y_pred_mean_norm[m, :], color='r', label=('Predicted Non-linear P(k)' if m==0 else None))
                plt.title('Comparison of predicted and true P(k) for models ' + f"{test_start:03d}" + ' - ' + f"{test_end:03d}" + ' (' + f"{mcmc:01d}" + ' runs)', fontsize=10, wrap=True)
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
                        plt.savefig('Power_spectra_z0_predicted_nonlinear_dimensionless.pdf', dpi=800)
		
                plt.clf()

                #Error plot
                plt.hlines(y=1.000, xmin=-1, xmax=high_k_cut, color='k', linestyles='solid', alpha=0.5, label=None)
                plt.hlines(y=0.990, xmin=-1, xmax=high_k_cut, color='k', linestyles='dashed', alpha=0.5, label='1% error')
                plt.hlines(y=1.010, xmin=-1, xmax=high_k_cut, color='k', linestyles='dashed', alpha=0.5, label=None)
                plt.hlines(y=0.950, xmin=-1, xmax=high_k_cut, color='k', linestyles='dotted', alpha=0.5, label='5% error')
                plt.hlines(y=1.050, xmin=-1, xmax=high_k_cut, color='k', linestyles='dotted', alpha=0.5, label=None)
                for m in range(50):
                        plt.plot(k_z0_test, error[m, :], label=('Residual error' if m==0 else None))
                #plt.xlim(right = 0.5)
                #plt.ylim(top = 1.15, bottom = 0.85)
                plt.title('Residual error on non-linear P(k) for prediction of models ' + f"{test_start:03d}" + ' - ' + f"{test_end:03d}" + ' (' + f"{mcmc:01d}" + ' runs with ' + f"{pc:01d}" + 'pcs)', fontsize=10, wrap=True)
                plt.xlabel('k (1/Mpc)')
                plt.ylabel('Predicted Non-linear P(k)/True Non-linear P(k)')
                plt.xscale('log')
                plt.legend()
		
                if save == False:
                        plt.show()
                elif save == True:
                        plt.savefig('Power_spectra_z0_predicted_residual_nonlinear_dimensionless.pdf', dpi=800)
		
                plt.clf()

        return k_z0_test, error, sens, test_start, test_end

if __name__ == "__main__":
        test_k, error, sens, test_start, test_end = emulator_dimensionless(bins, test_model, pc, samp, step, mcmc, mcmc_params, plots, save, Sens, info_model)
