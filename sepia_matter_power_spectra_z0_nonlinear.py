import numpy as np
import matplotlib.pyplot as plt
from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
import sepia.SepiaPlot as SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
from scipy import interpolate
import random

bins = 100
#cutoff = (np.pi*1260)/1400

#test_model = [3, 12, 18, 90, 138]
#test_models = random.randint(0, 149)
test_models = 50
#test_model = 11
print(test_models)

#Some setup parameters for the model that I don't really touch now
pc = 3
samp = 50
step = 20
mcmc = 1000

mcmc_params = False
plots = True
save = True

def emulator_nonlinear(bins, test_models, pc, samp, step, mcmc, mcmc_params = False, plots = False, save = False, boost = True, cosmo = np.arange(9), lin = 'class', pk = 'reg', plot_k = False, method = 'rag-obs', rebin=True):

    #The parameters file
    parameters_z0 = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/slhs_nested_3x50_w0_m0p6_m1p2_wa_m1p6_p0p5_with_running_and_fgas_and_As.txt', skiprows=1, max_rows=150, usecols=cosmo)

    #Normalize the design to be between 0,1
    design_max = np.max(parameters_z0,0)
    design_min = np.min(parameters_z0,0)
    print(design_max)
    print(design_min)
    parameters_z0_norm = (parameters_z0-design_min)/(design_max-design_min)
            
    #Split into test and train data
    test_parameters = parameters_z0_norm[test_models, :]
    #test_parameters = test_parameters[:, np.newaxis]
    #test_parameters.reshape(1,-1)
    print(test_parameters)
    train_parameters = np.delete(parameters_z0_norm, test_models, axis=0)
    #train_parameters = np.delete(parameters_z0_norm, np.arange(test_start, test_end+1), axis=0)

    if pk == 'reg':
        model_000_k = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_000_N1260_L700_DMO/power_spectra/power_matter_0122.txt', skiprows = 20, usecols = 1)
        length = len(model_000_k)
    elif pk =='nbk':
        model_000_k = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_000_N1260_L700_DMO/PS/PS_k_078.csv', skiprows = 1, usecols = 0)*(parameters_z0[0, 2])
        length = len(model_000_k)
        
    if rebin == 'k_eff':
        P_k_z0_nonlinear = np.zeros([150, 8])
    else:
        P_k_z0_nonlinear = np.zeros([150, bins])

    #low_k_cut = (2*np.pi)/350
    low_k_cut = 0.025
    #high_k_cut = (2*np.pi*1260)/1400
    high_k_cut = 4.5
    k_z0_test = np.logspace(np.log10(low_k_cut), np.log10(high_k_cut), bins)
    
    #This is used to build up k, the full P(k), linear P(k) and non-linear P(k)
    for i in range(0, 150):

        print('model = ' + str(i))
        if i in np.arange(0, 50):
            size = 700
        elif i in np.arange(50, 100):
            size = 1400
        elif i in np.arange(100, 150):
            size = 350

        if rebin == True:
            k_rebin = np.arange(2, 257, 2)*(np.pi/size)
            print(k_rebin)

        if pk == 'reg':
            k_z0 = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/power_spectra/power_matter_0122.txt', skiprows = 20, usecols = 1)

            P_k_z0 = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/power_spectra/power_matter_0122.txt', skiprows = 20, usecols = 2)
        elif pk == 'nbk':
            k_z0 = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/PS/PS_k_078.csv', skiprows = 1, usecols = 0)*(parameters_z0[i, 2])
            #print(k_z0)
           # print(parameters_z0[i, 2])
            
            P_k_z0 = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" + '_N1260_L' + f"{size}" + '_DMO/PS/PS_k_078.csv', skiprows = 1, usecols = 1)/(parameters_z0[i, 2]**3)
            
        h = interpolate.interp1d(k_z0, P_k_z0, kind='cubic')#, fill_value="extrapolate") #if pk == 'nbk' else 'nan')

        if rebin == True:
            P_k_z0_interp = P_k_z0
            k_z0_test = k_rebin
        elif rebin == False:
            P_k_z0_interp = h(k_z0_test)
        elif rebin == 'k_eff':
            k_eff = np.sum(P_k_z0 * k_z0) /np.sum(P_k_z0)
            print(k_eff)
            low_k = 0.0352 * k_eff
            high_k = 8 * k_eff
            k_z0_test = np.arange(low_k, high_k, k_eff)
            print(k_z0_test)
            P_k_z0_interp = h(k_z0_test)
        #print(P_k_z0_interp)

        if plot_k == True:
            plt.figure(3)
            if size == 700:
                plt.plot(k_z0_test, P_k_z0_interp, color = 'tab:blue', label=('Models 000-049' if i==0 else None))
            elif size == 1400:
                plt.plot(k_z0_test, P_k_z0_interp, color = 'tab:orange', label=('Models 050-099' if i==50 else None))
            elif size == 350:
                plt.plot(k_z0_test, P_k_z0_interp, color = 'tab:green', label=('Models 100-149' if i==100 else None))
        elif plot_k == False:
            pass    
                    
        if lin == 'camb':
            k_z0_linear = np.logspace(-3, np.log10(50), 300)*(parameters_z0[i, 2])
            P_k_camb_linear = np.loadtxt('/home/arijconl/BAHAMAS_XL/pk_lin_camb2022_slhs_nested_3x50_kmax_50_running_w0_m1p2_m0p6_wa_m1p6_p0p5.txt', skiprows=index, max_rows=1)/(parameters_z0[i, 2]**3)
            
            f = interpolate.interp1d(k_z0_linear, P_k_camb_linear, kind='cubic')
            y = f(k_z0_test)
        elif lin == 'class':
            P_k_class_linear = np.loadtxt('/storage/users/arijsalc/BAHAMAS_XL/DMO/model_' + f"{i:03d}" +'_N1260_L' + f"{size}" + '_DMO/class_linear_spectra_z_0.txt', skiprows=2, usecols=(0, 2))

            f = interpolate.interp1d(P_k_class_linear[:, 0], P_k_class_linear[:, 1], kind='cubic')
            if rebin == True:
                y = f(k_rebin)
            elif rebin == False:
                y = f(k_z0_test)
            elif rebin == 'k_eff':
                y = f(k_z0_test) 

        if rebin == 'k_eff':
            k_plot = k_z0_test
        else:
            u=0
            for t in k_z0:
                if t <= max(P_k_class_linear[:, 0]):
                    u+=1
            k_plot = k_z0[:u]
        
        if boost == True:
            P_k_z0_nonlinear[i, :] = P_k_z0_interp / y
        elif boost == False:
            P_k_z0_nonlinear[i, :] = P_k_z0_interp

        e = interpolate.interp1d(P_k_class_linear[:, 0], P_k_class_linear[:, 1], kind='cubic')
        p = f(k_plot)

        if plot_k == True:
            plt.figure(1)
            if i in range(50):
                plt.plot(k_plot, (P_k_z0[:u]/p if rebin != 'k_eff' else P_k_z0_nonlinear[i, :]), color='tab:blue', label=('Models 000-049' if i==0 else None))
            elif i in range(50, 100):
                plt.plot(k_plot, (P_k_z0[:u]/p if rebin != 'k_eff' else P_k_z0_nonlinear[i, :]), color='tab:orange', label=('Models 050-099' if i==50 else None))
            elif i in range(100, 150):
                plt.plot(k_plot, (P_k_z0[:u]/p if rebin != 'k_eff' else P_k_z0_nonlinear[i, :]), color='tab:green', label=('Models 100-149' if i==100 else None))
        
            plt.figure(2)
            k_fund = (2*np.pi)/size
            if i in range(50):
                plt.plot(k_plot/k_fund, (P_k_z0[:u]/p if rebin != 'k_eff' else P_k_z0_nonlinear[i, :]), color='tab:blue', label=('Models 000-049' if i==0 else None))
            elif i in range(50, 100):
                plt.plot(k_plot/k_fund, (P_k_z0[:u]/p if rebin != 'k_eff' else P_k_z0_nonlinear[i, :]), color='tab:orange', label=('Models 050-099' if i==50 else None))
            elif i in range(100, 150):
                plt.plot(k_plot/k_fund, (P_k_z0[:u]/p if rebin != 'k_eff' else P_k_z0_nonlinear[i, :]), color='tab:green', label=('Models 100-149' if i==100 else None))
        elif plot_k == False:
            pass

    #  print(P_k_z0_nonlinear)
    P_k_z0_nonlinear_test = P_k_z0_nonlinear[test_models, :]
    P_k_z0_nonlinear_train = np.delete(P_k_z0_nonlinear, test_models, axis=0)
    #print(P_k_z0_nonlinear_train)

    if plot_k == True:
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
        
    elif plot_k == False:
        pass
    
    #Log of non-linear boost
    P_k_z0_nonlinear_train_log = np.log10(P_k_z0_nonlinear_train)
#    print(P_k_z0_nonlinear_train_log)
    k = np.linspace(0, 1, bins)
    print(np.size(P_k_z0_nonlinear_train_log, 0))
    #k_z0_test_array = np.repeat(k_z0_test, np.size(P_k_z0_nonlinear_train_log, 0), axis=0).reshape(-1, 100)
    #print(k_z0_test_array)

    print(np.concatenate((P_k_z0_nonlinear_train_log[:50, :], P_k_z0_nonlinear_train_log[100:, :])).shape)
    #SepiaData model building
    if method == 'sim-only':
        data = SepiaData(x_sim = train_parameters, y_sim = P_k_z0_nonlinear_train_log, y_ind_sim = k)
    elif method == 'rag-obs':
        if test_models in range(50):
            data = SepiaData(y_obs = P_k_z0_nonlinear_train_log[50:, :], y_ind_obs = k, y_sim=P_k_z0_nonlinear_train_log[:50, :], t_sim = train_parameters[:50, :], y_ind_sim=k)
        elif test_models in range(50, 100):
            data = SepiaData(y_obs = np.concatenate((P_k_z0_nonlinear_train_log[:50, :], P_k_z0_nonlinear_train_log[100:, :])), y_ind_obs = k, y_sim=P_k_z0_nonlinear_train_log[50:100, :], t_sim = train_parameters[50:100, :], y_ind_sim=k)
        elif test_models in range(100, 150):
            data = SepiaData(y_obs = P_k_z0_nonlinear_train_log[:100, :], y_ind_obs = k, y_sim=P_k_z0_nonlinear_train_log[100:, :], t_sim = train_parameters[100:, :], y_ind_sim=k)

    data.transform_xt()
    data.standardize_y()
    data.create_K_basis(n_pc=pc)
    if method == 'rag-obs':
        data.create_D_basis(D_type = 'linear')

    print(data)

    model = SepiaModel(data)

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

    pred_samples = model.get_samples(numsamples=10)

    print("pred_samples = " + str(pred_samples))

    #Prediction of model000
    if method == 'sim-only':
        pred = SepiaEmulatorPrediction(samples = pred_samples, model=model, x_pred = test_parameters.reshape(len(test_models),-1) if isinstance(test_models, list) else test_parameters.reshape(1, -1))
    elif method == 'rag-obs':
        pred = SepiaEmulatorPrediction(samples = pred_samples, model=model, t_pred = test_parameters.reshape(len(test_models),-1) if isinstance(test_models, list) else test_parameters.reshape(1, -1))
    
    predy = pred.get_y()

    #print(predy)

    y_pred_mean = np.mean(predy,0)

    print(y_pred_mean.shape)
    
    #Unlog the predicted ratio
    y_pred_mean_norm = 10**(y_pred_mean)
    
    error = y_pred_mean_norm/P_k_z0_nonlinear_test

    if plots == False:
        pass
    elif plots == True:
        #Plot of non-linear P(k), true and predicted
        #The predicted is multiplied by linear interpolated P(k) to get back the non-linear spectrum
        if isinstance(test_models, int):
            plt.plot(k_z0_test, P_k_z0_nonlinear_test, color='b', label=('True Non-linear P(k) for model_' + f"{test_models:03d}"))
            plt.plot(k_z0_test, y_pred_mean_norm.reshape(100, -1), color='r', label=('Predicted Non-linear P(k) for model_' + f"{test_models:03d}"))
            plt.title('Comparison of predicted and true P(k) for model_' + f"{test_models:03d}" + ' (' + f"{mcmc:01\
d}" + ' runs)', fontsize=10, wrap=True)
        elif isinstance(test_models, list):
            for m in range(len(test_models)):
                plt.plot(k_z0_test, P_k_z0_nonlinear_test[m, :], color='b', label=('True Non-linear P(k) for model_' + f"{test_models[m]:03d}"))
                plt.plot(k_z0_test, y_pred_mean_norm[m, :], color='r', label=('Predicted Non-linear P(k) for model_' + f"{test_models[m]:03d}"))
                plt.title('Comparison of predicted and true P(k) for ' + f"{len(test_models)}" + ' models (' + f"{mcmc:01d}" + ' runs)', fontsize=10, wrap=True)
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
            plt.savefig('./Plots/Power_spectra_z0_predicted_nonlinear.pdf', dpi=800)
        plt.clf()

        #Error plot
        plt.hlines(y=1.000, xmin=-1, xmax=high_k_cut, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=high_k_cut, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=high_k_cut, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=high_k_cut, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=high_k_cut, color='k', linestyles='dotted', alpha=0.5, label=None)
        if isinstance(test_models, int):
            plt.plot(k_z0_test, error.reshape(100, -1), label=('Residual error for model_' + f"{test_models:03d}"))
            plt.title('Residual error on non-linear P(k) for prediction of model_' + f"{test_models}" + ' (' + f"{mcmc:01d}" + ' runs with ' + f"{pc:01d}" + 'pcs)', fontsize=10, wrap=True)
        elif isinstance(test_models, list):
            for m in range(len(test_models)):
                plt.plot(k_z0_test, error[m, :], label=('Residual error for model_' + f"{test_models[m]:03d}"))
                plt.title('Residual error on non-linear P(k) for prediction of ' + f"{len(test_models)}" + ' models (' + f"{mcmc:01d}" + ' runs with ' + f"{pc:01d}" + 'pcs)', fontsize=10, wrap=True)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('Predicted Non-linear P(k)/True Non-linear P(k)')
        plt.xscale('log')
        plt.legend()

        if save == False:
            plt.show()
        elif save == True:
            plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear.pdf', dpi=800)

        plt.clf()

    return k_z0_test, error, test_models

if __name__ == "__main__":
    test_k, error, test_models = emulator_nonlinear(bins, test_models, pc, samp, step, mcmc, mcmc_params, plots, save, pk = 'nbk', plot_k = True, rebin = 'k_eff')
