import numpy as np
import pylab as pb
import scipy.stats as st
import matplotlib.figure as fi
from data_loader import bahamasXLDMOData
from flamingo_data_loader import flamingoDMOData
from gpy_emulator_improved import gpyImprovedEmulator
import camb
from scipy.signal import savgol_filter

def parameter_sweep(parameters, method='BXL_emu', bins=100):

    pred = np.zeros((parameters.shape[0], bins))

    if method == 'BXL_emu':
        flamingo = flamingoDMOData(test_cosmology=parameters[0,:], cutoff=(.003,20))
        flamingo.weights()
        emulator_model = gpyImprovedEmulator(flamingo, 'variance_weights_no', error=False, fix_variance=False, ARD=True, save=True)
        pred[0,:] = emulator_model.pred

        flamingo_sweep = flamingoDMOData(test_cosmology=parameters[1:,:], cutoff=(.003,20))
        emulator_model_sweep = gpyImprovedEmulator(flamingo_sweep, 'variance_weights_no', error=False, fix_variance=False, ARD=True, model_upload=True)
        pred[1:,:] = emulator_model_sweep.pred
            
    return pred

if __name__ == "__main__":

    method = 'BXL_emu'

    if method == 'BXL_emu':

        parameter_list = {'\Omega_m':0, 'f_b':1, 'h_0':2, 'n_s':3, '\sigma_8':4, 'w_0':5, 'w_a':6, r'\Omega_{\nu}h^2':7, r'\alpha_s':8, 'log_{10}(DM_{mass})':9}
        parameter = r'\alpha_s'

        precision = 5
        
        f = flamingoDMOData(cutoff=(.003,20))
        f.weights()

        if parameter != 'log_{10}(DM_{mass})':
            params = np.linspace(0,1,10)
            params = (params*(f.design_max[parameter_list[parameter]]-f.design_min[parameter_list[parameter]]))+f.design_min[parameter_list[parameter]]
        else:
            #params = np.logspace(np.log10(f.design_min[9]), np.log10(f.design_max[9]), 10)
            params = np.logspace(8.7, 10.7, 10)
            #params = [5e8, 5e9, 5e10]

            
        test = np.tile(f.flamingo_parameters[9], 11).reshape(11,-1)
        #test = np.tile(f.flamingo_parameters[9], 4).reshape(4,-1)
        print(test.shape)
        test[1:,parameter_list[parameter]] = params
        
        print(test)
        print(test.shape)
        pred = parameter_sweep(test)
        print(test)

    elif method == 'CAMB':
        pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, As=2e-9, ns=0.965, halofit_version='mead')

    print(pred)

    for i in range(len(pred)):
        if i==0:
            pb.plot(f.k_test, f.Y_test[8,:], color='k', label=f'{f.flamingo_parameters[9,parameter_list[parameter]]:.{precision}f} (\'true\' FLAMINGO model)' if parameter!='log_{10}(DM_{mass})' else f'{np.log10(f.flamingo_parameters[9,parameter_list[parameter]]):.{precision}f} (\'true\' FLAMINGO model)')
            pb.plot(f.k_test, pred[0], color='k', linestyle='dashed', label=f'{f.flamingo_parameters[9,parameter_list[parameter]]:.{precision}f} (predicted FLAMINGO model)' if parameter!='log_{10}(DM_{mass})' else f'{np.log10(f.flamingo_parameters[9,parameter_list[parameter]]):.{precision}f} (predicted FLAMINGO model)')
        else:
            pb.plot(f.k_test, pred[i], label=rf'{params[i-1]:.{precision}f}' if parameter!='log_{10}(DM_{mass})' else rf'{np.log10(params[i-1]):.{precision}f}')

    pb.xlabel(r'$k \: [1/Mpc]$', fontsize=15)
    pb.ylabel(r'$P(k) \: [Mpc^3]$', fontsize=15)
    pb.xscale('log')
    pb.yscale('log')
    #pb.title(f'GPy test with external cosmologies')
    pb.title(f'Emulator parameter sweep', wrap=True)
    pb.legend(loc='upper left', fontsize=7, title=f'Parameter: ${parameter}$')
    
    pb.savefig(f'./Plots/parameter_sweep_test.png', dpi=1200)
    pb.clf()
    
    for i in range(len(pred)):
        if i==0:
            pb.hlines(xmin=-1, xmax=100, y=1, color='k', label=f'{test[0,parameter_list[parameter]]:.{precision}f} (predicted FLAMINGO)' if parameter!='log_{10}(DM_{mass})' else f'{np.log10(test[0,parameter_list[parameter]]):.{precision}f} (predicted FLAMINGO)', linestyle='dashed')
        else:
            pb.plot(f.k_test, savgol_filter(pred[i]/pred[0], window_length=7, polyorder=3), label=rf'{params[i-1]:.{precision}f}' if parameter!='log_{10}(DM_{mass})' else rf'{np.log10(params[i-1]):.{precision}f}')
            #pb.plot(f.k_test, pred[i]/pred[0], label=rf'{params[i-1]:.{precision}f}' if parameter!='log_{10}(DM_{mass})' else rf'{np.log10(params[i-1]):.{precision}f}')
    pb.xlabel(r'$k \: [1/Mpc]$', fontsize=15)
    pb.ylabel(r'$P(k) \: [Mpc^3]$', fontsize=15)
    pb.xscale('log')
    pb.xlim(left=1e-2, right=30)
    #pb.title(f'Parameter sweeps predictions/FLAMINGO')
    pb.title(f'Parameter sweep (smoothed) predictions relative to predicted \n FLAMINGO model (L1000N3600)', wrap=True)
    pb.legend(loc='upper left', fontsize=6, title=f'Parameter: ${parameter}$')
    pb.savefig(f'./Plots/parameter_sweep_residual.png', dpi=1200)
    pb.clf()
    
