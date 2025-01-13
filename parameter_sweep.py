import numpy as np
import pylab as pb
import scipy.stats as st
import matplotlib.figure as fi
from data_loader import bahamasXLDMOData
from flamingo_data_loader import flamingoDMOData
from gpy_emulator_improved import gpyImprovedEmulator
import camb

def parameter_sweep(parameters, method='BXL_emu', bins=100):

    pred = np.zeros((parameters.shape[0], bins))

    if method == 'BXL_emu':
        flamingo = flamingoDMOData(test_cosmology=parameters[0,:])
        flamingo.weights()
        emulator_model = gpyImprovedEmulator(flamingo, 'variance_weights_no', error=False, fix_variance=False, ARD=True, save=True)
        pred[0,:] = emulator_model.pred

        flamingo_sweep = flamingoDMOData(test_cosmology=parameters[1:,:])
        emulator_model_sweep = gpyImprovedEmulator(flamingo_sweep, 'variance_weights_no', error=False, fix_variance=False, ARD=True, model_upload=True)
        pred[1:,:] = emulator_model_sweep.pred
            
    return pred

if __name__ == "__main__":

    method = 'BXL_emu'

    if method == 'BXL_emu':
        f = flamingoDMOData()
        f.weights()

        params = np.linspace(0,1,10)
        params = (params*(f.design_max[7]-f.design_min[7]))+f.design_min[7]
        #params = np.logspace(np.log10(f.design_min[9]), np.log10(f.design_max[9]), 10)
    
        test = np.tile(f.flamingo_parameters[8], 11).reshape(11,-1)
        print(test.shape)
        test[1:,7] = params
        
        print(test)
        print(test.shape)
        pred = parameter_sweep(test)
        print(test)

    elif method == 'CAMB':
        pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, As=2e-9, ns=0.965, halofit_version='mead')

    print(pred)

    for i in range(len(pred)):
        if i==0:
            pb.plot(f.k_test, f.Y_test[8,:], color='k', label='FLAMINGO (true)')
            pb.plot(f.k_test, pred[0], color='k', linestyle='dashed', label='FLAMINGO (predicted)')
        else:
            #pb.plot(f.k_test, pred[i], label=rf'$log_{10}$ (DM particle masses) = {np.log10(params[i-1])}')
            pb.plot(f.k_test, pred[i], label=rf'$\Omega_\nu h^2$ = {params[i-1]:06f}')

    pb.xlabel(r'$k \: [1/Mpc]$', fontsize=15)
    pb.ylabel(r'$P(k) \: [Mpc^3]$', fontsize=15)
    pb.xscale('log')
    pb.yscale('log')
    pb.title(f'GPy test with external cosmologies')
    pb.legend(loc='upper left', fontsize=10)
    
    pb.savefig(f'./Plots/parameter_sweep_test.png', dpi=1200)
    pb.clf()

    for i in range(len(pred)):
        if i==0:
            pb.hlines(xmin=-1, xmax=100, y=1, color='k', label=f'FLAMINGO ({test[0,0]})')
        else:
            #pb.plot(f.k_test, pred[i]/pred[0], label=rf'$log_{10}$ (DM particle masses) = {np.log10(params[i-1])}')
            pb.plot(f.k_test, pred[i]/pred[0], label=rf'$\Omega_\nu h^2$ = {params[i-1]:06f}')
    pb.xlabel(r'$k \: [1/Mpc]$', fontsize=15)
    pb.ylabel(r'$P(k) \: [Mpc^3]$', fontsize=15)
    pb.xscale('log')
    pb.xlim(left=1e-2, right=15)
    pb.title(f'Parameter sweeps predictions/FLAMINGO')
    pb.legend(loc='lower left', fontsize=6)
    pb.savefig(f'./Plots/parameter_sweep_residual.png', dpi=1200)
    pb.clf()
    
