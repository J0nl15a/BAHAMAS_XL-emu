import numpy as np
import pylab as pb
import scipy.stats as st
import matplotlib.figure as fi
from data_loader import bahamasXLDMOData
from flamingo_data_loader import flamingoDMOData
from gpy_emulator_improved import gpyImprovedEmulator

def parameter_sweep(parameters):

    pred = []
    for p in range(parameters.shape[0]):
        flamingo = flamingoDMOData(test_cosmology=test[p,:], log=False)
        parameters_norm = (parameters-flamingo.design_min)/(flamingo.design_max-flamingo.design_min)
        if p == 0:
            flamingo.weights()
            emulator_model = gpyImprovedEmulator(flamingo, 'variance_weights', fix_variance=True, ARD=True, save=True)
        else:
            emulator_model = gpyImprovedEmulator(flamingo, 'variance_weights', fix_variance=True, ARD=True, model_upload=True)
        pred.append(emulator_model.pred)
    
    return pred

if __name__ == "__main__":

    f = flamingoDMOData(log=False)
    Omega_m_dist = st.uniform(f.design_min[0], f.design_max[0]-f.design_min[0])
    f_b_dist = st.uniform(f.design_min[1], f.design_max[1]-f.design_min[1])
    h_0_dist = st.uniform(f.design_min[2], f.design_max[2]-f.design_min[2])
    n_s_dist = st.uniform(f.design_min[3], f.design_max[3]-f.design_min[3])
    A_s_dist = st.uniform(f.design_min[4], f.design_max[4]-f.design_min[4])
    w_0_dist = st.uniform(f.design_min[5], f.design_max[5]-f.design_min[5])
    w_a_dist = st.uniform(f.design_min[6], f.design_max[6]-f.design_min[6])
    Omega_nu_dist = st.uniform(f.design_min[7], f.design_max[7]-f.design_min[7])
    alpha_s_dist = st.uniform(f.design_min[8], f.design_max[8]-f.design_min[8])

    params = np.linspace(0,1,10)
    params = (params*(f.design_max[8]-f.design_min[8]))+f.design_min[8]
    
    test = np.tile(f.flamingo_parameters, 11).reshape(11,-1)
    print(test)
    Omega_m_samples = Omega_m_dist.rvs(1)
    f_b_samples = f_b_dist.rvs(1)
    h_0_samples = h_0_dist.rvs(1)
    n_s_samples = n_s_dist.rvs(1)
    A_s_samples = A_s_dist.rvs(1)
    w_0_samples = w_0_dist.rvs(1)
    w_a_samples = w_a_dist.rvs(1)
    Omega_nu_samples = Omega_nu_dist.rvs(1)
    alpha_s_samples = alpha_s_dist.rvs(9)
    print(Omega_m_samples)
    #test[1,0] = Omega_m_samples
    #test[2,1] = f_b_samples
    #test[3,2] = h_0_samples
    #test[4,3] = n_s_samples
    #test[5,4] = A_s_samples
    #test[6,5] = w_0_samples
    #test[7,6] = w_a_samples
    #test[8,7] = Omega_nu_samples
    #test[1:,8] = alpha_s_samples
    test[1:,8] = params
    #print(flamingo.design_min)
    #print(flamingo.design_max)
    print(test)
    pred = parameter_sweep(test)
    print(test)
    print(pred)

    for i in range(len(pred)):
        if i==0:
            pb.plot(f.k_test, pred[0], color='k', label='FLAMINGO')
        else:
            pb.plot(f.k_test, pred[i], label=rf'$\alpha_s$ = {params[i-1]}')

    pb.xlabel(r'$k \: [1/Mpc]$', fontsize=15)
    pb.ylabel(r'$P(k) \: [Mpc^3]$', fontsize=15)
    pb.xscale('log')
    pb.yscale('log')
    pb.title(f'GPy test with external cosmologies')
    pb.legend(loc='upper left', fontsize=10)
    
    pb.savefig(f'./Plots/parameter_sweep_test.png', dpi=1200)
    pb.clf()
    
