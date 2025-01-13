import numpy as np
import matplotlib.pyplot as plt
import camb
import pylab as pb

flamingo_params = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_params.txt', usecols=np.arange(9), skiprows=1)

H0 = flamingo_params[2].copy()*100
ombh2 = (flamingo_params[0].copy()*flamingo_params[1].copy())*(flamingo_params[2].copy()**2)
omch2 = (flamingo_params[0].copy()*(1-flamingo_params[1].copy()))*(flamingo_params[2].copy()**2)-flamingo_params[7].copy()
As = flamingo_params[4].copy()
ns = flamingo_params[3].copy()
w0 = flamingo_params[5].copy()
wa = flamingo_params[6].copy()
mnu = 93.14 * flamingo_params[7].copy()
alpha_s = flamingo_params[8].copy()

print(H0,68.1)
print(ombh2,0.0225387846) 
print(omch2,0.11872745361) 
print(As,2.099e-09)
print(ns,0.967)

pars = camb.set_params(H0=H0,
                       ombh2=ombh2,
                       omch2=omch2,
                       ns=ns,
                       w=w0,
                       wa=wa,
                       mnu=mnu,
                       nrun=alpha_s,
                       nnu=3.046,
                       kmax=50,
                       dark_energy_model='DarkEnergyPPF',
                       As=As)

pars.set_matter_power(redshifts=[0], kmax=50)

pars.NonLinear = camb.model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=0.001, maxkh=30, npoints=300)

print(kh.shape)
print(pk.shape)

file = open(r'./BXL_data/FLAMINGO_data/FLAMINGO_camb_pk.txt', 'w')
np.savetxt('./BXL_data/FLAMINGO_data/FLAMINGO_camb_pk.txt', np.column_stack([kh,pk.reshape(-1,1)]))
file.close()

f_1GpcHR = np.loadtxt('./BXL_data/FLAMINGO_data/power_matter_L1000N3600_DMO_z0.txt', usecols=(1,2), skiprows=1)

pb.plot(kh*flamingo_params[2], pk[0,:]/(flamingo_params[2]**3), label='CAMB spectra')
pb.plot(f_1GpcHR[:,0], f_1GpcHR[:,1], label='FLAMINGO 1GpcHR')
pb.xscale('log')
pb.yscale('log')
pb.legend()
pb.savefig('./Plots/flamingo_camb.png', dpi=1200)
pb.clf()
