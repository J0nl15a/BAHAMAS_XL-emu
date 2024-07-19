from classy import Class
import numpy as np
import pylab as pb
from scipy import interpolate
from flamingo_data_loader import flamingoDMOData

flamingo = flamingoDMOData(pk='powmes', lin='class', flamingo_lin='camb', boost=False)

### Ian's example code for spectra

z = 0
k = flamingo.flamingo_k[9]
#kmin = 1e-3
kmax = max(k)
#nk = 100

# Cosmological parameters
h = flamingo.flamingo_parameters[2]
omega_m = flamingo.flamingo_parameters[0]
f_b = flamingo.flamingo_parameters[1]
omega_b = omega_m*f_b
omega_cdm = omega_m-omega_b
A_s = flamingo.flamingo_parameters[4]
n_s = flamingo.flamingo_parameters[3]

params = {"h" : h,
          "Omega_b" : omega_b,
          "Omega_cdm" : omega_cdm,
          "output": "mPk",
          "z_pk" : z,
          "A_s" : A_s,
          "n_s" : n_s,
          "P_k_max_1/Mpc" : kmax,
          }

# Run CLASS
model = Class()
model.set(params)
model.compute()

# Wavenumbers in 1/Mpc
Pk = np.zeros(len(k))
Pk_true = np.loadtxt('./BXL_data/FLAMINGO_data/power_matter_L1000N3600_DMO_z0.txt', skiprows=20, usecols=(1,2))

# Extract the power spectrum (in Mpc^3)
for i in range(len(k)):
    Pk[i] = model.pk(k[i], z)

file = open(r'./BXL_data/FLAMINGO_data/FLAMINGO_CLASS_linear_pk.txt', 'w')
np.savetxt('./BXL_data/FLAMINGO_data/FLAMINGO_CLASS_linear_pk.txt', np.column_stack([k,Pk.reshape(-1,1)]))
file.close()

print(Pk_true[:,0])
quit()
pb.plot(k, Pk, label='CLASS Pk')
pb.plot(Pk_true[:,0], Pk_true[:,1], label='FLAMINGO spectra (L1000N3600)')
pb.title(r'Nonlinear $P(k)$ from CLASS using FLAMINGO cosmology', fontsize=10, wrap=True)
pb.xlabel(r'$k \: [1/Mpc]$')
pb.ylabel(r'$P(k) \: [Mpc^3]$')
pb.xscale('log')
pb.yscale('log')
pb.legend(fontsize=5)
pb.savefig(f'./Plots/CLASS_spectra.png', dpi=1200)
pb.clf()

f = interpolate.interp1d(k, Pk, kind='cubic', bounds_error=False, fill_value=np.nan)
Pk_interp = f(Pk_true[:,0])

pb.plot(Pk_true[:,0], Pk_interp/Pk_true[:,1])
pb.title(r'Residual $P(k)$ of CLASS/FLAMINGO', fontsize=10, wrap=True)
pb.xlabel(r'$k \: [1/Mpc]$')
pb.ylabel(r'$P(k) \: [Mpc^3]$')
pb.xscale('log')
pb.yscale('log')
pb.legend(fontsize=5)
pb.savefig(f'./Plots/CLASS_spectra_residual.png', dpi=1200)
pb.clf()
quit()
    
for f in range(len(flamingo.flamingo_k)):
    if f==9:
        pass
    elif f in [0,1,3,7,12]:
        pb.plot(flamingo.flamingo_k[f], flamingo.flamingo_P_k[f], label='L'+f'{flamingo.flamingo_sims[f][0]}'+'N'+f'{flamingo.flamingo_sims[f][1]}', linestyle='dotted')
    else:
        pb.plot(flamingo.flamingo_k[f], flamingo.flamingo_P_k[f], label='L'+f'{flamingo.flamingo_sims[f][0]}'+'N'+f'{flamingo.flamingo_sims[f][1]}', linestyle='dashed')
    
pb.plot(k, Pk, label='CLASS Pk', color='k')
pb.plot(k, Pk_true, label='FLAMINGO spectra (L1000N3600)', color='k', alpha=.5)
pb.title(r'Nonlinear $P(k)$ from CLASS using FLAMINGO cosmology', fontsize=10, wrap=True)
pb.xlabel(r'$k \: [1/Mpc]$')
pb.ylabel(r'$P(k) \: [Mpc^3]$')
pb.xscale('log')
pb.yscale('log')
pb.legend(fontsize=5)
pb.savefig(f'./Plots/CLASS_spectra.png', dpi=1200)
pb.clf()

pb.hlines(y=1.000, xmin=-1, xmax=1000, color='k', linestyles='solid', alpha=0.5, label=None)
pb.hlines(y=0.990, xmin=-1, xmax=1000, color='k', linestyles='dashed', alpha=0.5, label='1% error')
pb.hlines(y=1.010, xmin=-1, xmax=1000, color='k', linestyles='dashed', alpha=0.5, label=None)
pb.hlines(y=0.950, xmin=-1, xmax=1000, color='k', linestyles='dotted', alpha=0.5, label='5% error')
pb.hlines(y=1.050, xmin=-1, xmax=1000, color='k', linestyles='dotted', alpha=0.5, label=None)

for f in range(len(flamingo.flamingo_k)):
    Pk = np.zeros(len(flamingo.flamingo_k[f]))

    params = {"h" : h,
              "Omega_b" : omega_b,
              "Omega_cdm" : omega_cdm,
              "output": "mPk",
              "z_pk" : z,
              "A_s" : A_s,
              "n_s" : n_s,
              "non linear" : "halofit",
              "P_k_max_1/Mpc" : max(flamingo.flamingo_k[f])*1.05,
    }

    model = Class()
    model.set(params)
    model.compute()
    
    for i in range(len(flamingo.flamingo_k[f])):
        Pk[i] = model.pk(flamingo.flamingo_k[f][i], z)

    if f<10:
        pb.plot(flamingo.flamingo_k[f], Pk/flamingo.flamingo_P_k[f], label='L'+f'{flamingo.flamingo_sims[f][0]}'+'N'+f'{flamingo.flamingo_sims[f][1]}')
    else:
        pb.plot(flamingo.flamingo_k[f], Pk/flamingo.flamingo_P_k[f], label='L'+f'{flamingo.flamingo_sims[f][0]}'+'N'+f'{flamingo.flamingo_sims[f][1]}', linestyle='dotted')

pb.title(r'Difference in $P(k)$ from CLASS against POWMES FLAMINGO spectra', fontsize=10, wrap=True)
pb.xlabel(r'$k \: [1/Mpc]$')
pb.ylabel(r'Precision')
pb.xscale('log')
pb.xlim(right=1000)
pb.ylim(top=1.2, bottom=0.8)
pb.legend(fontsize=5)
pb.savefig(f'./Plots/CLASS_spectra_resdiual.png', dpi=1200)
pb.clf()
