import numpy as np
import matplotlib.pyplot as plt
import camb

pars = camb.CAMBparams()
pars.set_cosmology(H0=68.1, ombh2=0.0225387846, omch2=0.11872745361)
pars.InitPower.set_params(As=2.099e-09, ns=0.967)
pars.set_matter_power(redshifts=[0], kmax=50)

pars.NonLinear = camb.model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=0.001, maxkh=30, npoints=300)

print(kh.shape)
print(pk.shape)

file = open(r'./BXL_data/FLAMINGO_camb_pk.txt', 'w')
np.savetxt('./BXL_data/FLAMINGO_camb_pk.txt', np.column_stack([kh,pk.reshape(-1,1)]))
file.close()

