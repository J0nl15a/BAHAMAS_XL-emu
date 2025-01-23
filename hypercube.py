import math
import numpy as np
import pylab as pb
import matplotlib.ticker as ticker
from classy import Class
import camb
from scipy import interpolate

def bahamasxl_hypercube_sampling(number_of_samples, parameters=np.arange(9), save=False, labels=[], method='CLASS'):

    resample = np.full((number_of_samples, len(parameters)), True)
    test_parameters = np.zeros((number_of_samples, len(parameters)))
    print(test_parameters)
    print(resample)
    bxl_parameters = np.loadtxt('./BXL_data/slhs_nested_3x50_w0_m0p6_m1p2_wa_m1p6_p0p5_with_running_and_fgas_and_As.txt', skiprows=1, max_rows=150, usecols=parameters)
    bxl_As = bxl_parameters[:, 4].copy()
    sig8 = np.loadtxt('./BXL_data/slhs_nested_3x_Om_fb_h_ns_sigma8_w0_wa_Omnuh2_alphas_fgasfb.txt', max_rows=150, usecols=4)
    bxl_parameters[:, 4] = sig8
    DM_particle_mass = np.loadtxt('./BXL_data/DM_masses.txt').reshape(-1,1)
    bxl_parameters = np.hstack((bxl_parameters, DM_particle_mass))
    
    test_parameters = np.random.uniform([min(bxl_parameters[:,i]) for i in parameters], [max(bxl_parameters[:,i]) for i in parameters], (number_of_samples,len(parameters)))

    while True in resample:
        print(test_parameters)
        for s in range(number_of_samples):
            for p in range(len(parameters)):
                if test_parameters[s,p] in bxl_parameters[:,p]:
                    #if p==9:
                        #test_parameters[s,p] = np.random.uniform(np.log10(min(bxl_parameters[:,p])), np.log10(max(bxl_parameters[:,p])), 1)
                    #else:
                    test_parameters[s,p] = np.random.uniform(min(bxl_parameters[:,p]), max(bxl_parameters[:,p]), 1)
                resample[s,p]=False

    #test_parameters = np.hstack((test_parameters, np.zeros((number_of_samples,1))))
    default_As = 2e-9
    '''FLAMINGO params'''
    #flamingo = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_params.txt', skiprows=1, usecols=np.arange(10))
    #flamingo_As = flamingo[4].copy()
    #sig8 = np.loadtxt('./BXL_data/FLAMINGO_data/FLAMINGO_params.txt', skiprows=1, usecols=10)
    #flamingo[4] = sig8
    #test_parameters = flamingo.reshape(1,-1)
    '''BXL params'''
    test_parameters = np.hstack((bxl_parameters[0,:].reshape(1,-1), np.zeros((number_of_samples,1))))
    print(test_parameters)
    print(bxl_As[0])
    #print(flamingo_As)

    for s in range(number_of_samples): 
        if method == 'CLASS':
            
            params = {"h": test_parameters[s,2],
                      "Omega_cdm": test_parameters[s,0]-(test_parameters[s,0]*test_parameters[s,1])-(test_parameters[s,7]/(test_parameters[s,2]**2)),
                      "Omega_b": test_parameters[s,0]*test_parameters[s,1],
                      "Omega_Lambda": 0,  # use dark fluid instead
                      "Omega_ncdm": test_parameters[s,7]/(test_parameters[s,2]**2),
                      "N_ncdm": 1,
                      "deg_ncdm": 3,
                      "ncdm_fluid_approximation": 3,
                      "T_ncdm": 1.9517578050 / 2.7255, # specified in units of T_cmb
                      "T_cmb": 2.7255,
                      "N_ur": 0.00441,
                      "fluid_equation_of_state": "CLP",
                      "w0_fld": test_parameters[s,5],
                      "wa_fld": test_parameters[s,6],
                      "cs2_fld": 1.0,
                      "sigma8": test_parameters[s,4],
                      "n_s": test_parameters[s,3],
                      "k_pivot": 0.05,
                      "alpha_s": test_parameters[s,8],
                      "reio_parametrization": "reio_none",
                      "YHe": "BBN",
                      "compute damping scale": "yes",
                      "tol_background_integration": 1e-12,
                      "tol_ncdm_bg": 1e-10
                      }

            model = Class()
            model.set(params)
            model.compute()
            
            actual_As = model.get_current_derived_parameters(['A_s'])
            print(actual_As)

            test_parameters[s,-1] = actual_As['A_s']
        
            model.struct_cleanup()
            model.empty()
            
        elif method == 'CAMB':
            pars = camb.set_params(
                H0 = test_parameters[s,2]*100,
                ombh2 = (test_parameters[s,0] * test_parameters[s,1]) * test_parameters[s,2]**2,
                omch2 = ((test_parameters[s,0] - (test_parameters[s,0] * test_parameters[s,1])) * test_parameters[s,2]**2) - test_parameters[s,7],
                ns = test_parameters[s,3],
                w = test_parameters[s,5],
                wa = test_parameters[s,6],
                mnu = 93.14 * test_parameters[s,7],
                nrun = test_parameters[s,8],
                nnu = 3.046,
                kmax = 50,
                dark_energy_model = 'DarkEnergyPPF',
                As = default_As
            )
            pars.set_matter_power(redshifts=[0.], kmax=50)
            results = camb.get_results(pars)
            s8_fid = results.get_sigma8_0()
            pars.InitPower.set_params(As=default_As*(test_parameters[s,4]**2/s8_fid**2), ns=test_parameters[s,3], nrun=test_parameters[s,8])
            
            results = camb.get_results(pars)
            s8 = results.get_sigma8_0()
            print(s8, test_parameters[s,4])
            actual_As = default_As*(test_parameters[s,4]**2/s8_fid**2)
            test_parameters[s,-1] = actual_As

    if save==True:
        file = open(r'./BXL_data/External_models/External_params.txt', 'w')
        np.savetxt('./BXL_data/External_models/External_params.txt', '# Omega_m, f_b, h0, ns, sigma_8, w0, wa, Omeag_nuh^2, alpha_s, fgas/f_b, A_s')
        np.savetxt('./BXL_data/External_models/External_params.txt', np.column_stack([test_parameters]))
        file.close()

    return test_parameters


def hypercube_plot(data, parameter_labels, save_to, title, dim=(10,10), marker_colour=['tab:blue', 'tab:orange', 'tab:green'], marker_label=['Intermediate', 'Low', 'High'], legend_title='Resolution level:'):

    """
    A function to generate a scatter triangle plot from a simulation Latin hypercube.

    Parameters
    ----------
    :param data: Hypercube parameter values in a numpy array shaped as (number of cosmological models, number of parameters).
    :param parameter_labels: A dictionary containing the hypercube parameter labels as keys and the axis tick labels as values.
    :param save_to: A string for the path, filename and file extension to save the figure to.
    :param title: A string for the the title of the figure.
    :param dim: A tuple indicating the dimensions for the triangle plot. Default=9x9.
    :param marker_colour: A list of colours for the data points in the figure. Default=List of 3 tab colours.
    :param marker_label: A list of labels for the differently marked data points. Default=List of 3 marker labels.
    :param legend_title: A string for the legend title.

    Returns
    -------
    :figure: A figure of the hypercube as a scatter triangle plot.
    """
    
    fig, ax = pb.subplots(dim[0], dim[1], sharex='col', sharey='row')

    #Loops over the dimensions of the hypercube for both x and y axis
    for y in range(dim[1]):
        for x in range(dim[0]):
            #Plotting data in the lower left hand corner of the subplot
            if x > y:
                #Condition for when data points have different colours/labels 
                if len(marker_colour)>1:
                    #Plotting data points in each scatter subplot with desired label and colour
                    for i,d in enumerate(data):
                        if d.ndim>1:
                            ax[x,y].scatter(d[:, y], d[:, x], s=.5, color=marker_colour[i], label=marker_label[i] if y==0 and x==1 else None)
                        else:
                            ax[x,y].scatter(d[y], d[x], s=.5, color=marker_colour[i], label=marker_label[i] if y==0 and x==1 else None)

                #Plotting data points when only one colour/label is applied
                else:
                    ax[x,y].scatter(data[:, y], data[:, x], s=.5, color=marker_colour[0], label=marker_label[0] if y==0 and x==1 else None)

                #Assigning axis tick values on the plot from the values of the parameter_labels dict
                if y==0:
                    #y axis ticks
                    ax[x,y].tick_params(axis="y", labelsize=6, rotation=45)
                    ax[x,y].set_yticks(list(parameter_labels.values())[x])
                    ax[x,y].set_yticklabels(list(parameter_labels.values())[x])
                    #x axis ticks on corner plot
                    if x==dim[0]-1:
                        ax[x,y].tick_params(axis="x", labelsize=6, rotation=45)
                        ax[x,y].set_xticks(list(parameter_labels.values())[y])
                        ax[x,y].set_xticklabels(list(parameter_labels.values())[y])
                else:
                    #Removing axis ticks on left hand side of the remaining subplots
                    ax[x,y].tick_params(left=False)
                    #x axis ticks for remaining subplots
                    if x==dim[0]-1:
                        ax[x,y].tick_params(axis="x", labelsize=6, rotation=45)
                        ax[x,y].set_xticks(list(parameter_labels.values())[y])
                        ax[x,y].set_xticklabels(list(parameter_labels.values())[y])

            #For diagonal plots, axis are removed and parameter labels are assigned from the keys of the parameter_labels dict
            elif x==y:
                ax[x,y].axis("off")
                ax[x,y].text(0.5, 0.5, list(parameter_labels.keys())[x], horizontalalignment='center', verticalalignment='center', transform=ax[x,y].transAxes, fontsize=10)

            #Remove all other axis above the diagonal
            else:
                ax[x,y].remove()

        if y==9:
            ax[x,y].set_xscale('log')
            ax[x,y].set_yscale('log')
                
    #Removing whitespace
    fig.subplots_adjust(wspace=0,hspace=0)
    #Figure title
    fig.suptitle(title, fontsize=15, wrap=True)
    #Creating legend
    le = fig.legend(loc='center right', fontsize=9, title=legend_title)
    for l in range(len(marker_colour)):
        le.legend_handles[l]._sizes = [30]
    #Saving figure
    pb.savefig(save_to, dpi=1200)
    pb.clf()
    
    return

def bahamasxl_test_statistic(cosmo, k_test, stat='matter_power', save=False, method='CLASS', redshifts=[0]):
    print(redshifts)

    if stat == 'matter_power':
        for z in redshifts:
            print(cosmo)
            Pk_lin = np.zeros((len(cosmo), len(k_test)))
            Pk_nonlin = np.zeros((len(cosmo), len(k_test)))
            Pk_boost = np.zeros((len(cosmo), len(k_test)))
            for i,k in enumerate(cosmo):
                print(i)
                Omega_m = cosmo[i,0]
                f_b = cosmo[i,1]
                h0 = cosmo[i,2]
                ns = cosmo[i,3]
                sig8 = cosmo[i,4]
                w0 = cosmo[i,5]
                wa = cosmo[i,6]
                Omega_nuh2 = cosmo[i,7]
                alpha_s = cosmo[i,8]
                A_s = cosmo[i,9]

                omb = f_b*Omega_m
                omc = Omega_m - omb - (Omega_nuh2/h0**2)
                ombh2 = omb*(h0**2)
                omch2 = (Omega_m - omb)*(h0**2) - Omega_nuh2
                mnu = 93.14*Omega_nuh2

                if method == 'CLASS':
                    params = {"h": h0,
                              "Omega_cdm": omc,
                              "Omega_b": omb,
                              "Omega_Lambda": 0,  # use dark fluid instead
                              "Omega_ncdm": Omega_nuh2/h0**2,
                              "N_ncdm": 1,
                              "deg_ncdm": 3,
                              "ncdm_fluid_approximation": 3,
                              "T_ncdm": 1.9517578050 / 2.7255, # specified in units of T_cmb
                              "T_cmb": 2.7255,
                              "N_ur": 0.00441,
                              "fluid_equation_of_state": "CLP",
                              "w0_fld": w0,
                              "wa_fld": wa,
                              "cs2_fld": 1.0,
                              #"A_s": A_s[i],
                              "sigma8": sig8,
                              "n_s": ns,
                              "k_pivot": 0.05,
                              "alpha_s": alpha_s,
                              "reio_parametrization": "reio_none",
                              "YHe": "BBN",
                              "compute damping scale": "yes",
                              "tol_background_integration": 1e-12,
                              "tol_ncdm_bg": 1e-10,
                              "P_k_max_1/Mpc": 30,
                              "non linear": "halofit",
                              "output": "mPk",
                              "z_pk" : z
                    }
            
                    model = Class()
                    model.set(params)
                    model.compute()
                    
                    #redshift=params['z_pk']
                    #print(redshift)

                    for k in range(len(k_test)):
                        Pk_lin[i, k] = model.pk_lin(k_test[k], z)

                        Pk_nonlin[i, k] = model.pk(k_test[k], z)
                
                    model.struct_cleanup()
                    model.empty()
                      
                elif method == 'CAMB':
                    pars = camb.set_params(H0=h0*100,
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
                                           As=A_s)
                    pars.set_matter_power(redshifts=[0.], kmax=50)
                        
                    pars.NonLinear = camb.model.NonLinear_none
                    results = camb.get_results(pars)
                    kh_lin, z_lin, pk_lin = results.get_matter_power_spectrum(minkh=1e-3, maxkh=50, npoints = 300)
                    lin_func = interpolate.interp1d(kh_lin*h0, pk_lin/(h0**3), kind='cubic')
                    Pk_lin[i,:] = lin_func(k_test)
                
                    pars.NonLinear = camb.model.NonLinear_both
                    results.calc_power_spectra(pars)
                    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-3, maxkh=50, npoints = 300)
                    nonlin_func = interpolate.interp1d(kh*h0, pk/(h0**3), kind='cubic')
                    Pk_nonlin[i,:] = nonlin_func(k_test)

                Pk_boost[i,:] = Pk_nonlin[i,:]/Pk_lin[i,:]
                
                print(Pk_boost, Pk_nonlin, Pk_lin)

            if save==True:
                file = open(r'./BXL_data/External_models/External_pk.txt', 'w')
                np.savetxt('./BXL_data/External_models/External_pk.txt', '# z, Pk_boost, Pk_nonlin, Pk_lin')
                np.savetxt('./BXL_data/External_models/External_pk.txt', np.column_stack([z, Pk_boost, Pk_nonlin, Pk_lin]))
                file.close()

    if len(redshifts) == 1:
        return Pk_boost, Pk_nonlin, Pk_lin
    else:
        return
    
if __name__ == "__main__":
    from flamingo_data_loader import flamingoDMOData
    import random
    hr = random.randint(100,149)
    ir = random.randint(0,49)
    lr = random.randint(50,99)
    flam = flamingoDMOData()

    method = 'CLASS'
    test = bahamasxl_hypercube_sampling(1, np.arange(9), method=method)
    print(test)

    print(flam.parameters.shape)
    for x in range(test.shape[1]-1):
        if np.any(np.isin(test[:,x], flam.parameters[:,x])):
            print('FAIL')
            
    
    labels = {r'$\Omega_m$':[.20,.25,.30,.35], r'$f_b$':[.14,.15,.16,.17], r'$h_0$':[.64,.7,.76], r'$n_s$':[.95,.97,.99], r'$\sigma_8$':[0.72,0.80,0.88], r'$w_0$':[-.7,-.9,-1.1], r'$w_a$':[.2,-.5,-1.2], r'$\Omega_{\nu}h^2$':[.005,.003,.001], r'$\alpha_s$':[.024,.006,-.012], r'$DM_{mass}$':[1e9,1e10,1e11]}#, r'$\frac{f_{gas}}{f_b}$':[.4,.5,.6]}
    marker_colour=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']
    marker_label=['Intermediate', 'Low', 'High', 'Test', 'FLAMINGO']

    #hypercube = hypercube_plot(data, parameter_labels=labels, save_to='./Plots/BXL_hypercube.png', title='BAHAMAS XL Latin hypercube design', marker_colour=['tab:red'], marker_label=['All data'])
    hypercube = hypercube_plot(dim=(10,10), data=[flam.parameters[:50, :], flam.parameters[50:100, :], flam.parameters[100:, :], test, flam.flamingo_parameters], parameter_labels=labels, save_to='./Plots/BXL_hypercube_test.png', title='BAHAMAS XL Latin hypercube design w/ test points and FLAMINGO', marker_colour=marker_colour, marker_label=marker_label)
    #hypercube = hypercube_plot(test, parameter_labels=labels, save_to='./Plots/BXL_hypercube_alltest.png', title='BAHAMAS XL Latin hypercube design', marker_colour=['tab:red'], marker_label=['Test data'])

    for a,b in enumerate([flam.As[:50], flam.As[50:100], flam.As[100:], test[:,-1], flam.flamingo_As]):
        #pb.scatter(b, b, s=.5, color=marker_colour[a], label=marker_label[a])
        pb.hist(b/1e-9, color=marker_colour[a], label=marker_label[a])
    print(flam.flamingo_As)
    #pb.xlim(left=1e-9, right=1e-7)
    pb.title(r'$A_s$ values')
    pb.legend(loc='center right', fontsize=9, title='Resolution level:')
    pb.savefig('./Plots/A_s.png', dpi=1200)
    pb.clf()

    print(test)
    print(flam.As[0])

    Pk, Pk_non, Pk_lin = bahamasxl_test_statistic(test, flam.k_test, method=method)#np.append(flam.flamingo_parameters,flam.flamingo_As).reshape(1,-1))#np.append(flam.parameters[0, :]), flam.As[0]).reshape(1,-1)
    print(Pk, Pk_non, Pk_lin)

    for m in range(len(test)):
        pb.plot(flam.k_test, Pk_non[m,:], color='tab:purple', label=f'Non-linear {method} P(k)' if m==0 else None)
        pb.plot(flam.k_test, Pk_lin[m,:], color='tab:purple', linestyle='dashed', label=f'Linear {method} P(k)' if m==0 else None)    
    pb.plot(flam.k[ir,:], flam.P_k[ir,:], color ='tab:blue', label='Non-linear IR BXL P(k)')
    pb.plot(flam.k_linear[ir,:], flam.P_k_linear[ir,:], color ='tab:blue', linestyle='dashed', label='Linear IR BXL P(k)')
    pb.plot(flam.k[lr,:], flam.P_k[lr,:], color ='tab:orange', label='Non-linear LR BXL P(k)')
    pb.plot(flam.k_linear[lr,:], flam.P_k_linear[lr,:], color ='tab:orange', linestyle='dashed', label='Linear LR BXL P(k)')
    pb.plot(flam.k[hr,:], flam.P_k[hr,:], color ='tab:green', label='Non-linear HR BXL P(k)')
    pb.plot(flam.k_linear[hr,:], flam.P_k_linear[hr,:], color ='tab:green', linestyle='dashed', label='Linear HR BXL P(k)')
    pb.plot(flam.flamingo_k[8], flam.flamingo_P_k[8], color ='tab:pink', label='Non-linear FLAMINGO fiducial P(k)')
    pb.plot(flam.flamingo_k_linear, flam.flamingo_Pk_linear, color ='tab:pink', linestyle='dashed', label='Linear FLAMINGO fiducial P(k)')
    pb.xscale('log')
    pb.yscale('log')
    pb.legend()
    pb.savefig('./Plots/test_Pk.png', dpi=1200)
    pb.clf()

    for m in range(len(test)):
        pb.plot(flam.k_test, Pk[m,:], color='tab:purple', label=f'{method} boost' if m==0 else None)
    pb.plot(flam.k_test, flam.P_k_nonlinear[ir,:], color='tab:blue', label='BXL IR boost')
    pb.plot(flam.k_test, flam.P_k_nonlinear[lr,:], color='tab:orange', label='BXL LR boost')
    pb.plot(flam.k_test, flam.P_k_nonlinear[hr,:], color='tab:green', label='BXL HR boost')
    pb.plot(flam.k_test, flam.flamingo_P_k_nonlinear[8], color='tab:pink', label='FLAMINGO boost')
    pb.xscale('log')
    pb.yscale('log')
    pb.legend()
    pb.savefig('./Plots/test_boost.png', dpi=1200)
    pb.clf()

    '''pb.plot(flam.k_test, Pk[m,:]/flam.flamingo_P_k_nonlinear[8])
    pb.title(f'Ratio of {method} to FLAMINGO')
    pb.xscale('log')
    pb.savefig('./Plots/test_ratio.png', dpi=1200)
    pb.clf()'''
