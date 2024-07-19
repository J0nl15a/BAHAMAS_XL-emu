import math
import numpy as np
import pylab as pb
import matplotlib.ticker as ticker
from classy import Class

def bahamasxl_hypercube_sampling(number_of_samples, parameters=np.arange(10), save=False, labels=[]):

    resample = np.full((number_of_samples, len(parameters)), True)
    test_parameters = np.zeros((number_of_samples, len(parameters)))
    print(test_parameters)
    print(resample)
    bxl_parameters = np.loadtxt('./BXL_data/slhs_nested_3x50_w0_m0p6_m1p2_wa_m1p6_p0p5_with_running_and_fgas_and_As.txt', skiprows=1, max_rows=150, usecols=parameters)
    sig8 = np.loadtxt('./BXL_data/slhs_nested_3x_Om_fb_h_ns_sigma8_w0_wa_Omnuh2_alphas_fgasfb.txt', max_rows=150, usecols=4)
    bxl_parameters[:, 4] = sig8

    test_parameters = np.random.uniform([min(bxl_parameters[:,i]) for i in parameters], [max(bxl_parameters[:,i]) for i in parameters], (number_of_samples,len(parameters)))

    while True in resample:
        print(test_parameters)
        for s in range(number_of_samples):
            for p in range(len(parameters)):
                if test_parameters[s,p] in bxl_parameters[:,p]:
                    test_parameters[s,p] = np.random.uniform(min(bxl_parameters[:,p]), max(bxl_parameters[:,p]), 1)
                resample[s,p]=False

    test_parameters = np.hstack((test_parameters, test_parameters[:,4].reshape(-1,1)))
    for s in range(number_of_samples): 
        params = {"h" : test_parameters[s,2],
                  "Omega_b" : test_parameters[s,0]*test_parameters[s,1],
                  "Omega_cdm" : 1-(test_parameters[s,0]*test_parameters[s,1]),
                  "output": "mPk",
                  "z_pk" : 0,
                  "A_s" : 2.1e-9,
                  "n_s" : test_parameters[s,3],
                  "alpha_s": test_parameters[s,8],
        }

        model = Class()
        model.set(params)
        model.compute()
        print(params['A_s'])

        class_sigma8 = model.sigma8()
        print(class_sigma8)
        print(test_parameters[s,4])

        #actual_As_v1 = math.sqrt((params['A_s']/class_sigma8)**2) * test_parameters[s,4]
        actual_As = math.sqrt(((params['A_s']**2)/(class_sigma8**2))*(test_parameters[s,4]**2))
        #actual_As_v3 = (params['A_s']/class_sigma8) * test_parameters[s,4]

        #print(actual_As_v1)
        print(actual_As)
        #print(actual_As_v3)

        test_parameters[s,4] = actual_As
        
        model.struct_cleanup()
        model.empty()

    if save==True:
        file = open(r'./BXL_data/External_models/External_params.txt', 'w')
        np.savetxt('./BXL_data/External_models/External_params.txt', '# Omega_m, f_b, h0, ns, A_s, w0, wa, Omeag_nuh^2, alpha_s, fgas/f_b, sigma_8')
        #for p in param:
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
                            ax[x,y].scatter(d[:, y] if y!=4 else d[:, -1], d[:, x] if x!=4 else d[:, -1], s=.5, color=marker_colour[i], label=marker_label[i] if y==0 and x==1 else None)
                        else:
                            ax[x,y].scatter(d[y] if y!=4 else d[-1], d[x] if x!=4 else d[-1], s=.5, color=marker_colour[i], label=marker_label[i] if y==0 and x==1 else None)

                #Plotting data points when only one colour/label is applied
                else:
                    ax[x,y].scatter(data[:, y] if y!=4 else d[:, -1], data[:, x] if x!=4 else d[:, -1], s=.5, color=marker_colour[0], label=marker_label[0] if y==0 and x==1 else None)

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
    
if __name__ == "__main__":
    from flamingo_data_loader import flamingoDMOData
    import random
    flam = flamingoDMOData(sigma8=True)
        
    test = bahamasxl_hypercube_sampling(10, np.arange(9))
    print(test)
    for x in range(test.shape[1]-1):
        if test[:,x] in flam.parameters[:,x]:
            print('FAIL')
            
    
    labels = {r'$\Omega_m$':[.20,.25,.30,.35], r'$f_b$':[.14,.15,.16,.17], r'$h_0$':[.64,.7,.76], r'$n_s$':[.95,.97,.99], r'$\sigma_8$':[0.72,0.80,0.88], r'$w_0$':[-.7,-.9,-1.1], r'$w_a$':[.2,-.5,-1.2], r'$\Omega_{\nu}h^2$':[.005,.003,.001], r'$\alpha_s$':[.025,0,-.025]}#, r'$\frac{f_{gas}}{f_b}$':[.4,.5,.6]}
    marker_colour=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:grey']
    marker_label=['Intermediate', 'Low', 'High', 'Test', 'FLAMINGO']

    #hypercube = hypercube_plot(data, parameter_labels=labels, save_to='./Plots/BXL_hypercube.png', title='BAHAMAS XL Latin hypercube design', marker_colour=['tab:red'], marker_label=['All data'])
    hypercube = hypercube_plot(dim=(9,9), data=[flam.parameters_sig8[:50, :], flam.parameters_sig8[50:100, :], flam.parameters_sig8[100:, :], test, flam.flamingo_parameters_sig8], parameter_labels=labels, save_to='./Plots/BXL_hypercube_test.png', title='BAHAMAS XL Latin hypercube design w/ test points and FLAMINGO', marker_colour=marker_colour, marker_label=marker_label)
    #hypercube = hypercube_plot(test, parameter_labels=labels, save_to='./Plots/BXL_hypercube_alltest.png', title='BAHAMAS XL Latin hypercube design', marker_colour=['tab:red'], marker_label=['Test data'])

    for a,b in enumerate([flam.parameters_sig8[:50, :], flam.parameters_sig8[50:100, :], flam.parameters_sig8[100:, :], test, flam.flamingo_parameters_sig8]):

        if b.ndim>1:
            pb.scatter(b[:, 4], b[:, 4], s=.5, color=marker_colour[a], label=marker_label[a])
        else:
            pb.scatter(b[4], b[4], s=.5, color=marker_colour[a], label=marker_label[a])

    pb.title(r'$A_s$ values')
    pb.legend(loc='center right', fontsize=9, title='Resolution level:')
    pb.savefig('./Plots/A_s.png', dpi=1200)
    pb.clf()
