import numpy as np
import pylab as pb
import matplotlib.ticker as ticker

def hypercube_sampling():

    return

def hypercube_plot(data, parameter_labels, save_to, title, dim=(9,9), marker_colour=['tab:blue', 'tab:orange', 'tab:green'], marker_label=['Intermediate', 'Low', 'High'], legend_title='Resolution level:'):

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
                        ax[x,y].scatter(d[:, y], d[:, x], s=1, color=marker_colour[i], label=marker_label[i] if y==0 and x==1 else None)

                #Plotting data points when only one colour/label is applied
                else:
                    ax[x,y].scatter(data[:, y], data[:, x], s=1, color=marker_colour[0], label=marker_label[0] if y==0 and x==1 else None)

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
    fig.suptitle(title, fontsize=15)
    #Creating legend
    le = fig.legend(loc='center right', fontsize=9, title=legend_title)
    for l in range(len(marker_colour)):
        le.legend_handles[l]._sizes = [30]
    #Saving figure
    pb.savefig(save_to, dpi=1200)
    pb.clf()
    
    return
    
if __name__ == "__main__":
    from data_loader import BXL_DMO_Pk
    import random
    
    t = random.randint(0,149)
    data = BXL_DMO_Pk(t, 100, pk='nbk-rebin-std', lin='rebin', holdout=False, sigma8=True).parameters
    labels = {r'$\Omega_m$':[.20,.25,.30,.35], r'$f_b$':[.14,.15,.16,.17], r'$h_0$':[.64,.7,.76], r'$n_s$':[.95,.97,.99], r'$\sigma_8$':[0.72,0.80,0.88], r'$w_0$':[-.7,-.9,-1.1], r'$w_a$':[.2,-.5,-1.2], r'$\Omega_{\nu}h^2$':[.005,.003,.001], r'$\alpha_s$':[.025,0,-.025]}

    #hypercube = hypercube_plot(data, parameter_labels=labels, save_to='./Plots/BXL_hypercube.png', title='BAHAMAS XL Latin hypercube design', marker_colour=['tab:red'], marker_label=['All data'])
    hypercube = hypercube_plot(data=[data[:50, :], data[50:100, :], data[100:, :]], parameter_labels=labels, save_to='./Plots/BXL_hypercube.png', title='BAHAMAS XL Latin hypercube design')
