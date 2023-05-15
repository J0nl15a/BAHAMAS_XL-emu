import numpy as np
import matplotlib.pyplot as plt
from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
import sepia.SepiaPlot as SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
from scipy import interpolate
import random
from sepia_matter_power_spectra_z0_nonlinear_v3 import emulator_nonlinear
from sepia_matter_power_spectra_z0_nonlinear_dimensionless import emulator_dimensionless
#from sepia_matter_power_spectra_z0_median import emulator_median

cutoff = 10
boxlimit = (2*np.pi)/700

print(cutoff)

bins=100

#test_model = random.randint(0, 49)
#test_model = 49

#High Omega_nuh^2
#test_model = 28

#Low Omega_nuh^2
#test_model = 11

#print(test_model)

#Some setup parameters for the model that I don't really touch now
pc = 0.99
samp = 50
step = 20
mcmc = 1000
num_models = 2
Sens = True

sens_main_list = []
sens_total_list = []

test = 14

if test == 1:
        errors = []
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        for _ in range(10):
                n = random.randint(0, 149)
                test_k, error, test_models = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False)
                plt.plot(test_k, error.reshape(100, -1), label=('Model_' + f"{test_models:03d}"))
                errors.append(error)
                print(error)
        plt.title('Non-linear response of nbodykit P(k)', wrap=True)
        plt.xlim(left=0.01, right=5)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_nbk.pdf', dpi=800)
        plt.clf()
        errors = np.array(errors)
        errors = errors.reshape(10, -1)
        print('errors = ' + str(errors))
        mean_error = np.mean(errors, axis=0)
        median_error = np.median(errors, axis=0)
        sigma1_error = (np.quantile(errors, 0.8413)-np.quantile(errors, 0.1587))/2
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        plt.plot(test_k, mean_error)
        plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        plt.title('Mean error for Non-linear response of nbodykit P(k) with 10 models', wrap=True)
        plt.xlim(left=0.01, right=5)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        #plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_nbk_crosstest.pdf', dpi=800)
        plt.clf()

elif test == 11:
        errors = []
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        for i in range(3):
                if i == 0:
                        n = random.randint(0, 49)
                elif i == 1:
                        n = random.randint(50, 99)
                elif i == 2:
                        n = random.randint(100, 149)
                test_k, error, test_models = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False)
                plt.plot(test_k, error.reshape(100, -1), label=('Model_' + f"{test_models:03d}"))
                errors.append(error)
                print(error)
        plt.title('Non-linear response of nbodykit P(k)', wrap=True)
        plt.xlim(left=0.01, right=5)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_nbk_v2.pdf', dpi=800)
        plt.clf()
        errors = np.array(errors)
        errors = errors.reshape(3, -1)
        print('errors = ' + str(errors))
        mean_error = np.mean(errors, axis=0)
        median_error = np.median(errors, axis=0)
        sigma1_error = (np.quantile(errors, 0.8413)-np.quantile(errors, 0.1587))/2
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        plt.plot(test_k, mean_error)
        plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        plt.title('Mean error for Non-linear response of nbodykit P(k) with 3 models', wrap=True)
        plt.xlim(left=0.01, right=5)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        #plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_nbk_crosstest_v2.pdf', dpi=800)
        plt.clf()

elif test == 12:
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(0, 49)
        test_k2, error2, test_models2 = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'sim-only')
        test_k, error, test_models = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'rag-obs')
        plt.plot(test_k, error.reshape(100, -1), label='Ragged observations method')
        plt.plot(test_k2, error2.reshape(100, -1), label='Simulation only method')
        plt.title('Emulator error for model_' + f"{n:03d}" + ' with different SEPIA methods', wrap=True)
        plt.xlim(left=0.01, right=5)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_nbk_method_comparison.pdf', dpi=800)
        plt.clf()

elif test == 13:
        fig, axs = plt.subplots(1, 3, tight_layout=True, sharey=True, gridspec_kw={'wspace': 0})
        #fig.subplots_adjust(wspace=0)
        axs[0].hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        axs[0].hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        axs[0].hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        axs[0].hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        axs[0].hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(0, 49)
        test_k2, error2, test_models2 = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'sim-only', rebin=False)
        test_k, error, test_models = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'rag-obs', rebin=False)
        axs[0].plot(test_k, error.reshape(100, -1), label='Ragged observations method')
        axs[0].plot(test_k2, error2.reshape(100, -1), label='Simulation only method')
        axs[0].set_title('Model_' + f"{n:03d}", wrap=True)
        axs[0].set_xlim(left=0.01, right=5)
        axs[0].set_xscale('log')
        axs[0].set_ylabel('Residual Error', fontsize = 10)
        axs[0].legend(fontsize=6)
        axs[1].hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        axs[1].hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        axs[1].hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        axs[1].hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        axs[1].hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(50, 99)
        test_k2, error2, test_models2 = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'sim-only', rebin=False)
        test_k, error, test_models = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'rag-obs', rebin=False)
        axs[1].plot(test_k, error.reshape(100, -1), label='Ragged observations method')
        axs[1].plot(test_k2, error2.reshape(100, -1), label='Simulation only method')
        axs[1].set_title('Model_' + f"{n:03d}", wrap=True)
        axs[1].set_xlim(left=0.01, right=5)
        axs[1].set_xscale('log')
        plt.xlabel('k [1/Mpc]')
        #plt.ylabel('Residual Error')
        #plt.legend(fontsize=6)
        axs[2].hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        axs[2].hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        axs[2].hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        axs[2].hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        axs[2].hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(100, 149)
        test_k2, error2, test_models2 = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'sim-only', rebin=False)
        test_k, error, test_models = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'rag-obs', rebin=False)
        axs[2].plot(test_k, error.reshape(100, -1), label='Ragged observations method')
        axs[2].plot(test_k2, error2.reshape(100, -1), label='Simulation only method')
        axs[2].set_title('Model_' + f"{n:03d}", wrap=True)
        axs[2].set_xlim(left=0.01, right=5)
        axs[2].set_xscale('log')
        #plt.ylabel('Residual Error')
        #plt.legend(fontsize=6)
        fig.suptitle('Emulator comparison with different SEPIA integrations', wrap=True)
        #plt.xlabel('k [1/Mpc]')
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_nbk_method_comparison_v2.pdf', dpi=800)
        plt.clf()

elif test == 14:
        errors = []
        errors2 = []
        fig, axs = plt.subplots(1, 3, tight_layout=True, sharey=True, gridspec_kw={'wspace': 0})
        axs[0].hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        axs[0].hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        axs[0].hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        axs[0].hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        axs[0].hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        for _ in range(5):
                n = random.randint(0, 49)
                test_k2, error2, test_models2 = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'sim-only', rebin=False)
                test_k, error, test_models = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'rag-obs', rebin=False)
                errors.append(error)
                errors2.append(error2)
        errors = np.array(errors)
        errors = errors.reshape(5, -1)
        errors2 = np.array(errors2)
        errors2 = errors2.reshape(5, -1)
        mean_error = np.mean(errors, axis=0)
        sigma1_error = (np.quantile(errors, 0.8413)-np.quantile(errors, 0.1587))/2
        mean_error2 = np.mean(errors2, axis=0)
        sigma1_error2 = (np.quantile(errors2, 0.8413)-np.quantile(errors2, 0.1587))/2
        axs[0].plot(test_k, mean_error.reshape(100, -1), label='Ragged observations method')
        axs[0].plot(test_k2, mean_error2.reshape(100, -1), label='Simulation only method')
        axs[0].fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        axs[0].fill_between(test_k2, mean_error2+sigma1_error2, mean_error2-sigma1_error2, linewidth=0, alpha=0.3)
        axs[0].set_title('Models 0-49', wrap=True)
        axs[0].set_xlim(left=0.01, right=5)
        axs[0].set_xscale('log')
        axs[0].set_ylabel('Mean Residual Error', fontsize = 10)
        axs[0].legend(fontsize=6)
        errors = []
        errors2 = []
        axs[1].hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        axs[1].hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        axs[1].hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        axs[1].hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        axs[1].hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        for _ in range(5):
                n = random.randint(50, 99)
                test_k2, error2, test_models2 = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'sim-only', rebin=False)
                test_k, error, test_models = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'rag-obs', rebin=False)
                errors.append(error)
                errors2.append(error2)
        errors = np.array(errors)
        errors = errors.reshape(5, -1)
        errors2 = np.array(errors2)
        errors2 = errors2.reshape(5, -1)
        mean_error = np.mean(errors, axis=0)
        sigma1_error = (np.quantile(errors, 0.8413)-np.quantile(errors, 0.1587))/2
        mean_error2 = np.mean(errors2, axis=0)
        sigma1_error2 = (np.quantile(errors2, 0.8413)-np.quantile(errors2, 0.1587))/2
        axs[1].plot(test_k, mean_error.reshape(100, -1), label='Ragged observations method')
        axs[1].plot(test_k2, mean_error2.reshape(100, -1), label='Simulation only method')
        axs[1].fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        axs[1].fill_between(test_k2, mean_error2+sigma1_error2, mean_error2-sigma1_error2, linewidth=0, alpha=0.3)
        axs[1].set_title('Models 50-99', wrap=True)
        axs[1].set_xlim(left=0.01, right=5)
        axs[1].set_xscale('log')
        axs[1].set_xlabel('k [1/Mpc]', fontsize = 10)
        errors = []
        errors2 = []
        axs[2].hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        axs[2].hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        axs[2].hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        axs[2].hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        axs[2].hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        for _ in range(5):
                n = random.randint(100, 149)
                test_k2, error2, test_models2 = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'sim-only', rebin=False)
                test_k, error, test_models = emulator_nonlinear(bins, n, pc, samp, step, mcmc, pk = 'nbk', plot_k = False, method = 'rag-obs', rebin=False)
                errors.append(error)
                errors2.append(error2)
        errors = np.array(errors)
        errors = errors.reshape(5, -1)
        errors2 = np.array(errors2)
        errors2 = errors2.reshape(5, -1)
        mean_error = np.mean(errors, axis=0)
        sigma1_error = (np.quantile(errors, 0.8413)-np.quantile(errors, 0.1587))/2
        mean_error2 = np.mean(errors2, axis=0)
        sigma1_error2 = (np.quantile(errors2, 0.8413)-np.quantile(errors2, 0.1587))/2
        axs[2].plot(test_k, mean_error.reshape(100, -1), label='Ragged observations method')
        axs[2].plot(test_k2, mean_error2.reshape(100, -1), label='Simulation only method')
        axs[2].fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        axs[2].fill_between(test_k2, mean_error2+sigma1_error2, mean_error2-sigma1_error2, linewidth=0, alpha=0.3)
        axs[2].set_title('Models 100-149', wrap=True)
        axs[2].set_xlim(left=0.01, right=5)
        axs[2].set_xscale('log')
        fig.suptitle('Emulator comparison with different SEPIA integrations (Average error with 10 holdout samples)', wrap=True)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_nbk_method_comparison_v3.pdf', dpi=800)
        plt.clf()
        
elif test == 2:
        k, P_k_median, start, end = emulator_median(cutoff, test_model=1, pc=pc, samp=samp, step=step, mcmc=mcmc)
        for i in range(50):
        	plt.plot(k, P_k_median[i, :], label = 'model_' + f"{i:03d}")
        plt.title('Non-linear P(k) normalised by median (' + f"{mcmc:01d}" + ' runs with ' + f"{pc:01d}" + 'pcs)')
        plt.xlim(left=0.007, right=12)
        plt.xlabel('k (1/Mpc)')
        plt.ylabel('P(k) (Mpc^3)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize=6)
        plt.show()
        plt.savefig('./Plots/Power_spectra_z0_predicted_median_all.pdf', dpi=800)
        plt.clf()

elif test == 3:
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        for n in range(1, 4):
                test_k, error, sens, test_start, test_end = emulator_dimensionless(bins, n, pc, samp, step, mcmc)
                mean_error = np.mean(error, axis=0)
                median_error = np.median(error, axis=0)
                sigma1_error = (np.quantile(error, 0.8413)-np.quantile(error, 0.1587))/2
                max_error = np.max(error, axis=0)
                min_error = np.min(error, axis=0)
                plt.plot(test_k, mean_error, label=('Models ' + f"{test_start:03d}" + ' - ' + f"{test_end:03d}"))
                plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
                #plt.fill_between(test_k, max_error, min_error, linewidth=0, alpha=0.3)
        plt.title('Mean error for Non-linear response of P(k) across different training/testing configurations', wrap=True)
        plt.xlim(left=0.01, right=7)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_dimensionless_crosstest.pdf', dpi=800)
        plt.clf()

elif test == 4:
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        for n in range(1, 4):
                test_k, error, test_start, test_end = emulator_nonlinear(bins, n, pc, samp, step, mcmc)
                test_k_dl, error_dl, sens, test_start_dl, test_end_dl = emulator_dimensionless(bins, n, pc, samp, step, mcmc)
                mean_error = np.mean(error, axis=0)
                sigma1_error = (np.quantile(error, 0.8413)-np.quantile(error, 0.1587))/2
                plt.plot(test_k, mean_error, label='Regular emulator' if n==1 else None, color='r')
                plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3, color='r')
                mean_error_dl = np.mean(error_dl, axis=0)
                sigma1_error_dl = (np.quantile(error_dl, 0.8413)-np.quantile(error_dl, 0.1587))/2
                plt.plot(test_k_dl, mean_error_dl, label='Dimensionless emulator' if n==1 else None, color='k')
                plt.fill_between(test_k_dl, mean_error_dl+sigma1_error_dl, mean_error_dl-sigma1_error_dl, linewidth=0, alpha=0.3, color='k')
        plt.title('Comparison of regular to dimensionless emulator') #(Models ' + f"{test_start:03d}" + ' - ' + f"{test_end:03d}" + ')', wrap=True)
        plt.xlim(left=0.01, right=7)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_dimensionless_comparison.pdf', dpi=800)
        plt.clf()

elif test == 5:
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(1, 3)
        for m in [1000, 10000, 100000]:
                test_k, error, sens, test_start, test_end = emulator_dimensionless(bins, n, pc, samp, step, m)
                mean_error = np.mean(error, axis=0)
                median_error = np.median(error, axis=0)
                sigma1_error = (np.quantile(error, 0.8413)-np.quantile(error, 0.1587))/2
                plt.plot(test_k, mean_error, label=('Models with ' + f"{m:03d}" + ' MCMC steps'))
                plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        plt.title('MCMC comparison for models ' + f"{test_start:03d}" + ' - ' + f"{test_end}", wrap=True)
        plt.xlim(left=0.01, right=7)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_dimensionless_mcmctest.pdf', dpi=800)
        plt.clf()

elif test == 9:
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(1, 3)
        for b in [200, 100, 50, 25]:
                test_k, error, sens, test_start, test_end = emulator_dimensionless(b, n, pc, samp, step, mcmc)
                mean_error = np.mean(error, axis=0)
                median_error = np.median(error, axis=0)
                sigma1_error = (np.quantile(error, 0.8413)-np.quantile(error, 0.1587))/2
                plt.plot(test_k, mean_error, label=('Models with ' + f"{b:03d}" + ' bins used'))
                plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        plt.title('Bins comparison for models ' + f"{test_start:03d}" + ' - ' + f"{test_end}", wrap=True)
        plt.xlim(left=0.01, right=7)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_dimensionless_binstest.pdf', dpi=800)
        plt.clf()

elif test == 6:
        for n in range(1, 4):
                test_k, error, sens, test_start, test_end = emulator_dimensionless(bins, n, pc, samp, step, mcmc)

                sens_main_norm = sens['smePm']/np.sum(sens['smePm'])
                print(sens_main_norm)
                sens_main_list.append(sens_main_norm)
                sens_total_norm = sens['stePm']/(np.sum(sens['stePm']))
                sens_total_list.append(sens_total_norm)
        

        #print(sens_main_list)
        sens_main_list = np.array(sens_main_list)
        sens_main_list.reshape(3, -1)
        sens_main_cumulat = np.cumsum(sens_main_list, axis=1)
        #print(sens_main_cumulat)
        sens_main_cumulat[::-1].sort()
        #print('sens_main_norm = ' + str(sens_main_cumulat))
        sens_total_list = np.array(sens_total_list)
        sens_total_list.reshape(3, -1)
        sens_total_cumulat = np.cumsum(sens_total_list, axis=1)
        sens_total_list[::-1].sort()
        
        parameters_names = ['Omega_m', 'f_b', 'h0', 'ns', 'A_s', 'w0', 'wa', 'Omega_nuh^2', 'alpha_s']

        for j in reversed(range(9)):
	        plt.bar(np.arange(3), sens_main_cumulat[:, j]*100, label=parameters_names[j])
        plt.xticks(np.arange(3), labels=['0-49', '50-99', '100-149'], fontsize=6)
        plt.title('Main effect sensitivity')
        plt.xlabel('Test Models')
        plt.ylabel('Sensitivity contribution (%)')
        plt.legend(fontsize=5)
        plt.tight_layout()
        #plt.show()
        plt.savefig('./Plots/Power_spectra_z0_predicted_sensitivity_main.pdf', dpi=800)
        plt.clf()

        for j in reversed(range(9)):
	        plt.bar(np.arange(3), sens_total_cumulat[:, j]*100, label=parameters_names[j])
        plt.xticks(np.arange(3), labels=['0-49', '50-99', '100-149'], fontsize=6)
        plt.title('Total effect sensitivity')
        plt.xlabel('Test Models')
        plt.ylabel('Sensitivity contribution (%)')
        plt.legend(fontsize=5)
        plt.tight_layout()
        #plt.show()
        plt.savefig('./Plots/Power_spectra_z0_predicted_sensitivity_total.pdf', dpi=800)
        plt.clf()

elif test == 7:
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(1, 3)
        test_k, error, test_start, test_end = emulator_nonlinear(bins, n, pc, samp, step, mcmc)
        mean_error = np.mean(error, axis=0)
        sigma1_error = (np.quantile(error, 0.8413)-np.quantile(error, 0.1587))/2
        plt.plot(test_k, mean_error, label=('Boost function'))
        plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        test_k_nb, error_nb, test_start_nb, test_end_nb = emulator_nonlinear(bins, n, pc, samp, step, mcmc, boost=False)
        mean_error_nb = np.mean(error_nb, axis=0)
        sigma1_error_nb = (np.quantile(error_nb, 0.8413)-np.quantile(error_nb, 0.1587))/2
        plt.plot(test_k_nb, mean_error_nb, label=('Absolute P(k)'))
        plt.fill_between(test_k_nb, mean_error_nb+sigma1_error_nb, mean_error_nb-sigma1_error_nb, linewidth=0, alpha=0.3)
        plt.title('Comparison of emulating on absolute P(k) vs boost function (Models ' + f"{test_start:03d}" + ' - ' + f"{test_end:03d}" + ')', wrap=True)
        plt.xlim(left=0.01, right=7)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_absolute_comparison.pdf', dpi=800)
        plt.clf()

elif test == 8:
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(1, 3)
        test_k_10, error_10, test_start_10, test_end_10 = emulator_nonlinear(bins, n, pc, samp, step, mcmc, cosmo=np.arange(10))
        mean_error_10 = np.mean(error_10, axis=0)
        sigma1_error_10 = (np.quantile(error_10, 0.8413)-np.quantile(error_10, 0.1587))/2
        plt.plot(test_k_10, mean_error_10, label=('Full 10 parameter cosmology'))
        plt.fill_between(test_k_10, mean_error_10+sigma1_error_10, mean_error_10-sigma1_error_10, linewidth=0, alpha=0.3)
        test_k, error, test_start, test_end = emulator_nonlinear(bins, n, pc, samp, step, mcmc)
        mean_error = np.mean(error, axis=0)
        sigma1_error = (np.quantile(error, 0.8413)-np.quantile(error, 0.1587))/2
        plt.plot(test_k, mean_error, label=('9 parameters (f_gas/f_b dropped)'))
        plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        #test_k_, error_nb, test_start_nb, test_end_nb = emulator_nonlinear(bins, n, pc, samp, step, mcmc)
        #mean_error_nb = np.mean(error_nb, axis=0)
        #sigma1_error_nb = (np.quantile(error_nb, 0.8413)-np.quantile(error_nb, 0.1587))/2
        #plt.plot(test_k_nb, mean_error_nb, label=('Absolute P(k)'))
        #plt.fill_between(test_k_nb, mean_error_nb+sigma1_error_nb, mean_error_nb-sigma1_error_nb, linewidth=0, alpha=0.3)
        plt.title('Comparison of emulating on different numbers of cosmological parameters (Models ' + f"{test_start:03d}" + ' - ' + f"{test_end:03d}" + ')', wrap=True)
        plt.xlim(left=0.01, right=7)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_cosmo_comparison.pdf', dpi=800)
        plt.clf()
        
elif test == 10:
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(1, 3)
        for p in [8, 5, 3, 2]:
                test_k, error, sens, test_start, test_end = emulator_dimensionless(bins, n, p, samp, step, mcmc)
                mean_error = np.mean(error, axis=0)
                median_error = np.median(error, axis=0)
                sigma1_error = (np.quantile(error, 0.8413)-np.quantile(error, 0.1587))/2
                plt.plot(test_k, mean_error, label=('Models with ' + f"{p:03d}" + ' pcs used'))
                plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        plt.title('PCA comparison for models ' + f"{test_start:03d}" + ' - ' + f"{test_end}", wrap=True)
        plt.xlim(left=0.01, right=7)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_dimensionless_pcatest.pdf', dpi=800)
        plt.clf()

elif test == 11:
        plt.hlines(y=1.000, xmin=-1, xmax=7, color='k', linestyles='solid', alpha=0.5, label=None)
        plt.hlines(y=0.990, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        plt.hlines(y=1.010, xmin=-1, xmax=7, color='k', linestyles='dashed', alpha=0.5, label=None)
        plt.hlines(y=0.950, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        plt.hlines(y=1.050, xmin=-1, xmax=7, color='k', linestyles='dotted', alpha=0.5, label=None)
        n = random.randint(1, 3)
        test_k_camb, error_camb, sens, test_start_camb, test_end_camb = emulator_dimensionless(bins, n, pc, samp, step, mcmc, lin='camb')
        mean_error_camb = np.mean(error_camb, axis=0)
        sigma1_error_camb = (np.quantile(error_camb, 0.8413)-np.quantile(error_camb, 0.1587))/2
        plt.plot(test_k_camb, mean_error_camb, label=('CAMB linear theory'))
        plt.fill_between(test_k_camb, mean_error_camb+sigma1_error_camb, mean_error_camb-sigma1_error_camb, linewidth=0, alpha=0.3)
        test_k, error, sens, test_start, test_end = emulator_dimensionless(bins, n, pc, samp, step, mcmc)
        mean_error = np.mean(error, axis=0)
        sigma1_error = (np.quantile(error, 0.8413)-np.quantile(error, 0.1587))/2
        plt.plot(test_k, mean_error, label=('CLASS linear theory'))
        plt.fill_between(test_k, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=0.3)
        plt.title('Comparison of CAMB vs CLASS linear theories (Models ' + f"{test_start:03d}" + ' - ' + f"{test_end:03d}" + ')', wrap=True)
        plt.xlim(left=0.01, right=7)
        plt.xscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('Residual Average Error')
        plt.legend(fontsize=6)
        plt.savefig('./Plots/Power_spectra_z0_predicted_residual_nonlinear_linear_comparison.pdf', dpi=800)
        plt.clf()
