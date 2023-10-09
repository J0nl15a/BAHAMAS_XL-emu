import numpy as np
import math
import pylab as pb
from GPy import kern
from data_loader import BXL_DMO_Pk
from gpy_emulator import gpy_emulator
from gpy_emu_extension import gpy_HR_emulator, gpy_LR_emulator
import random

errors = []

#for _ in range(10):

class gpy_single_k_emu:
    
    def __init__(self, data, plot=False, adap_var=True):
        self.y = []
        self.error = []
        self.data = data

        print(data.k_test)
        for i, m in enumerate(data.k_test):
            print(i)
            print(m)
            if adap_var == True:
                if m <= .1:
                    if i == 0:
                        self.var=(100,100,100)
                    elif i == 1:
                        self.var=(100,1/(math.sqrt(data.modes['Low res'][i])),100)
                    elif i == 2:
                        self.var=(1/(math.sqrt(data.modes['Med res'][i])),1/(math.sqrt(data.modes['Low res'][i])),100)
                    else:
                        #var=(1,10,.01)
                        self.var = (1/(math.sqrt(data.modes['Med res'][i])), 1/(math.sqrt(data.modes['Low res'][i])), 1/(math.sqrt(data.modes['High res'][i])))
                elif i == 14:
                    self.var = (1,100,.01)
                else:
                    #self.var = (1/(math.sqrt(data.modes['Med res'][i])), 1/(math.sqrt(data.modes['Low res'][i])), 1/(math.sqrt(data.modes['High res'][i])))
                    self.var = (1,10,.01)
            else:
                self.var = (1,10,.01)
            print(self.var)
            gpy = gpy_emulator(data, kern.RBF(9, ARD=True), single_k = i, var_fix=True, var=self.var)
            print(gpy.y, gpy.error)
            self.y.append(float(gpy.y))
            self.error.append(float(gpy.error))
        #print(self.pred_Pk, self.error)

        return

    def plot(self):
        if self.data.test_models in range(50):
            pb.plot(self.data.k_test[:3], self.data.Y_test[:3], color='b', linestyle='dashed', label='Extended data')
            pb.plot(self.data.k_test[:3], self.y[:3], color='r', linestyle='dashed')
            pb.plot(self.data.k_test[2:], self.data.Y_test[2:], color='b', label=('True Non-linear P(k) for model_' + f"{self.data.test_models:03d}"))
            pb.plot(self.data.k_test[2:], self.y[2:], color='r', label=('Predicted Non-linear P(k) for model_' + f"{self.data.test_models:03d}"))
            
            pb.title('GPy test (kernel = RBF)')
            pb.xlabel('k (1/Mpc)')
            pb.ylabel('P(k) (Mpc^3)')
            pb.xscale('log')
            pb.yscale('log')
            pb.legend()
            pb.savefig('./Plots/gpy_single_k_med.pdf', dpi=800)
            pb.clf()
        
            pb.hlines(y=1.000, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='solid', alpha=0.5, label=None)
            pb.hlines(y=0.990, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label='1% error')
            pb.hlines(y=1.010, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label=None)
            pb.hlines(y=0.950, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label='5% error')
            pb.hlines(y=1.050, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label=None)
            pb.plot(self.data.k_test[:2], self.error[:2], alpha=0)
            pb.plot(self.data.k_test[2:], self.error[2:], label=('Residual error for model_' + f"{self.data.test_models:03d}"))
            pb.title('GPy error test (kernel = RBF)')
            pb.xlabel('k (1/Mpc)')
            pb.ylabel('Residual error')
            pb.xscale('log')
            pb.legend()
            pb.xlim(right=10)
            pb.savefig('./Plots/gpy_single_k_error_med.pdf', dpi=800)
            pb.clf()
        
        elif self.data.test_models in range(50, 100):
            pb.plot(self.data.k_test[:2], self.data.Y_test[:2], color='b', linestyle='dashed', label='Padded data')
            pb.plot(self.data.k_test[:2], self.y[:2], color='r', linestyle='dashed')
            pb.plot(self.data.k_test[-2:], self.data.Y_test[-2:], color='b', linestyle='dashed')
            pb.plot(self.data.k_test[-2:], self.y[-2:], color='r', linestyle='dashed')
            pb.plot(self.data.k_test[1:-1], self.data.Y_test[1:-1], color='b', label=('True Non-linear P(k) for model_' + f"{self.data.test_models:03d}"))
            pb.plot(self.data.k_test[1:-1], self.y[1:-1], color='r', label=('Predicted Non-linear P(k) for model_' + f"{self.data.test_models:03d}"))
            pb.title('GPy test (kernel = RBF)')
            pb.xlabel('k (1/Mpc)')
            pb.ylabel('P(k) (Mpc^3)')
            pb.xscale('log')
            pb.yscale('log')
            pb.legend()
            pb.savefig('./Plots/gpy_single_k_low.pdf', dpi=800)
            pb.clf()
        
            pb.hlines(y=1.000, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='solid', alpha=0.5, label=None)
            pb.hlines(y=0.990, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label='1% error')
            pb.hlines(y=1.010, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label=None)
            pb.hlines(y=0.950, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label='5% error')
            pb.hlines(y=1.050, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label=None)
            pb.plot(self.data.k_test[0], self.error[0], alpha=0)
            pb.plot(self.data.k_test[1:-1], self.error[1:-1], label=('Residual error for model_' + f"{self.data.test_models:03d}"))
            pb.title('GPy error test (kernel = RBF)')
            pb.xlabel('k (1/Mpc)')
            pb.ylabel('Residual error')
            pb.xscale('log')
            pb.legend()
            pb.xlim(right=10)
            pb.savefig('./Plots/gpy_single_k_error_low.pdf', dpi=800)
            pb.clf()
                
        elif self.data.test_models in range(100,150):
            pb.plot(self.data.k_test[:4], self.data.Y_test[:4], color='b', linestyle='dashed', label='Padded data')
            pb.plot(self.data.k_test[:4], self.y[:4], color='r', linestyle='dashed')
            pb.plot(self.data.k_test[3:], self.data.Y_test[3:], color='b', label=('True Non-linear P(k) for model_' + f"{self.data.test_models:03d}"))
            pb.plot(self.data.k_test[3:], self.y[3:], color='r', label=('Predicted Non-linear P(k) for model_' + f"{self.data.test_models:03d}"))
            pb.title('GPy test (kernel = RBF)')
            pb.xlabel('k (1/Mpc)')
            pb.ylabel('P(k) (Mpc^3)')
            pb.xscale('log')
            pb.yscale('log')
            pb.legend()
            pb.savefig('./Plots/gpy_single_k_high.pdf', dpi=800)
            pb.clf()
                
            pb.hlines(y=1.000, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='solid', alpha=0.5, label=None)
            pb.hlines(y=0.990, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label='1% error')
            pb.hlines(y=1.010, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label=None)
            pb.hlines(y=0.950, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label='5% error')
            pb.hlines(y=1.050, xmin=-1, xmax=max(self.data.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label=None)
            pb.plot(self.data.k_test[:3], self.error[:3], alpha=0)
            pb.plot(self.data.k_test[3:], self.error[3:], label=('Residual error for model_' + f"{self.data.test_models:03d}"))
            pb.title('GPy error test (kernel = RBF)')
            pb.xlabel('k (1/Mpc)')
            pb.ylabel('Residual error')
            pb.xscale('log')
            pb.legend()
            pb.xlim(right=10)
            pb.savefig('./Plots/gpy_single_k_error_high.pdf', dpi=800)
            pb.clf()
            
        return
                                                                                                    
    """errors.append(error)
    errors = np.array(errors)
    errors = errors.reshape(10, -1)
    mean_error = np.mean(errors, axis=0)
    sigma1_error = (np.quantile(errors, 0.8413)-np.quantile(errors, 0.1587))/2


    pb.hlines(y=1.000, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='solid', alpha=0.5, label=None)
    pb.hlines(y=0.990, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label='1% error')
    pb.hlines(y=1.010, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label=None)
    pb.hlines(y=0.950, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label='5% error')
    pb.hlines(y=1.050, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label=None)
    pb.plot(boost.k_test, mean_error, label=('Mean residual error for 10 models'))
    pb.fill_between(boost.k_test, mean_error+sigma1_error, mean_error-sigma1_error, linewidth=0, alpha=.3)
    pb.title('Average GPy error test (kernel = RBF)')
    pb.xlabel('k (1/Mpc)')
    pb.ylabel('Mean residual error')
    pb.xscale('log')
    pb.legend()
    pb.xlim(right=10)
    pb.savefig('./Plots/gpy_single_k_mean_error.pdf', dpi=800)
    pb.clf()"""

def ext_comparison(p, plot=True):
    
    n = random.randint(p[0], p[1])
    print(n)
    if n in range(0, 50):
        k=2
    elif n in range(50, 100):
        k=1
    elif n in range(100, 150):
        k=3
    boost = BXL_DMO_Pk(n, 100, pk='nbk-rebin-std', lin='rebin')
    boost_pad = BXL_DMO_Pk(n, 100, pk='nbk-rebin-std', lin='rebin', pad=True)
    b0 = gpy_HR_emulator(boost, ['Low'], [len(boost.k_test)-1], test=((len(boost.k_test)-1) if n in range(50,100) else False)).data
    #b2 = gpy_LR_emulator(b1, 'Low', 1, PT=True).data
    #b3 = gpy_LR_emulator(b2, 'Med', 2).data
    boost_ext = gpy_LR_emulator(b0, ['Med', 'Low', 'High'], [2, 1, 3], test=k).data
        
    pad = gpy_single_k_emu(n, boost_pad)
    emu = gpy_single_k_emu(n, boost_ext)

    emu.plot()
    

    if plot == True:
    
        fig, (ax1,ax2) = pb.subplots(1, 2, sharey=True, tight_layout=True, gridspec_kw={'wspace': 0})
        if n in range(50, 100):
            ax1.plot(boost.k_test[-2:], boost_pad.Y_test[-2:], color='b', linestyle='dashed')
            ax1.plot(boost.k_test[-2:], pad.y[-2:], color='r', linestyle='dashed')
            ax2.plot(boost.k_test[-2:], boost_ext.Y_test[-2:], color='b', linestyle='dashed')
            ax2.plot(boost.k_test[-2:], emu.y[-2:], color='r', linestyle='dashed')
        ax1.plot(boost.k_test[:k+1], boost_pad.Y_test[:k+1], color='b', linestyle='dashed', label='Extended data')
        ax1.plot(boost.k_test[:k+1], pad.y[:k+1], color='r', linestyle='dashed')
        ax2.plot(boost.k_test[:k+1], boost_ext.Y_test[:k+1], color='b', linestyle='dashed')
        ax2.plot(boost.k_test[:k+1], emu.y[:k+1], color='r', linestyle='dashed')
        ax1.plot(boost.k_test[k:] if n not in range(50,100) else boost.k_test[k:-1], boost_pad.Y_test[k:] if n not in range(50,100) else boost_pad.Y_test[k:-1], color='b', label=('True Non-linear P(k) for model_' + f"{n:03d}"))
        ax1.plot(boost.k_test[k:] if n not in range(50,100) else boost.k_test[k:-1], pad.y[k:] if n not in range(50,100) else pad.y[k:-1], color='r', label=('Predicted Non-linear P(k) for model_' + f"{n:03d}"))
        ax2.plot(boost.k_test[k:] if n not in range(50,100) else boost.k_test[k:-1], boost_ext.Y_test[k:] if n not in range(50,100) else boost_ext.Y_test[k:-1], color='b')
        ax2.plot(boost.k_test[k:] if n not in range(50,100) else boost.k_test[k:-1], emu.y[k:] if n not in range(50,100) else emu.y[k:-1], color='r')
        ax1.set_title('Pade approximant padding', wrap=True)
        ax2.set_title('HR/LR emulator extension', wrap=True)
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        fig.supylabel('P(k) (Mpc^3)', fontsize = 10)
        ax1.legend(fontsize=6)
        fig.supxlabel('k [1/Mpc]', fontsize = 10)
        fig.suptitle('P(k) extension method comparison on GPy emulator', wrap=True)
        pb.savefig('./Plots/gpy_pad_v_emu_extension.pdf', dpi=800)
        pb.clf()
    
        #fig2, (ax3,ax4) = pb.subplots(1, 2, sharey=True, tight_layout=True, gridspec_kw={'wspace': 0})
        pb.hlines(y=1.000, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='solid', alpha=0.5, label=None)
        pb.hlines(y=0.990, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        pb.hlines(y=1.010, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label=None)
        pb.hlines(y=0.950, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        pb.hlines(y=1.050, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label=None)
        #ax4.hlines(y=1.000, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='solid', alpha=0.5)
        #ax4.hlines(y=0.990, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5)
        #ax4.hlines(y=1.010, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5)
        #ax4.hlines(y=0.950, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5)
        #ax4.hlines(y=1.050, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5)
        if n in range(50, 100):
            pb.plot(boost.k_test[-2:], pad.error[-2:], linestyle='dashed', color='tab:blue')
            pb.plot(boost.k_test[-2:], emu.error[-2:], linestyle='dashed', color='tab:orange')
        pb.plot(boost.k_test[:k+1], pad.error[:k+1], label=('Error on extended data'), linestyle='dashed', color='tab:blue')
        pb.plot(boost.k_test[:k+1], emu.error[:k+1], linestyle='dashed', color='tab:orange')
        pb.plot(boost.k_test[k:] if n not in range(50,100) else boost.k_test[k:-1], pad.error[k:] if n not in range(50,100) else pad.error[k:-1], label=('Pade approximant padding'), color='tab:blue')
        pb.plot(boost.k_test[k:] if n not in range(50,100) else boost.k_test[k:-1], emu.error[k:] if n not in range(50,100) else emu.error[k:-1], label=('HR/LR emulator extension'), color='tab:orange')
        #ax3.set_title('Pade approximant padding', wrap=True)
        #ax4.set_title('HR/LR emulator extension', wrap=True)
        pb.xscale('log')
        #ax4.set_xscale('log')
        pb.xlim(0, 10)
        #ax4.set_xlim(0, 10)
        if n in range(50, 100):
            pb.ylim(0.9, 1.15)
            #ax4.set_ylim(0.9, 1.15)
        #fig2.supylabel('Residual error', fontsize = 10)
        pb.ylabel('Residual error', fontsize = 10)
        pb.legend(fontsize=6)
        #fig2.supxlabel('k [1/Mpc]', fontsize = 10)
        pb.xlabel('k [1/Mpc]', fontsize = 10)
        pb.title('Error of P(k) extension method comparison on GPy emulator', wrap=True)
        #fig2.suptitle('Error of P(k) extension method comparison on GPy emulator', wrap=True)
        pb.savefig('./Plots/gpy_pad_v_emu_extension_error.pdf', dpi=800)
        pb.clf()

    return pad, emu, n

if __name__ == '__main__':

    models =[]
    boost = BXL_DMO_Pk(0, 100, pk='nbk-rebin-std', lin='rebin')
    res = 'All'
    if res == 'Med':
        k=2
        q=(0,49)
    elif res == 'Low':
        k=1
        q=(50,99)
    elif res == 'High':
        k=3
        q=(100,149)
    elif res == 'All':
        k=3
        q=(0,149)

    runs = int((q[1]-q[0]+1)*.2)
    pad_err = np.zeros((runs,15))
    emu_err = np.zeros((runs,15))
    for r in range(runs):
        pad, emu, n = ext_comparison(q, plot=False)
        pad_err[r, :] = pad.error
        emu_err[r, :] = emu.error
        models.append(n)
    print(models)
    pad_avg = np.mean(pad_err, axis=0)
    emu_avg = np.mean(emu_err, axis=0)
    pad_sig1 = (np.quantile(pad_err, 0.8413)-np.quantile(pad_err, 0.1587))/2
    emu_sig1 = (np.quantile(emu_err, 0.8413)-np.quantile(emu_err, 0.1587))/2

    print(emu_avg)
    print(emu_sig1)

    pb.hlines(y=1.000, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='solid', alpha=0.5, label=None)
    pb.hlines(y=0.990, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label='1% error')
    pb.hlines(y=1.010, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dashed', alpha=0.5, label=None)
    pb.hlines(y=0.950, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label='5% error')
    pb.hlines(y=1.050, xmin=-1, xmax=max(boost.k_test)+2, color='k', linestyles='dotted', alpha=0.5, label=None)
    if res=='Low' or res=='All':
        pb.plot(boost.k_test[-2:], pad_avg[-2:], linestyle='dashed', color='tab:blue')
        pb.plot(boost.k_test[-2:], emu_avg[-2:], linestyle='dashed', color='tab:orange')
    pb.plot(boost.k_test[:k+1], pad_avg[:k+1], label=('Error on extended data'), linestyle='dashed', color='tab:blue')
    pb.plot(boost.k_test[:k+1], emu_avg[:k+1], linestyle='dashed', color='tab:orange')
    pb.plot(boost.k_test[k:-1] if res=='Low' or res=='All' else boost.k_test[k:], pad_avg[k:-1] if res=='Low' or res=='All' else pad_avg[k:], label=('Pade approximant padding'), color='tab:blue')
    pb.plot(boost.k_test[k:-1] if res=='Low' or res=='All' else boost.k_test[k:], emu_avg[k:-1] if res=='Low' or res=='All' else emu_avg[k:], label=('HR/LR emulator extension'), color='tab:orange')
    pb.fill_between(boost.k_test[k:-1] if res=='Low' or res=='All' else boost.k_test[k:], (pad_avg[k:-1]+pad_sig1) if res=='Low' or res=='All' else (pad_avg[k:]+pad_sig1), (pad_avg[k:-1]-pad_sig1) if res=='Low' or res=='All' else (pad_avg[k:]-pad_sig1), linewidth=0, alpha=0.3, color='tab:blue')
    pb.fill_between(boost.k_test[k:-1] if res=='Low' or res=='All' else boost.k_test[k:], (emu_avg[k:-1]+emu_sig1) if res=='Low' or res=='All' else (emu_avg[k:]+emu_sig1), (emu_avg[k:-1]-emu_sig1) if res=='Low' or res=='All' else (emu_avg[k:]-emu_sig1), linewidth=0, alpha=0.3, color='tab:orange', label='1 S.D.')
    pb.xscale('log')
    pb.xlim(0, 10)
    if res=='Low' or res=='All':
        pb.ylim(min(pad_avg[k:-1]-pad_sig1)-.01, max(pad_avg[k:-1]+pad_sig1))
    pb.ylabel('Mean Residual error', fontsize = 10)
    pb.legend(fontsize=6)
    pb.xlabel('k [1/Mpc]', fontsize = 10)
    pb.title('Mean Error of P(k) extension method comparison on GPy emulator (using ' + f"{runs}" + ' models from range ' + f"{q[0]}" + '-' + f"{q[1]}" + ')', wrap=True)
    pb.savefig('./Plots/gpy_pad_v_emu_extension_error_mean.pdf', dpi=800)
    pb.clf()
