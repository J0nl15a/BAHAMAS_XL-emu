import numpy as np
import math
import pylab as pb
import matplotlib.figure as fi
import GPy
import random
import pickle


class gpyImprovedEmulator:
    
    def __init__(self, data, variance_method, error=True, kernel=GPy.kern.RBF(10), fix_variance=False, variance=(1e-8,1e-7,1e-9), save=False, model_upload=False, ARD=False, flamingo=False):

        self.data = data
        self.kernel = kernel
        self.flamingo = flamingo

        print(self.data.Y_test)
        print('done')

        
        if ARD == True:
            try:
                self.kernel = GPy.kern.RBF(self.data.X_test.shape[1], ARD=True)
            except IndexError:
                self.kernel = GPy.kern.RBF(self.data.X_test.shape[0], ARD=True)

        if not isinstance(self.data.holdout, bool):
            self.test_model = self.data.holdout
        else:                
            self.test_model = 'External cosmology'
            
        try:
            self.pred = np.zeros((self.data.X_test.shape[0], self.data.bins))
            self.error = np.zeros((self.data.X_test.shape[0], self.data.bins))
        except IndexError:
            self.pred = np.zeros(self.data.bins)
            self.error = np.zeros(self.data.bins)
        
        '''if flamingo==True:
            self.pred = np.zeros(self.data.Y_test.shape[1])
            self.error = np.zeros((self.data.Y_test.shape[0], self.data.Y_test.shape[1]))'''

        self.weights = np.zeros((149 if not isinstance(self.data.holdout, bool) else 150, len(self.data.k_test)))
        print(data.k_test)
        #for i,k in enumerate(self.data.k_test):
        for i in range(1):
            print(i)
            #print(k)
            
            if model_upload == True:
                self.model = pickle.load(open(f"./gpy_model/gpy_model_{i}.pkl", "rb"))
            else:
                if variance_method == 'modes':
                    if k <= .1:
                        if i == 0:
                            self.var=(100,100,100)
                        elif i == 1:
                            self.var=(100,1/(math.sqrt(self.data.modes['Low res'][i])),100)
                        elif i == 2:
                            self.var=(1/(math.sqrt(self.data.modes['Med res'][i])),1/(math.sqrt(self.data.modes['Low res'][i])),100)
                        else:
                            self.var = (1/(math.sqrt(self.data.modes['Med res'][i])),1/(math.sqrt(self.data.modes['Low res'][i])),1/(math.sqrt(self.data.modes['High res'][i])))
                    elif i == 14:
                        self.var = (1,100,.01)
                    else:
                        self.var = (1,10,.01)
                    
                elif variance_method == 'variance_weights':
                    self.var = (self.data.IR_mass_res_weights[i], self.data.LR_mass_res_weights[i], self.data.HR_mass_res_weights[i])
                else:
                    self.var = variance
                
                print(np.reshape(self.data.Y_train[:3, i], (3,1)))
                print(self.data.X_train.shape, self.data.Y_train.shape)
                #self.model = GPy.models.GPHeteroscedasticRegression(self.data.X_train, np.reshape(self.data.Y_train[:, i], (len(self.data.Y_train),1)), self.kernel)
                self.model = GPy.models.GPHeteroscedasticRegression(self.data.X_train, self.data.Y_train.reshape((len(self.data.Y_train),self.data.bins)), self.kernel)
                #self.model = GPy.models.GPHeteroscedasticRegression(self.data.X_train[100:, :], np.reshape(self.data.Y_train[100:, i], (50,1)), self.kernel)
                
                if fix_variance == True:
                    index1 = 50
                    index2 = 100
                    if self.test_model in range(50):
                        index1 -= 1
                        index2 -= 1
                    elif self.test_model in range(50, 100):
                        index2 -= 1

                    print(self.var[0], self.var[1], self.var[2])
                    print(self.model.het_Gauss.variance[0])
                    print(self.model.het_Gauss.variance[50])
                    print(self.model.het_Gauss.variance[100])

                    #self.model.het_Gauss.variance[:index1] *= self.var[0]-min(self.var)
                    #self.model.het_Gauss.variance[index1:index2] *= self.var[1]-min(self.var)
                    #self.model.het_Gauss.variance[index2:] *= self.var[2]-min(self.var)

                    #if k<0.1:
                    self.model.het_Gauss.variance[:index1] = 1e-8+((self.var[0]-min(self.var))*self.data.Y_train[:index1, i]).reshape(index1,1)**2
                    self.model.het_Gauss.variance[index1:index2] = 1e-8+((self.var[1]-min(self.var))*self.data.Y_train[index1:index2, i]).reshape(index2-index1,1)**2
                    if not isinstance(self.data.holdout, bool):
                        self.model.het_Gauss.variance[index2:] = 1e-8+((self.var[2]-min(self.var))*self.data.Y_train[index2:, i]).reshape(150-(index2+1),1)**2
                    else:
                        self.model.het_Gauss.variance[index2:] = 1e-8+((self.var[2]-min(self.var))*self.data.Y_train[index2:, i]).reshape(150-(index2),1)**2
                        
                    ####### Method for weighting on whole function
                    '''self.model.het_Gauss.variance[:index1] = 1e-8+((self.var[0]-1)*self.data.Y_train[:index1, :]).reshape(index1,-1)**2
                    self.model.het_Gauss.variance[index1:index2] = 1e-8+((self.var[1]-1)*self.data.Y_train[index1:index2, :]).reshape(index2-index1,-1)**2
                    if not isinstance(self.data.holdout, bool):
                        self.model.het_Gauss.variance[index2:] = 1e-8+((self.var[2]-1)*self.data.Y_train[index2:, :]).reshape(150-(index2+1),-1)**2
                    else:
                        self.model.het_Gauss.variance[index2:] = 1e-8+((self.var[2]-1)*self.data.Y_train[index2:, :]).reshape(150-(index2),-1)**2'''

                    '''self.model.het_Gauss.variance[:index1] = 1e-8+((self.var[0])*self.data.Y_train[:index1, i]).reshape(index1,1)**2
                    if not isinstance(self.data.holdout, bool):
                        self.model.het_Gauss.variance[index2:] = 1e-8+((self.var[2])*self.data.Y_train[index2:, i]).reshape(150-(index2+1),1)**2
                    else:
                        self.model.het_Gauss.variance[index2:] = 1e-8+((self.var[2])*self.data.Y_train[index2:, i]).reshape(150-(index2),1)**2
                    self.model.het_Gauss.variance[index1:index2] = 1e-8+((self.var[1])*self.data.Y_train[index1:index2, i]).reshape(index2-index1,1)**2'''
                        
                    #self.model.het_Gauss.variance.fix()
                    
                    print(self.model.het_Gauss.variance[0])
                    print(self.model.het_Gauss.variance[50])
                    print(self.model.het_Gauss.variance[100])
                    #self.model.het_Gauss.variance[:index1] *= self.var[2]

                self.model.optimize()
                '''self.weights[:index1, i] = self.model.het_Gauss.variance[:index1].flatten()
                self.weights[index1:index2, i] = self.model.het_Gauss.variance[index1:index2].flatten()
                self.weights[index2:, i] = self.model.het_Gauss.variance[index2:].flatten()'''
                
                print(self.model.het_Gauss.variance[0])
                print(self.model.het_Gauss.variance[50])
                print(self.model.het_Gauss.variance[100])
                print(self.model.het_Gauss.variance)

                
            if self.data.X_test.ndim > 1:
                model_output = self.model._raw_predict(self.data.X_test.reshape(len(self.data.X_test), -1))
            else:
                model_output = self.model._raw_predict(self.data.X_test.reshape(1, -1))
            if self.data.log == True:
                #self.y = 10**(self.data.normalise(undo = model_output[0], k=i))
                self.y = 10**(self.data.normalise(undo = model_output[0]))
            else:
                self.y = float(model_output[0])
            self.post_variance = model_output[1]

            
            if save == True:
                pickle.dump(self.model, open(f"./gpy_model/gpy_model_{i}.pkl", "wb"))

            if self.data.X_test.ndim > 1:
                #self.pred[:, i] = self.y.reshape(-1)
                self.pred = self.y
                if error == True:
                    #self.error[:, i] = self.y.reshape(-1)/self.data.Y_test[:, i]
                    self.error = self.y/self.data.Y_test
            else:
                #self.pred[i] = self.y
                self.pred = self.y
                if error == True:
                    if isinstance(self.data.holdout, bool):
                        if flamingo == True:
                            #self.error[:, i] = self.y/self.data.Y_test[:, i]
                            self.error = self.y/self.data.Y_test
                    else:
                        print(self.data.Y_test)
                        #self.error[i] = float(self.y/self.data.Y_test[i])
                        self.error = self.y/self.data.Y_test

            print(self.y, self.error)

        print(self.pred)
        print(self.error)
        
        return

    
    def plot(self, name, plot_file='png'):
        w,h=fi.figaspect(.3)
        fig, (ax1, ax2) = pb.subplots(1, 2, figsize=(w,h), dpi=1200)

        ax2.hlines(y=1.000, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='solid', alpha=0.5, label=None)
        ax2.hlines(y=0.990, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        ax2.hlines(y=1.010, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='dashed', alpha=0.5, label=None)
        ax2.hlines(y=0.950, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        ax2.hlines(y=1.050, xmin=-1, xmax=int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10, color='k', linestyles='dotted', alpha=0.5, label=None)
        
        if not isinstance(self.data.holdout, bool):
            unmasked_indicies = np.arange(self.data.k_test.shape[0])[~self.data.nan_mask[self.data.holdout,:]]
        else:
            unmasked_indicies = np.arange(self.data.k_test.shape[0])

        if self.flamingo == False:
            '''if unmasked_indicies.shape != self.data.k_test.shape:
                ax1.plot(self.data.k_test, self.data.Y_test, color='b', linestyle='dashed', label='Padded data')
                ax1.plot(self.data.k_test, self.pred, color='r', linestyle='dashed')
                ax2.plot(self.data.k_test, self.error, color='tab:blue', linestyle='dashed', label='Padded data')
            
            ax1.plot(self.data.k_test[unmasked_indicies], self.data.Y_test[unmasked_indicies], color='b', label=('True P(k) for model_' + f"{self.test_model:03d}") if not isinstance(self.data.holdout, bool) else ('True P(k) for external test model'))
            ax1.plot(self.data.k_test[unmasked_indicies], self.pred[unmasked_indicies], color='r', label=('Predicted P(k) for model_' + f"{self.test_model:03d}") if not isinstance(self.data.holdout, bool) else ('Predicted P(k) for external test model'))

            ax2.plot(self.data.k_test[unmasked_indicies], self.error[unmasked_indicies], color='tab:blue', label=('model_' + f"{self.test_model:03d}" + ' error') if not isinstance(self.data.holdout, bool) else ('External test model error'))'''

            ax1.plot(self.data.k_test, self.data.Y_test, color='b', label=('True P(k) for model_' + f"{self.test_model:03d}") if not isinstance(self.data.holdout, bool) else ('True P(k) for external test model'))
            ax1.plot(self.data.k_test, self.pred.reshape(-1,1), color='r', label=('Predicted P(k) for model_' + f"{self.test_model:03d}") if not isinstance(self.data.holdout, bool) else ('Predicted P(k) for external test model'))
            ax2.plot(self.data.k_test, self.error.reshape(-1,1), color='tab:blue', label=('model_' + f"{self.test_model:03d}" + ' error') if not isinstance(self.data.holdout, bool) else ('External test model error'))

            fig.suptitle(f'GPy test (kernel = {name})')
            ax1.legend(loc='upper left')

            ax2.legend(loc='lower left')
            '''if max(self.error) > 1.2 and min(self.error) < 0.8:
                ax2.set_ylim(top=1.2, bottom=0.8)
            elif min(self.error) < 0.8:
                ax2.set_ylim(bottom=0.8)
            elif max(self.error) > 1.2:
                ax2.set_ylim(top=1.2)'''

        else:
            #model_lines = ['tab:pink', '#637939', 'tab:green', '#b5cf6b', '#a1d99b', 'tab:purple', 'tab:blue', '#e7ba52', 'tab:cyan', 'tab:olive', 'tab:orange', 'tab:red', 'tab:brown', 'tab:grey']
            model_lines = ['tab:pink', '#637939', 'tab:green', '#b5cf6b', 'tab:purple', 'tab:blue', '#e7ba52', 'tab:cyan', 'tab:olive', 'tab:orange', 'tab:red', 'tab:brown']
            for m in range(len(emulator.data.Y_test)):
                '''unmasked_indicies = np.arange(emulator.data.k_test.shape[0])[~emulator.data.flamingo_mask[m, :]]
                if unmasked_indicies.shape != emulator.data.k_test.shape:
                    ax1.plot(emulator.data.k_test, emulator.data.Y_test[m,:], color=model_lines[m], linestyle='dashed', label='Padded data' if m==0 else None)
                    ax2.plot(emulator.data.k_test, emulator.error[m,:], color=model_lines[m], linestyle='dashed', label='Padded data' if m==0 else None)'''
                    
                #ax1.plot(emulator.data.k_test[unmasked_indicies], emulator.data.Y_test[m, unmasked_indicies], color=model_lines[m], label=(f'L{boost.flamingo_sims[m][0]}N{boost.flamingo_sims[m][1]})'))
                ax1.plot(emulator.data.k_test[unmasked_indicies], emulator.data.Y_test[m, unmasked_indicies], color=model_lines[m], linestyle='dashed', label=f'Original data' if m==0 else None)
                ax2.plot(emulator.data.k_test[unmasked_indicies], emulator.error[m, unmasked_indicies], color=model_lines[m], label=(f'L{boost.flamingo_sims[m][0]}N{boost.flamingo_sims[m][1]}' if m<4 else f'L{boost.flamingo_sims[m+1][0]}N{boost.flamingo_sims[m+1][1]}'))
                ax1.plot(emulator.data.k_test, emulator.pred[m, :], color=model_lines[m], label=(f'L{boost.flamingo_sims[m][0]}N{boost.flamingo_sims[m][1]}' if m<4 else f'L{boost.flamingo_sims[m+1][0]}N{boost.flamingo_sims[m+1][1]}'))
      
            #ax1.plot(emulator.data.k_test, emulator.pred, color='k', label=('Predicted P(k) for FLAMINGO cosmology'))
                    
            fig.suptitle(f'GPy test for FLAMINGO (kernel = {name})')
            ax1.legend(loc='upper left', fontsize=6, title='Predicted P(k) for FLAMINGO cosmology', ncol=2)
                    
            ax2.legend(loc='lower left', fontsize=6, title='Predicted/Original', ncol=2)
            ax2.set_ylim(top=1.2, bottom=0.8)


        ax1.set_xlabel(r'$k \: [1/Mpc]$', fontsize=15)
        ax1.set_ylabel(r'$P(k) \: [Mpc^3]$', fontsize=15)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        ax2.set_xlabel(r'$k \: [1/Mpc]$', fontsize=15)
        ax2.set_ylabel('Residual error', fontsize=15)
        ax2.set_xscale('log')
        ax2.set_xlim(right=(int(max(self.data.k_test)+1) if max(self.data.k_test)>9.99 else 10))
        fig.subplots_adjust(wspace=0.15)
        if self.flamingo == True:
            pb.savefig(f'./Plots/gpy_improved_test_flamingo.{plot_file}', dpi=1200)
        else:
            pb.savefig(f'./Plots/gpy_improved_test.{plot_file}', dpi=1200)
        pb.clf()
        
        return
                                                                                                    
    
if __name__ == '__main__':
    from flamingo_data_loader import flamingoDMOData
    
    model=104#np.random.randint(100,150)
    flamingo=False
    boost = flamingoDMOData(pk='powmes', lin='class', flamingo_lin='class', log=True, sigma8=True, cutoff=(.03, 20), holdout=model)
    #boost.weights()
    print(boost.X_train)
    print(boost.Y_train)
    print(boost.X_test)
    print(boost.Y_test)
    print(boost.X_test.shape)
    print(boost.Y_test.shape)
    emulator = gpyImprovedEmulator(boost, 'variance_weights_no', fix_variance=False, ARD=True, flamingo=flamingo)
    print(boost.X_train)
    print(boost.Y_train)
    print(boost.X_test)
    print(boost.Y_test)
    #emulator.pred = 10**((emulator.pred*std_Pk)+mean_Pk)
    #emulator.error = emulator.pred/emulator.data.Y_test
    #emulator.plot('ARD')

    print(emulator.weights)
    #for i in range(50):
    #    pb.plot(boost.k_test, emulator.weights[i+100, :], color='tab:green', label='High Res' if i==0 else None)
    #    pb.plot(boost.k_test, 2*emulator.weights[i, :], color='tab:blue', label='Intermediate Res' if i==0 else None)
    #    pb.plot(boost.k_test, 4*emulator.weights[i+50, :], color='tab:orange', label='Low Res' if i==0 else None)
    #pb.title('Weights calculated internally for the emulator, for mass resolution effects', wrap=True)
    #pb.xlabel('k (1/Mpc)')
    #pb.ylabel('Variance')
    #pb.xscale('log')
    #pb.xlim(right=20)
    #pb.legend(loc='upper left', fontsize=10)
    #pb.savefig(f'./Plots/emulator_computed_weights.png', dpi=1200)
    #pb.clf()
    print(boost.Y_train)
    #quit()

    test_error=[]
    test_std = []
    tests = [('HR', 100, 150), ('LR', 50, 100), ('IR', 0, 50)]
    for k in range(len(tests)):
        error = []
        for m in range(tests[k][1], tests[k][2]):
            model_boost = flamingoDMOData(pk='powmes', lin='camb', flamingo_lin='camb', log=True, holdout=m, sigma8=True, cutoff=(.03, 20))
            model_boost.weights()
            model_emulator = gpyImprovedEmulator(model_boost, 'variance_weights_no', fix_variance=False, ARD=True, flamingo=flamingo)
            print(model_emulator.error)
            error.append(model_emulator.error)

        error = np.array(error)
        avg_error = np.mean(error, axis=0)
        std_error = np.std(error, axis=0)
        print(avg_error)
        test_error.append(avg_error)
        test_std.append(std_error)
        
    pb.hlines(y=1.000, xmin=-1, xmax=int(max(emulator.data.k_test)+1) if max(emulator.data.k_test)>9.99 else 10, color='k', linestyles='solid', alpha=0.5, label=None)
    pb.hlines(y=0.990, xmin=-1, xmax=int(max(emulator.data.k_test)+1) if max(emulator.data.k_test)>9.99 else 10, color='k', linestyles='dashed', alpha=0.5, label='1% error')
    pb.hlines(y=1.010, xmin=-1, xmax=int(max(emulator.data.k_test)+1) if max(emulator.data.k_test)>9.99 else 10, color='k', linestyles='dashed', alpha=0.5, label=None)
    pb.hlines(y=0.950, xmin=-1, xmax=int(max(emulator.data.k_test)+1) if max(emulator.data.k_test)>9.99 else 10, color='k', linestyles='dotted', alpha=0.5, label='5% error')
    pb.hlines(y=1.050, xmin=-1, xmax=int(max(emulator.data.k_test)+1) if max(emulator.data.k_test)>9.99 else 10, color='k', linestyles='dotted', alpha=0.5, label=None)

    '''pb.plot(emulator.data.k_test, test_error[0].reshape(-1,1), color='tab:green', linestyle='dashed', label='Padded data')
    #pb.plot(emulator.data.k_test[1:], test_error[0][1:], color='tab:green', label=f'Average {tests[0][0]} error')
    pb.plot(emulator.data.k_test[~model_boost.nan_mask[100,:]], test_error[0][~model_boost.nan_mask[100,:]].reshape(-1,1), color='tab:green', label=f'Average {tests[0][0]} error')
    pb.plot(emulator.data.k_test, test_error[1].reshape(-1,1), color='tab:orange', linestyle='dashed')
    #pb.plot(emulator.data.k_test[1:], test_error[1][1:], color='tab:orange', label=f'Average {tests[1][0]} error')
    pb.plot(emulator.data.k_test[~model_boost.nan_mask[50,:]], test_error[1][~model_boost.nan_mask[50,:]].reshape(-1,1), color='tab:orange', label=f'Average {tests[1][0]} error')
    pb.plot(emulator.data.k_test, test_error[2].reshape(-1,1), color='tab:blue', linestyle='dashed')
    #pb.plot(emulator.data.k_test[1:], test_error[2][1:], color='tab:blue', label=f'Average {tests[2][0]} error')
    pb.plot(emulator.data.k_test[~model_boost.nan_mask[0,:]], test_error[2][~model_boost.nan_mask[0,:]].reshape(-1,1), color='tab:blue', label=f'Average {tests[2][0]} error')
    if max(test_error[0]) > 1.2 and min(test_error[0]) < 0.8:
        pb.ylim(top=1.2, bottom=0.8)
    elif min(test_error[0]) < 0.8:
        pb.ylim(bottom=0.8)
    elif max(test_error[0]) > 1.2:
        pb.ylim(top=1.2)'''

    pb.plot(emulator.data.k_test, test_error[2].reshape(-1,1), color='tab:blue', label=f'Average {tests[2][0]} error')
    pb.plot(emulator.data.k_test, test_error[0].reshape(-1,1), color='tab:green', label=f'Average {tests[0][0]} error')
    pb.plot(emulator.data.k_test, test_error[1].reshape(-1,1), color='tab:orange', label=f'Average {tests[1][0]} error')

    pb.fill_between(emulator.data.k_test, (test_error[2].reshape(-1,1)+test_std[2].reshape(-1,1)).flatten(), (test_error[2].reshape(-1,1)-test_std[2].reshape(-1,1)).flatten(), color='tab:blue', linewidth=0, alpha=.5)#, label=f'1 $\sigma$')
    pb.fill_between(emulator.data.k_test, (test_error[0].reshape(-1,1)+test_std[0].reshape(-1,1)).flatten(), (test_error[0].reshape(-1,1)-test_std[0].reshape(-1,1)).flatten(), color='tab:green', linewidth=0, alpha=.5, label=f'1 $\sigma$')
    pb.fill_between(emulator.data.k_test, (test_error[1].reshape(-1,1)+test_std[1].reshape(-1,1)).flatten(), (test_error[1].reshape(-1,1)-test_std[1].reshape(-1,1)).flatten(), color='tab:orange', linewidth=0, alpha=.5)#, label=f'1 $\sigma$')
    
    pb.title(f'Average error on all model resolutions')
    pb.xlabel(r'$k \: [1/Mpc]$', fontsize=10)
    pb.ylabel('Residual error', fontsize=10)
    pb.xscale('log')
    pb.xlim(right=(int(max(emulator.data.k_test)+1) if max(emulator.data.k_test)>9.99 else 10))
    pb.legend()
    pb.savefig(f'./Plots/gpy_improved_avg_error.png', dpi=1200)
    pb.clf()

    
