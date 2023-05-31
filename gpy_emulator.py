import numpy as np
import GPy
import pylab as pb
from data_loader import BXL_DMO_Pk
import random

mlp = GPy.kern.MLP(9)
linear = GPy.kern.Linear(9)
polynomial = GPy.kern.Poly(9)
RBF = GPy.kern.RBF(9)
Bias = GPy.kern.Bias(9)

class gpy_emulator:

    def __init__(self, P_k_data, kern):
        self.test_models = P_k_data.test_models

        self.model = GPy.models.GPHeteroscedasticRegression(P_k_data.X_train, P_k_data.Y_train, kern)
        self.model.optimize()
        self.y = self.model.predict(P_k_data.X_test.reshape(-1, 1))

    def plot(self):
        self.model.plot()
        pb.errorbar(P_k_data.X_train, P_k_data.Y_train, yerr=np.array(self.model.likelihood.flattened_parameters).flatten(), fmt=None, ecolor='r', zorder=1)

        pb.title('GPy test 1 (kernel = ' + f"{kern}" + ')')
        pb.xlabel('k (1/Mpc)')
        pb.ylabel('P(k) (Mpc^3)')
        pb.xscale('log')
        pb.yscale('log')

        pb.savefig('./Plots/gpy_test_1.pdf', dpi=800)

if __name__ == "__main__":
    test_model = random.randint(0, 149)
    nbk_boost = BXL_DMO_Pk(test_model, 100, pk = 'nbk-rebin')
    gpy = gpy_emulator(nbk_boost, linear)

    print(gpy.model.kern)
    print(gpy.y)
    
