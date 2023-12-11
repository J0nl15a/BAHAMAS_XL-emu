# BAHAMAS_XL-emu
A Gaussian Process emulator framework for predicting the boost function of the matter power spectra from input cosmologies. (IN DEVELOPMENT)

* Trained on 150 dark-matter only, cosmological models from the BAHAMAS XL (IN DEVELOPMENT) simulation suite; models are spilt in three different box sizes with the same number of particles: 350, 700 & 1400Mpc on a side.
* 9 input parameters: $\Omega_m$, $f_b$, $H_0$ (given as $h$), $A_s$, $n_s$, $\Omega_{\nu}h^2$, $w_0$, $w_a$ & $\alpha_s$.
* Includes data pre-processing for scale cuts and rebinning via spline interpolation.
* Preset for power spectra computed using POWMES and nbodykit.
* Compatible with CAMB or CLASS linear theory.
* Modes for emulating on a 'k by k' basis and adjusting the variance to 'weight' the models.

* Built using GPy package: https://github.com/SheffieldML/GPy/tree/devel

https://github.com/J0nl15a/BAHAMAS_XL-emu/assets/104151005/a69810a7-9f9a-48e0-8bc8-ef1704956000

