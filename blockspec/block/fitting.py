import emcee
from scipy.optimize import minimize, differential_evolution
import numpy as np


def powerlaw(x, k, gamma):
    k = k*10**(-11)
    return k*np.power(x/1000, gamma)  # * np.exp(np.divide(x, 6000))


def cutoff_powerlaw(x, k, gamma, ec):
    k = k*10**(-11)
    return k*np.power(x/1000, gamma) * np.exp(np.divide(x, ec*1000))


def fit_ll(spect,
           model='powerlaw',
           bounds=None,
           names=None,
           nwalkers=100,
           nsamples=500,
           nburnin=150):

    if model == 'powerlaw':
        spec_function = powerlaw
        if bounds is None:
            bounds = [[10**(-2), 10**(4)], [-5, 1]]
        if names is None:
            names = ["$\Phi$ [$10^{-11} cm^{-2}s^{-1}TeV^{-1}$]", "$\Gamma$"]

    elif model == 'cutoff_powerlaw':
        spec_function = cutoff_powerlaw
        if bounds is None:
            bounds = [[10**(-3), 10**(4)], [-7, 1], [0.5, 300]]
        if names is None:
            names = ["$\Phi$ [$10^{-11} cm^{-2}s^{-1}TeV^{-1}$]", "$\Gamma$", "$E_c$ [$GeV$]"]

    bin_width = (spect.energy_binning[1:] - spect.energy_binning[:-1])/1000

    emig = np.ma.divide(spect.energy_migration, spect.energy_migration.sum(axis=2)[:, :, np.newaxis])

    emig = emig.filled(0)

    def countssim_zd_e(A):
        sim_flux = spec_function(spect.energy_center, *A)
        Tjki = emig * (sim_flux * bin_width)[np.newaxis, :, np.newaxis]
        Tjki = Tjki * spect.on_time_per_zd[:, np.newaxis, np.newaxis]
        Tjki = Tjki * spect.effective_area[:, :, np.newaxis]
        return np.sum(Tjki, axis=1)

    def log_like_zd_e(A):
        start = 1
        stop = None
        y = spect.on_histo_zenith[:, start:stop]
        bsim = 0.2 * spect.off_histo_zenith[:, start:stop]
        ysim = countssim_zd_e(A)[:, start:stop] + bsim

        # mask = ysim > 0
        mask2 = (bsim > 0) & (ysim > 0)
        b = np.sum(y[mask2] * np.log(bsim[mask2]) - bsim[mask2])
        s = np.sum(y[mask2] * np.log(ysim[mask2]) - ysim[mask2])  # - np.log(factorial(y))
        return s - b

    bounds = np.array(bounds)

    def bla(A):
        return -1*log_like_zd_e(A)

    result = differential_evolution(bla, bounds)

    print(result)

    def lnprior(theta):
        theta = np.array(theta)
        if np.all((theta > bounds.T[0]) & (theta < bounds.T[1])):
            return 0.0
        return -np.inf

    def lnprob(A):
        lp = lnprior(A)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_like_zd_e(A)

    ndim = bounds.shape[0]
    pos = [result["x"] + result["x"] * 0.01 * np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(pos, nsamples)

    samples = sampler.chain[:, nburnin:, :].reshape((-1, ndim))
    parameters = []
    for i in map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [5, 16, 50, 84, 95], axis=0))):
        parameters.append(i)

    return {"parameters": parameters,
            "fit_function": spec_function,
            "boundaries": bounds,
            "labels": names,
            "samples": sampler.chain,
            "lnprobability": sampler.lnprobability}