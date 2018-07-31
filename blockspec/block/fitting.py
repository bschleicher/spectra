import emcee
from scipy.optimize import differential_evolution
import numpy as np


def powerlaw_model(bounds=None, labels=None, names=None, start_values=None):

    def powerlaw(x, k, gamma):
        k = k*10**(-11)
        return k*np.power(x/1000, gamma)  # * np.exp(np.divide(x, 6000))

    if bounds is None:
        bounds = [[10**(-2), 10**(4)], [-8, 1]]
    if labels is None:
        labels = ["$\Phi$ [$10^{-11} cm^{-2}s^{-1}TeV^{-1}$]", "$\Gamma$"]
    if names is None:
        names = ["flux", "index"]

    return powerlaw, start_values, bounds, labels, names


def cutoff_powerlaw_model(bounds=None, labels=None, names=None, pivot=1000, start_values=None):
    """ Default values for a cutoff powerlaw model"""

    # use the logarithmic version to prevent rounding errors
    def cutoff_powerlaw(x, k, gamma, ec):
        k = k*10**(-11)
        log_thing = gamma * np.log(x / pivot) - np.divide(x, ec*1000)
        return k * np.exp(log_thing)

    if bounds is None:
        bounds = [[10**(-3), 10**(4)], [-8, 1], [0.5, 300]]
    if labels is None:
        labels = ["$\Phi$ [$10^{-11} cm^{-2}s^{-1}TeV^{-1}$]", "$\Gamma$", "$E_c$ [$TeV$]"]
    if names is None:
        names = ["flux", "index", "cutoff"]

    return cutoff_powerlaw, start_values, bounds, labels, names


def line_model(start_values=None, names=None, labels=None, bounds=None):

    def line(x, a, b):
        return a + b * x

    if start_values is None:
        start_values = [10**(-2), -2]
    if names is None:
        names = ["flux", "index"]
    if labels is None:
        labels = ["y0", "m"]

    return line, start_values, bounds, labels, names


def _parameter_values_from_samples(samples):
    parameters = []
    # Create arrays for paramters with order: [value, err_up, err_low]
    for i in map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))):
        parameters.append(i)
    return parameters


def fit_ll(spect,
           nwalkers=100,
           nsamples=500,
           nburnin=150,
           model='powerlaw',
           start_values=None,
           bounds=None,
           labels=None,
           names=None):

    if model == 'powerlaw':
        spec_function, start_values, bounds, labels, names = powerlaw_model(bounds=bounds,
                                                                            labels=labels,
                                                                            names=names)

    elif model == 'cutoff_powerlaw':
        spec_function, start_values, bounds, labels, names = cutoff_powerlaw_model(bounds=bounds,
                                                                                   labels=labels,
                                                                                   names=names)
    elif isinstance(model, function):
        spec_function = model
        if None in (bounds, names, labels):
            raise ValueError('If you provide a function as a model, you also have to provide bounds, names and labels.')

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

    parameters = _parameter_values_from_samples(samples)

    return {"parameters": parameters,
            "boundaries": bounds,
            "labels": labels,
            "names": names,
            "samples": sampler.chain,
            "lnprobability": sampler.lnprobability}


def fit_points(spect,
               min_energy=1000,
               min_significance=0.7,
               min_len=2,
               high_bin=-1,
               fit_log=True,
               use_sigma=False,
               model="line",
               start_values=None,
               bounds=None,
               labels=None,
               names=None):

    from scipy.optimize import curve_fit
    from scipy import integrate

    if model == 'line':
        line, start_values, bounds, labels, names = line_model()
    elif isinstance(model, function):
        if None in (start_values, names, labels):
            raise ValueError('If you provide a function as a model, '
                             'you also have to provide start_values, names and labels.')
    else:
        raise ValueError("'model' must either be line or a function")

    selection = (spect.energy_center > min_energy) & (spect.significance_histo > min_significance)

    x = spect.energy_center[selection][:high_bin] / 1000
    if fit_log:
        x = np.log10(x)

    if len(x) > min_len:
        y = np.abs(spect.differential_spectrum[selection][:high_bin])

        if fit_log:
            y = np.log10(y)

        fit_kwargs = {"xdata": x, "ydata": y, "p0": start_values}

        if use_sigma:
            fit_kwargs["sigma"] = np.abs(spect.differential_spectrum_err[0][selection][:high_bin])
        if bounds is not None:
            fit_kwargs["bounds"] = bounds

        popt, pcov = curve_fit(line, **fit_kwargs)

        sample = np.random.multivariate_normal(popt, pcov, 10000)

        if fit_log:
            sample[:, :1] = np.power(10, sample[:, :1])

        parameters = _parameter_values_from_samples(sample)

        x3 = np.logspace(np.log10(1), np.log10(50), 200)
        linepoints = np.power(10, line(np.log10(x3), popt[0], popt[1]))
        y3 = np.power(10, line(np.log10(x3)[:, np.newaxis], sample[:, 0], sample[:, 1]))
        low = np.percentile(y3, 16, axis=1)
        up = np.percentile(y3, 84, axis=1)
        # low90 = np.percentile(y3, 5, axis=1)
        # up90 = np.percentile(y3, 95, axis=1)

        # Create arrays for paramters with order: [value, err_up, err_low]

        lower = integrate.simps(low, x3)
        upper = integrate.simps(up, x3)
        value = integrate.simps(linepoints, x3)
        flux_integral = [value, upper - value, value - lower]

        return {"parameters": parameters,
                "names": names,
                "labels": labels,
                "samples": sample}
    else:
        return {"parameters": [[np.nan] * 3] * len(names),
                "names": names,
                "labels": labels,
                "samples": None}