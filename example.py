import matplotlib.pyplot as plt
import numpy as np

from blockspec.spec import Spectrum
from blockspec.spec.energy_functions import forest_energy_length

from blockspec.spec.calc_a_eff_parallel import calc_a_eff_parallel_hd5

# Monte Carlo simulated events:
ceres_list = []

for i in range(8):
    ceres_list.append("/home/michi/read_mars/ceres_part" + str(i) + ".h5")

# Monte Carlo surviving events:

#ganymed_mc = "/media/michi/523E69793E69574F/daten/gamma/hzd_gammasall-analysis.root"
ganymed_mc = "/home/michi/ml/mc/star/dortmund_disp2018-analysis.root"
# Star files of data:

#star_files = ["/home/michi/data/bump/july19_21.txt"]
#star_files = ["/home/michi/data/threshold/blocks/(289.097, 402.875].txt"]
#star_files = ["/home/michi/data/mrk501daily/blocks/20140623.txt"]
#star_files = ["/home/michi/data/mrk501_bb/blocks/20140623_20140623.txt"]

star_files = ["/home/michi/data/crab_zd/blocks/ 35 to  60.txt"]
#star_files = ["/media/michi/523E69793E69574F/daten/crab2.txt"]
#star_files = ["/media/michi/523E69793E69574F/daten/Mrk501/blocks/20140623__20140623.txt"]
#star_files = ["/media/michi/523E69793E69574F/daten/hzd_mrk501.txt"]
#star_files= ["/media/michi/523E69793E69574F/daten/Crab/blocks/20160225_20180124.txt"]
#star_files= ["/media/michi/523E69793E69574F/daten/Crab/blocks/20140820_20141001.txt"]

star_list = []
for entry in star_files:
    star_list += list(open(entry, "r"))


# Ganymed result:
#ganymed_result = "/home/michi/data/bump/july19_21-analysis.root"
#ganymed_result = "/home/michi/data/threshold/(289.097, 402.875]-analysis.root"
#ganymed_result = "/home/michi/data/mrk501daily/20140623-analysis.root"
#ganymed_result = "/home/michi/data/mrk501_bb/14-analysis.root"
ganymed_result = "/home/michi/data/crab_zd/ 35 to  60-analysis.root"

#ganymed_result = "/media/michi/523E69793E69574F/daten/crab2-analysis.root"
#ganymed_result = "/media/michi/523E69793E69574F/daten/Mrk501/19-analysis.root"
#ganymed_result = "/media/michi/523E69793E69574F/daten/hzd_mrk501-analysis.root"
#ganymed_result = "/media/michi/523E69793E69574F/daten/Crab/9-analysis.root"
#ganymed_result = "/media/michi/523E69793E69574F/daten/Crab/5-analysis.root"


spectrum = Spectrum(run_list_star=star_list,
                    ganymed_file_data=ganymed_result,
                    list_of_ceres_files=ceres_list,
                    ganymed_file_mc=ganymed_mc)
# spectrum.set_correction_factors()


def forest_energy(gamma_on):
    from sklearn.externals import joblib

    clf = joblib.load('/home/michi/random_forest.pkl')

    training_variables = ['MPointingPos.fZd',
                          'MHillas.fSize',
                          'MNewImagePar.fLeakage2',
                          'MHillas.fWidth',
                          'MHillasSrc.fDist',
                          'MHillasExt.fM3Long',
                          'MHillasExt.fSlopeLong']
    X = gamma_on[training_variables]

    predicted = clf.predict(X)
    return predicted

def cut(x):
    return x['MHillas.fSize'] > 125


def cut_function(x):
    size = x['MHillas.fSize']
    a = (np.pi * x['MHillas.fLength'] * x['MHillas.fWidth']) < (np.log10(size) * 898 - 1535)
    b = size > 125
    return np.logical_and(a, b)

ebins = np.logspace(np.log10(200), np.log10(50000), 16)
zdbins = np.linspace(0, 60, 31)

# areas = calc_a_eff_parallel_hd5(ebins, zdbins, False,
#                                0.04, path=ganymed_mc, list_of_hdf_ceres_files=ceres_list,
#                                energy_function=forest_energy_impact, slope_goal=None, impact_max=54000.0,
 #                               cut=cut)
# spectrum.set_effective_area(areas)
spectrum.set_theta_square(0.08)
# spectrum.optimize_theta()


#spectrum.optimize_ebinning(min_counts_per_bin=10, sigma_threshold=3, min_bin_percentage=0.2)
spectrum.set_energy_binning(ebins)
spectrum.set_zenith_binning(zdbins)
#spectrum.set_correction_factors(True)
spec = spectrum.calc_differential_spectrum(use_multiprocessing=False,
                                           efunc=forest_energy_length,
                                           # slope_goal=0.3,
                                           #cut=cut,
                                           )
print(spec)
print(spectrum.energy_migration)
print(spectrum.on_histo)
print(spectrum.off_histo)
print(spectrum.excess_histo)
print(spectrum.excess_histo_err)

spectrum.plot_flux(crab_do=True, hess_flare=False)
spectrum.plot_thetasq()
path = "/media/michi/523E69793E69574F/xy.json"
spectrum.save(path)

plt.show()

second = Spectrum()
second.load(path)

from blockspec.block.fitting import fit_ll

def powerlaw(x, k, gamma):
    k = np.power(10, k)
    return k*np.power(x/1000, gamma)  # * np.exp(np.divide(x, 6000))

bounds = [[-20, -5], [-8, 1]]
labels = ["$\Phi$ [cm^{-2}s^{-1}TeV^{-1}]", "$\Gamma$"]
names = ["flux", "index"]


def powerlaw(x, k, gamma):
    k = k*10**(-11)
    #b = np.power(10, b)
    ##gaussian = np.divide(b, sigma) * np.exp(-np.divide(x/1000 - mu,4*sigma)**2)
    #return k*np.power(x/1000, gamma) + gaussian # * np.exp(np.divide(x, 6000))
    log_thing = gamma * np.log(x / 1000) # - np.divide(x, xc * 1000)
    return k * np.exp(log_thing)

bounds = [[0, 20], [-8, 1]]
labels = ["$\Phi$ $[\mathrm{10^{-11}cm^{-2}s^{-1}TeV^{-1}}]$", "$\Gamma$"]
names = ["flux", "index", "cutoff"]


fit = fit_ll(second, model=powerlaw, bounds=bounds, labels=labels, names=names, nsamples=1500)
print(fit)

n = np.sum(0.2 * second.off_histo_zenith > 0)
print(n)

from corner import corner

corner(fit["samples"][:,200:,:].reshape(-1,len(labels)), show_titles=True, labels=labels)

corner(fit["samples"][:,150:,:].reshape(-1,len(labels)), show_titles=True)

second.plot_flux(crab_do=True)

ax = plt.gca()
from blockspec.block.block import conf_from_sample

x, mid, low, up = conf_from_sample(fit["samples"][:,150:,:], powerlaw)
ax.plot(x, mid, label="Fit LogLike")
ax.fill_between(x, low, up, label="LogLike 68 % confidence", alpha=0.4)
plt.legend()

second.plot_thetasq()




plt.show()

