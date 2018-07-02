from spectrum_class import Spectrum
import matplotlib.pyplot as plt
import numpy as np

# Monte Carlo simulated events:
ceres_list = []

for i in range(8):
    ceres_list.append("/home/michi/read_mars/ceres_part" + str(i) + ".h5")

# Monte Carlo surviving events:

ganymed_mc = "/media/michi/523E69793E69574F/daten/gamma/hzd_gammasall-analysis.root"

# Star files of data:

#star_files = ["/media/michi/523E69793E69574F/daten/crab2.txt"]
#star_files = ["/media/michi/523E69793E69574F/daten/Mrk501/blocks/20140623__20140623.txt"]
#star_files = ["/media/michi/523E69793E69574F/daten/mrk501.txt"]
#star_files= ["/media/michi/523E69793E69574F/daten/Crab/blocks/20160225_20180124.txt"]
star_files= ["/media/michi/523E69793E69574F/daten/Crab/blocks/20140820_20141001.txt"]
star_list = []
for entry in star_files:
    star_list += list(open(entry, "r"))

# Ganymed result:
#ganymed_result = "/media/michi/523E69793E69574F/daten/crab2-analysis.root"
#ganymed_result = "/media/michi/523E69793E69574F/daten/Mrk501/19-analysis.root"
#ganymed_result = "/media/michi/523E69793E69574F/daten/mrk501-analysis.root"
#ganymed_result = "/media/michi/523E69793E69574F/daten/Crab/9-analysis.root"
ganymed_result = "/media/michi/523E69793E69574F/daten/Crab/5-analysis.root"


spectrum = Spectrum(run_list_star=star_list,
                    ganymed_file_data=ganymed_result,
                    list_of_ceres_files=ceres_list,
                    ganymed_file_mc=ganymed_mc)
# spectrum.set_correction_factors()
def forest_energy_impact(gamma_on):
    from sklearn.externals import joblib

    clf = joblib.load('/home/michi/random_forest_impact.pkl')
    training_variables_impact = ['MPointingPos.fZd', 'MHillas.fLength', 'MHillas.fSize', 'MHillas.fWidth',
                                 'MHillasSrc.fDist', 'MHillasExt.fM3Long', 'MHillasExt.fSlopeLong',
                                 'MHillasExt.fSlopeSpreadWeighted', 'MHillasExt.fTimeSpreadWeighted',
                                 'MHillasSrc.fCosDeltaAlpha']
    X = gamma_on[training_variables_impact]

    predicted = clf.predict(X)

    gamma_on["impact"] = predicted


    clf = joblib.load('/home/michi/random_forest_impact_energy.pkl')

    training_variables = [
    'MPointingPos.fZd',
    'MHillas.fSize',
    'MNewImagePar.fLeakage2',
    'MHillas.fWidth',
    'MHillasSrc.fDist',
    'MHillasExt.fM3Long',
    'MHillasExt.fSlopeLong',
    'impact'
    ]
    X = gamma_on[training_variables]

    predicted = clf.predict(X)
    return predicted

def forest_energy(gamma_on):
    from sklearn.externals import joblib

    clf = joblib.load('/home/michi/random_forest.pkl')

    training_variables = [
    'MPointingPos.fZd',
    'MHillas.fSize',
    'MNewImagePar.fLeakage2',
    'MHillas.fWidth',
    'MHillasSrc.fDist',
    'MHillasExt.fM3Long',
    'MHillasExt.fSlopeLong'
    ]
    X = gamma_on[training_variables]

    predicted = clf.predict(X)
    return predicted
#spectrum.set_theta_square(0.040)
spectrum.optimize_theta()
#spectrum.optimize_ebinning(min_counts_per_bin=10, sigma_threshold=5)
spectrum.set_energy_binning(np.logspace(np.log10(200), np.log10(50000), 12))
spectrum.set_correction_factors(False)
spectrum.calc_differential_spectrum(use_multiprocessing=True,
                                    efunc=forest_energy_impact,
                                    #slope_goal=0.3
                                    )


spectrum.plot_flux(crab_magic=True)
spectrum.plot_thetasq()
path = "/media/michi/523E69793E69574F/xy.json"
spectrum.save(path)

plt.show()

second = Spectrum()
second.load(path)

second.plot_flux(hess_flare=True)
second.plot_thetasq()

plt.show()
