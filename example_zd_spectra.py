import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

from blockspec.spec import Spectrum

def forest_energy_length(gamma_on):
    from sklearn.externals import joblib

    clf = joblib.load('/home/michi/ml/mc/star/forest_impact_disp2018_length.pkl')
    training_variables_impact = ['MPointingPos.fZd', 'MHillas.fSize', 'MHillas.fWidth',
                                 'MHillasSrc.fDist', 'MHillasExt.fM3Long', 'MHillasExt.fSlopeLong',
                                 'MHillasSrc.fCosDeltaAlpha']
    X = gamma_on[training_variables_impact]

    predicted = clf.predict(X)

    gamma_on["impact"] = predicted

    clf = joblib.load('/home/michi/ml/mc/star/forest_impact_energy2018_length.pkl')

    training_variables = ['MPointingPos.fZd', 'MHillas.fSize', 'MNewImagePar.fLeakage2', 'MHillas.fLength',
                          'MHillasSrc.fDist', 'MHillasExt.fM3Long', 'MHillasExt.fSlopeLong', 'impact']
    X = gamma_on[training_variables]

    predicted = clf.predict(X)
    return predicted


ceres_list = []

for i in range(8):
    ceres_list.append("/home/michi/read_mars/ceres_part" + str(i) + ".h5")


ganymed_mc = "/home/michi/ml/mc/star/dortmund_disp2018-analysis.root"
star_files = ["/home/michi/data/mrk501_bb/blocks/20140623_20140623.txt"]
star_list = []
for entry in star_files:
    star_list += list(open(entry, "r"))

ganymed_result = "/home/michi/data/mrk501_bb/14-analysis.root"


spectrum = Spectrum(run_list_star=star_list,
                    ganymed_file_data=ganymed_result,
                    list_of_ceres_files=ceres_list,
                    ganymed_file_mc=ganymed_mc)



spectrum.set_theta_square(0.04)


ebins = np.logspace(np.log10(200), np.log10(50000), 16)
zdbins = np.linspace(0, 60, 31)
spectrum.set_energy_binning(ebins)
spectrum.set_zenith_binning(zdbins)

spec = spectrum.calc_differential_spectrum(use_multiprocessing=False,
                                           efunc=forest_energy_length)

spectrum.plot_flux(crab_do=True, hess_flare=False)
spectrum.plot_thetasq()
path = "/media/michi/523E69793E69574F/xy.json"
spectrum.save(path)

plt.show()