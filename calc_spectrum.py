from on_time_parallel import calc_on_time
from calc_a_eff_parallel import calc_a_eff_parallel
import read_data

import numpy as np
from fact.analysis.statistics import li_ma_significance
from read_mars import read_mars


def symmetric_log10_errors(value, error):
    """ Calculate upper and lower error, that appear symmetric in loglog-plots.

    :param value: ndarray or float
    :param error: ndarray or float
    :return: array of lower error and upper error.
    """
    error /= (value * np.log(10))
    error_l = value - np.ma.power(10, (np.ma.log10(value) - error))
    error_h = np.ma.power(10, (np.ma.log10(value) + error)) - value
    return [error_l, error_h]


def calc_spectrum(star_files=["/media/michi/523E69793E69574F/daten/421_flare_ed.txt"],
                  ganymed_result=None,
                  base_path="/media/michi/523E69793E69574F/daten/",
                  thetasqare_cut=0.04,
                  zdbins=np.linspace(0, 60, 15),
                  zdlabels=range(len(zdbins) - 1),
                  ebins=np.logspace(np.log10(200.0), np.log10(50000.0), 12),
                  elabels=range(len(ebins) - 1),
                  correction_factors=False
                  ):

    # Setup ThetaSqare Cut.
    # Zenith Distance Bins, MC are available from 0 to 60 deg.
    # Energy Bins, MC are av. from 0.2 to 50 TeV
    # If False, effective area is calculated with estimated energy and not MC energy.

    # On ISDC, put None, to read automatically processed Ganymed Output from star files:

    # Create the labels for binning in energy and zenith distance.

    # Iterate over a list of input STAR files:
    star_list = []
    for entry in star_files:
        star_list += list(open(entry, "r"))

    # Calcualtion of on time from ganymed input list of star files:

    on_time_per_zd = calc_on_time(star_list, zdbins, zdlabels)

    # Read the required leaves of the ganymed-analysis output file and calculate energy estimation:

    select_leaves = ['DataType.fVal', 'MPointingPos.fZd', 'FileId.fVal', 'MTime.fMjd', 'MTime.fTime.fMilliSec',
                     'MTime.fNanoSec', 'MHillas.fSize', 'ThetaSquared.fVal', 'MNewImagePar.fLeakage2']

    if not ganymed_result:
        print("\nRead data from Star files. ---------")
        histos = read_data.histos_from_list_of_mars_files(star_list, select_leaves, zdbins,
                                                          zdlabels, ebins, elabels, thetasqare_cut)

    else:
        print("\nRead data from output ganymed file ---------")
        data_cut = read_mars(ganymed_result, leaf_names=select_leaves)
        histos = read_data.calc_onoffhisto(data_cut, zdbins, zdlabels, ebins, elabels, thetasqare_cut)

    print("--------- Finished reading data.")

    print(histos)
    print(histos)

    on_histo = histos[0][0]
    off_histo = histos[0][1]

    exc_histo = on_histo - (1 / 5) * off_histo
    exc_histo_err = np.sqrt(on_histo + (1 / 25) * off_histo)

    overall_significance = li_ma_significance(np.sum(on_histo), np.sum(off_histo))

    # Calculate the effective area:

    a_eff = calc_a_eff_parallel(ebins,
                                zdbins,
                                correction_factors=correction_factors,
                                theta_square_cut=thetasqare_cut,
                                path=base_path)

    print(on_time_per_zd)
    print("On-Time:", np.sum(on_time_per_zd), "s")
    print("On-Time:", np.sum(on_time_per_zd) / (60 * 60), "h")

    # Calculate an effective effective area, that scales the effective area per on time
    a_eff = (a_eff * on_time_per_zd[:, np.newaxis]) / np.sum(on_time_per_zd)
    flux = np.divide(np.sum(exc_histo, axis=0), np.sum(a_eff, axis=0))
    flux = np.divide(flux, (np.sum(on_time_per_zd)))
    flux_err = np.ma.divide(np.sqrt(np.sum(on_histo, axis=0) + (1/25) * np.sum(off_histo, axis=0)),
                            np.sum(a_eff, axis=0)) / np.sum(on_time_per_zd)
    sig = li_ma_significance(on_histo, off_histo)
    # flux_e = np.ma.average(flux2d, axis=0, weights=sig)
    # flux_e_err = np.sqrt(np.ma.average((flux2d_err ** 2), axis=0, weights=sig))

    bin_centers = np.power(10, (np.log10(ebins[1:]) + np.log10(ebins[:-1])) / 2)
    bin_width = ebins[1:] - ebins[:-1]

    flux_de = np.divide(flux, np.divide(bin_width, 1000))
    flux_de_err = np.divide(flux_err, np.divide(bin_width, 1000))  # / (flux_de * np.log(10))
    flux_de_err_log10 = symmetric_log10_errors(flux_de, flux_de_err)

    return \
        bin_centers, bin_centers-ebins[:-1], ebins[1:]-bin_centers, \
        flux_de, flux_de_err_log10[0], flux_de_err_log10[1], \
        sig, overall_significance
