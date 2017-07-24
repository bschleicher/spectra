import on_time_parallel
import calc_a_eff_parallel
import read_data

import pandas as pd
from astropy import time as at
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import numpy as np
from fact.analysis.statistics import li_ma_significance
from multiprocessing import Pool
import read_mars
import matplotlib.colors as colors


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


if __name__ == '__main__':

    # Setup ThetaSqare Cut.
    # Zenith Distance Bins, MC are available from 0 to 60 deg.
    # Energy Bins, MC are av. from 0.2 to 50 TeV
    # If False, effective area is calculated with estimated energy and not MC energy.

    thetasq = 0.07
    zdbins = np.linspace(0, 60, 15)
    ebins = np.logspace(np.log10(200.0), np.log10(50000.0), 11)
    use_mc = True
    star_files = ["/media/michi/523E69793E69574F/daten/hzd_mrk501.txt"]
    # On ISDC, put None, to read automatically processed Ganymed Output from star files:
    ganymed_result = None
    base_path = "/media/michi/523E69793E69574F/daten/"

    # Iterate over a list of input STAR files:
    star_list = []
    for entry in star_files:
        star_list += list(open(entry, "r"))

    # Create the labels for binning in energy and zenith distance.
    zdlabels = range(len(zdbins) - 1)
    elabels = range(len(ebins) - 1)

    # Calcualtion of on time from ganymed input list of star files:

    on_time_per_zd = np.zeros([len(zdbins)-1])
    on_time_per_zd = on_time_parallel.calc_on_time(star_list, zdbins, zdlabels)

    # Read the required leaves of the ganymed-analysis output file, calculate energy estimation and return count
    # histograms:

    select_leaves = ['DataType.fVal', 'MPointingPos.fZd', 'FileId.fVal', 'MTime.fMjd', 'MTime.fTime.fMilliSec',
                     'MTime.fNanoSec', 'MHillas.fSize', 'ThetaSquared.fVal', 'MNewImagePar.fLeakage2']
    on_off_histos = np.zeros([2, len(zdlabels), len(elabels)])

    if not ganymed_result:
        print("\nRead data from Star files. ---------")
        on_off_histos = read_data.histos_from_list_of_mars_files(star_list, select_leaves, zdbins,
                                                                 zdlabels, ebins, elabels, thetasq)

    else:
        print("\nRead data from output ganymed file ---------")
        data_cut = read_mars.read_mars(ganymed_result, leaf_names=select_leaves)
        on_off_histos = read_data.calc_onoffhisto(data_cut, zdbins, zdlabels, ebins, elabels, thetasq)

    print("--------- Finished reading data.")

    on_histo = on_off_histos[0]
    off_histo = on_off_histos[1]

    exc_histo = on_histo - (1 / 5) * off_histo
    exc_histo_err = np.sqrt(on_histo + (1/25) * off_histo)

    overall_significance = li_ma_significance(np.sum(on_histo), np.sum(off_histo))

    # Calculate the effective area:

    a_eff = calc_a_eff_parallel.calc_a_eff_parallel(ebins, zdbins, use_mc=use_mc, theta_square_cut=str(thetasq),
                                                    path=base_path)
    print("\nOn-Time per Zd in s:")
    print(on_time_per_zd)
    print("\nOn-Time:", np.sum(on_time_per_zd), "s")
    print("On-Time:", np.sum(on_time_per_zd) / (60*60), "h")
    flux2d = np.ma.divide(exc_histo, a_eff) / on_time_per_zd[:, np.newaxis]
    flux2d_err = np.ma.divide(exc_histo_err, a_eff) / on_time_per_zd[:, np.newaxis]
    sig = li_ma_significance(on_histo, off_histo)
    flux_e = np.ma.average(flux2d, axis=0, weights=sig)
    flux_e_err = np.sqrt(np.ma.average((flux2d_err**2), axis=0, weights=sig))

    bin_centers = np.power(10, (np.log10(ebins[1:]) + np.log10(ebins[:-1])) / 2)
    bin_width = ebins[1:] - ebins[:-1]

    hess_x = np.array((1.70488, 2.1131, 2.51518, 3.02825, 3.65982, 4.43106, 5.37151, 6.50896, 7.87743, 9.52215,
                       11.4901, 13.8626, 16.8379, 20.4584, 24.8479, 30.2065, 36.7507, 44.8404))
    hess_y = np.array((4.15759e-11, 3.30552e-11, 1.7706e-11, 1.28266e-11, 7.57679e-12, 5.65619e-12, 2.85186e-12,
                       1.9475e-12, 1.10729e-12, 4.91077e-13, 3.00283e-13, 8.96491e-14, 4.27756e-14, 1.24023e-14,
                       3.49837e-15, 3.51992e-15, 2.24845e-15, 1.34066e-15))

    hess_yl = hess_y - np.array([3.47531e-11, 3.01035e-11, 1.61338e-11, 1.17782e-11, 6.91655e-12, 5.18683e-12,
                                 2.57245e-12, 1.75349e-12, 9.80935e-13, 4.16665e-13, 2.49006e-13, 6.43095e-14,
                                 2.83361e-14, 5.3776e-15, 3.49837e-17, 3.51992e-17, 2.24845e-17, 1.34066e-17])
    hess_yh = np.array([4.92074e-11, 3.61898e-11, 1.93749e-11, 1.39342e-11, 8.2767e-12, 6.15261e-12, 3.14967e-12,
                        2.15466e-12, 1.24357e-12, 5.73022e-13, 3.57712e-13, 1.20039e-13, 6.13372e-14, 2.27651e-14,
                        3.49837e-15, 3.51992e-15, 2.24845e-15, 1.34066e-15]) - hess_y

    crab_do_x_l = np.array([251.18867, 398.1072, 630.9573, 1000.0, 1584.8929, 2511.8861, 3981.072, 6309.573, 10000.0])
    crab_do_x_h = np.array([398.1072, 630.9573, 1000.0, 1584.8929, 2511.8861, 3981.072, 6309.573, 10000.0, 15848.929])
    crab_do_x = pow(10, (np.log10(crab_do_x_l) + np.log10(crab_do_x_h)) / 2)
    crab_do_y = np.array([7.600271e-10, 2.436501e-10, 6.902212e-11, 1.939978e-11, 5.119684e-12, 1.4333e-12,
                          3.756243e-13, 1.220526e-13, 2.425761e-14])
    crab_do_y_err = np.array([2.574657e-10, 3.90262e-11, 4.364535e-12, 1.083812e-12, 3.51288e-13, 1.207386e-13,
                              4.346247e-14, 1.683129e-14, 6.689311e-15])

    flux_de = (flux_e / (bin_width / 1000))
    flux_de_err = (flux_e_err / (bin_width / 1000))

    flux_de_err_log10 = symmetric_log10_errors(flux_de, flux_de_err)

    plt.figure("Spectrum")
    ax1 = plt.subplot(121)

    plt.errorbar(x=bin_centers, y=flux_de, yerr=flux_de_err_log10, xerr=[bin_centers-ebins[:-1], ebins[1:]-bin_centers],
                 fmt=".", label="FACT")
    plt.errorbar(x=hess_x * 1000, y=hess_y, yerr=[hess_yl, hess_yh], fmt=".", label="HESS 24.06.2014")
    # plt.errorbar(x=crab_do_x, y=crab_do_y, yerr=crab_do_y_err, xerr=[crab_do_x-crab_do_x_l,crab_do_x_h-crab_do_x],
    #  fmt="o", label="Crab Dortmund")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Flux [$\mathrm{cm^{-2}TeV^{-1}s^{-1}}$]")
    plt.legend()

    ax2 = plt.subplot(222, sharex=ax1)

    plt.errorbar(x=bin_centers, y=np.sum(exc_histo, axis=0), fmt="o", label="Counts")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Excess Events")
    plt.legend()

    ax3 = plt.subplot(224, sharex=ax1)
    plt.errorbar(x=bin_centers, y=li_ma_significance(np.sum(on_histo, axis=0), np.sum(off_histo, axis=0)), fmt="o",
                 label="LiMa Significance")
    plt.xscale("log")
    plt.grid(True)
    plt.xlabel("Energy [GeV]")
    plt.ylabel(" $\mathrm{\sigma}$")
    plt.legend()

    # ZD_min = data_cut["MPointingPos.fZd"].min()
    # ZD_max = data_cut["MPointingPos.fZd"].max()

    fig = plt.figure("2D Plots")
    gs = gspec.GridSpec(2, 4, height_ratios=[1, 4])

    ax5 = fig.add_subplot(gs[1, 0])
    im = ax5.imshow(a_eff, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower')
    plt.ylabel('Zenith Distance [deg]')
    plt.xlabel('log10(E [GeV])')
    plt.title('Effective Area')
    plt.colorbar(im, ax=ax5)

    fig.add_subplot(gs[1, 1])
    plt.imshow(exc_histo, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower')
    plt.title('Excess Events')
    plt.colorbar()

    fig.add_subplot(gs[1, 2])
    plt.imshow(on_histo, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower')
    plt.title('On Events')
    plt.colorbar()

    fig.add_subplot(gs[1, 3])
    plt.imshow(off_histo*0.2, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower')
    plt.title('Off Events')
    plt.colorbar()

    fig.add_subplot(gs[0, :])
    plt.text(0.0, 0.0, "On Time: {0:8.2f} h\n".format(np.sum(on_time_per_zd) / (60 * 60))
             + "$\mathrm{\sigma_{LiMa}}$: " + "{0:3.2f}\n".format(overall_significance))
    plt.axis("off")

    fig2 = plt.figure("Fluxes, Errors, Significance")
    fig2.add_subplot(131)
    plt.imshow(flux2d, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower', norm=colors.LogNorm())
    plt.ylabel('Zenith Distance [deg]')
    plt.xlabel('log10(E [GeV])')
    plt.title('Fluxes')
    plt.colorbar()

    fig2.add_subplot(132)
    plt.imshow(flux2d_err, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower', norm=colors.LogNorm())
    plt.title('Errors of Fluxes')
    plt.colorbar()

    fig2.add_subplot(133)
    plt.imshow(sig, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower')
    plt.title('LiMa Significances')
    plt.colorbar()

    plt.show()
