import matplotlib.gridspec as gspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import read_mars
from fact.analysis.statistics import li_ma_significance
from scipy.optimize import curve_fit

from blockspec.spec import calc_a_eff_parallel
from blockspec.spec import read_data

from blockspec.spec import on_time_parallel


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
    thetasq = 0.023
    zdbins = np.linspace(0, 60, 15)
    #ebins = np.logspace(np.log10(200.0), np.log10(50000.0), 12)
    ebins = np.array([251.18864315, 398.10717055, 630.95734448, 1000., 1584.89319246, 2511.88643151,
                      3981.07170553, 6309.5734448, 10000. , 15848.93192461])
    #ebins = np.array([200.0, 475.0, 726.0, 1236.0, 2084.0, 3642.0, 50000])
    corr_factors = False
    star_files = ["/media/michi/523E69793E69574F/daten/crab2.txt"]
    # On ISDC, put None, to read automatically processed Ganymed Output from star files:
    ganymed_result = "/media/michi/523E69793E69574F/daten/crab2-analysis.root"

    base_path = "/media/michi/523E69793E69574F/daten/"

    # Create the labels for binning in energy and zenith distance.
    zdlabels = range(len(zdbins) - 1)
    elabels = range(len(ebins) - 1)

    # Iterate over a list of input STAR files:
    star_list = []
    for entry in star_files:
        star_list += list(open(entry, "r"))

    # Calcualtion of on time from ganymed input list of star files:

    on_time_per_zd = np.zeros([len(zdbins)-1])
    on_time_per_zd = on_time_parallel.calc_on_time(star_list, zdbins, zdlabels)

    # Read the required leaves of the ganymed-analysis output file and calculate energy estimation:

    select_leaves = ['DataType.fVal', 'MPointingPos.fZd', 'FileId.fVal', 'MTime.fMjd', 'MTime.fTime.fMilliSec',
                     'MTime.fNanoSec', 'MHillas.fSize', 'ThetaSquared.fVal', 'MNewImagePar.fLeakage2']

    if not ganymed_result:
        print("\nRead data from Star files. ---------")
        histos = read_data.histos_from_list_of_mars_files(star_list, select_leaves, zdbins,
                                                          zdlabels, ebins, elabels, thetasq)

    else:
        print("\nRead data from output ganymed file ---------")
        data_cut = read_mars.read_mars(ganymed_result, leaf_names=select_leaves)
        histos = read_data.calc_onoffhisto(data_cut, zdbins, zdlabels, ebins, elabels, thetasq)

    print("--------- Finished reading data.")

    print(histos)
    print(histos)

    on_histo = histos[0][0]
    off_histo = histos[0][1]

    exc_histo = on_histo - (1 / 5) * off_histo
    exc_histo_err = np.sqrt(on_histo + (1 / 25) * off_histo)

    overall_significance = li_ma_significance(np.sum(on_histo), np.sum(off_histo))

    # Calculate the effective area:

    ceres_list= []

    for i in range(8):
        ceres_list.append("/home/michi/read_mars/ceres_part" + str(i) + ".h5")

    a_eff = calc_a_eff_parallel.calc_a_eff_parallel_hd5(ebins, zdbins,
                                                        correction_factors=corr_factors,
                                                        theta_square_cut=str(thetasq),
                                                        path=base_path + "gamma/hzd_gammasall-analysis.root",
                                                        list_of_hdf_ceres_files=ceres_list)

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

    hess_x = np.array((1.70488, 2.1131, 2.51518, 3.02825, 3.65982, 4.43106, 5.37151, 6.50896, 7.87743, 9.52215, 11.4901,
                       13.8626, 16.8379, 20.4584, 24.8479, 30.2065, 36.7507, 44.8404))
    hess_y = np.array((4.15759e-11, 3.30552e-11, 1.7706e-11, 1.28266e-11, 7.57679e-12, 5.65619e-12, 2.85186e-12,
                       1.9475e-12, 1.10729e-12, 4.91077e-13, 3.00283e-13, 8.96491e-14, 4.27756e-14, 1.24023e-14,
                       3.49837e-15, 3.51992e-15, 2.24845e-15, 1.34066e-15))

    hess_yl = hess_y - np.array(
        [3.47531e-11, 3.01035e-11, 1.61338e-11, 1.17782e-11, 6.91655e-12, 5.18683e-12, 2.57245e-12, 1.75349e-12,
         9.80935e-13, 4.16665e-13, 2.49006e-13, 6.43095e-14, 2.83361e-14, 5.3776e-15, 3.49837e-17, 3.51992e-17,
         2.24845e-17, 1.34066e-17])
    hess_yh = np.array(
        [4.92074e-11, 3.61898e-11, 1.93749e-11, 1.39342e-11, 8.2767e-12, 6.15261e-12, 3.14967e-12, 2.15466e-12,
         1.24357e-12, 5.73022e-13, 3.57712e-13, 1.20039e-13, 6.13372e-14, 2.27651e-14, 3.49837e-15, 3.51992e-15,
         2.24845e-15, 1.34066e-15]) - hess_y

    crab_do_x_l = np.array([251.18867, 398.1072, 630.9573, 1000.0, 1584.8929, 2511.8861, 3981.072, 6309.573, 10000.0])
    crab_do_x_h = np.array([398.1072, 630.9573, 1000.0, 1584.8929, 2511.8861, 3981.072, 6309.573, 10000.0, 15848.929])
    crab_do_x = pow(10, (np.log10(crab_do_x_l) + np.log10(crab_do_x_h)) / 2)
    crab_do_y = np.array(
        [7.600271e-10, 2.436501e-10, 6.902212e-11, 1.939978e-11, 5.119684e-12, 1.4333e-12, 3.756243e-13, 1.220526e-13,
         2.425761e-14])
    crab_do_y_err = np.array(
        [2.574657e-10, 3.90262e-11, 4.364535e-12, 1.083812e-12, 3.51288e-13, 1.207386e-13, 4.346247e-14, 1.683129e-14,
         6.689311e-15])

    flare_dc = pd.read_csv("/home/michi/Downloads/flare_dc_spectrum.csv")



    flux_de = np.divide(flux, np.divide(bin_width, 1000))
    flux_de_err = np.divide(flux_err,np.divide(bin_width, 1000)) # / (flux_de * np.log(10))
    flux_de_err_log10 = symmetric_log10_errors(flux_de, flux_de_err)

    def powerlaw(x, gamma, scale):
        return scale*np.power(x/1000,gamma)

    try:
        selection=(bin_centers > 300) & (bin_centers < 8000)
        popt, pcov = curve_fit(powerlaw, bin_centers[selection], flux_de[selection], p0=[-2.7, 10e-11])

        print(popt)
        print(np.sqrt(np.diag(pcov)))
    except:
        pass

    plt.figure("Spectrum")
    ax1 = plt.subplot(121)

    plt.errorbar(x=bin_centers, y=flux_de, yerr=flux_de_err_log10, xerr=[bin_centers-ebins[:-1], ebins[1:]-bin_centers]
                 , fmt=".", label="F")

    #plt.errorbar(x=np.power(10,flare_dc.x), y=flare_dc.y, xerr=[np.power(10, flare_dc.x)-np.power(10,flare_dc.x-0.1),
    #                                                            np.power(10, flare_dc.x+0.1)-np.power(10, flare_dc.x)],
    #             yerr=[flare_dc.yerr_low, flare_dc.yerr_high], fmt=".", label="Jens")
    #plt.errorbar(x=hess_x * 1000, y=hess_y, yerr=[hess_yl, hess_yh], fmt=".", label="HESS 24.06.2014")
    plt.errorbar(x=crab_do_x, y=crab_do_y, yerr=crab_do_y_err, xerr=[crab_do_x-crab_do_x_l,crab_do_x_h-crab_do_x],
      fmt="o", label="Crab Dortmund")


    savedf = np.vstack((bin_centers, bin_centers-ebins[:-1], ebins[1:]-bin_centers, flux_de, flux_de_err_log10[0], flux_de_err_log10[1]))
    np.savetxt("/home/michi/test/flare.txt", savedf.T, fmt="%.4e", header="Bin_center Bin_low Bin_high Flux Error_low Error_high")


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
    im = ax5.imshow(a_eff, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower', interpolation=None)
    plt.ylabel('Zenith Distance [deg]')
    plt.xlabel('log10(E [GeV])')
    plt.title('Effective Area')
    plt.colorbar(im, ax=ax5)

    fig.add_subplot(gs[1, 1])
    plt.imshow(exc_histo, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower', interpolation=None)
    plt.title('Excess Events')
    plt.colorbar()

    fig.add_subplot(gs[1, 2])
    plt.imshow(on_histo, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower', interpolation=None)
    plt.title('On Events')
    plt.colorbar()

    fig.add_subplot(gs[1, 3])
    plt.imshow(off_histo*0.2, extent=[np.log10(200), np.log10(50000), 0, 60], aspect=0.1, origin='lower', interpolation=None)
    plt.title('Off Events')
    plt.colorbar()

    fig.add_subplot(gs[0, 2:4])
    plt.plot((zdbins[1:]+zdbins[0:-1])/2,on_time_per_zd,"o")
    plt.xlabel('Zenith Distance [deg]')
    plt.ylabel('On Time [s]')

    fig.add_subplot(gs[0, 0:1])
    plt.text(0.0, 0.0, "Number of runs: {0:8}".format(len(star_list)) + "\nThetaSqare Cut: {0:1.3f}".format(thetasq)
             + "\nOn Time: {0:8.2f} h\n".format(np.sum(on_time_per_zd) / (60 * 60))
             + "$\mathrm{\sigma_{LiMa}}$: " + "{0:3.2f}\n".format(overall_significance))
    plt.axis("off")


    plt.figure("ThetaSqare")
    plt.errorbar(x=(histos[1][0][1][1:]+histos[1][0][1][0:-1])/2, y=histos[1][0][0],
                 xerr=(histos[1][0][1][2]-histos[1][0][1][1])/2, fmt=".", label="On Data")
    plt.errorbar(x=(histos[1][1][1][1:] + histos[1][1][1][0:-1]) / 2, y=0.2*histos[1][1][0],
                 xerr=(histos[1][1][1][2] - histos[1][1][1][1])/2, fmt=".", label="0.2 * Off Data")
    plt.axvline(x=thetasq, color='black', linestyle='-', label="Cut")
    plt.legend()
    plt.xlabel('$ \Theta^2 \, \mathrm{deg^2} $')
    plt.ylabel('Counts')

    plt.show()
