import on_time_parallel
import calc_a_eff_parallel

import pandas as pd
from astropy import time as at
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import numpy as np
from fact.analysis.statistics import li_ma_significance
import multiprocessing as mp
import read_mars


def read_list_of_mars_files(file_list, leaf_names):
    n_files = len(file_list)
    file_list = [entry.strip().replace(" ", "/").replace("star", "ganymed_run").replace("_I", "-analysis") for entry
                 in file_list if not entry.startswith('#')]

    df = pd.DataFrame(columns=leaf_names)

    pool = mp.Pool()
    result = [pool.apply_async(read_mars.read_mars, args=(file_list[i], 'Events', leaf_names)) for i in range(n_files)]
    pool.close()
    pool.join()
    for r in range(n_files):
        df = pd.concat([df, result[r]], axis=0)

    return df


if __name__ == '__main__':

    # Setup ThetaSqare Cut.
    # Zenith Distance Bins, MC are available from 0 to 60 deg.
    # Energy Bins, MC are av. from 0.2 to 50 TeV
    # If False, effective area is calculated with estimated energy and not MC energy.

    thetasq = 0.07
    zdbins = np.linspace(0, 60, 15)
    ebins = np.logspace(np.log10(200.0), np.log10(50000.0), 11)
    use_mc = False
    star_list = list(open("/media/michi/523E69793E69574F/daten/crab2.txt", "r"))
    # On ISDC, put None, to read automatically processed Ganymed Output from star files:
    ganymed_result = "/media/michi/523E69793E69574F/daten/crab2-analysis.root"
    base_path = "/media/michi/523E69793E69574F/daten/"

    zdlabels = range(len(zdbins) - 1)

    # Calcualtion of on time from ganymed input list of star files:

    on_time_per_zd = np.zeros([len(zdbins)-1])
    on_time_per_zd = on_time_parallel.calc_on_time(star_list, zdbins, zdlabels)

    # Read the required leaves of the ganymed-analysis output file and calculate energy estimation:

    select_leaves = ['DataType.fVal', 'MPointingPos.fZd', 'FileId.fVal', 'MTime.fMjd', 'MTime.fTime.fMilliSec',
                     'MTime.fNanoSec', 'MHillas.fSize', 'ThetaSquared.fVal', 'MNewImagePar.fLeakage2']
    if not ganymed_result:
        data_cut = read_list_of_mars_files(star_list, leaf_names=select_leaves)
    else:
        data_cut = read_mars.read_mars(ganymed_result, leaf_names=select_leaves)

    data_cut = data_cut.assign(energy=lambda x: (
        np.power(29.65 * x["MHillas.fSize"], (0.77 / np.cos((x["MPointingPos.fZd"] * 1.35 * np.pi) / 360))) + x[
            "MNewImagePar.fLeakage2"] * 13000))

    # Bin the data in ZD, Energy and theta:

    data_cut['Zdbin'] = pd.cut(data_cut["MPointingPos.fZd"], zdbins, labels=zdlabels, include_lowest=True)

    elabels = range(len(ebins) - 1)

    data_cut['Ebin'] = pd.cut(data_cut["energy"], ebins, labels=elabels, include_lowest=True)
    data_cut['theta'] = pd.cut(data_cut["ThetaSquared.fVal"], [0, thetasq, 10], labels=[0, 1], include_lowest=True)

    # Group by theta to split in on and off data and create histograms for on, off and excess events:

    theta_data = data_cut.groupby('theta').get_group(0)
    on_data = theta_data.groupby('DataType.fVal').get_group(1.0)
    off_data = theta_data.groupby('DataType.fVal').get_group(0.0)

    on_histo = np.histogram2d(on_data["MPointingPos.fZd"], on_data["energy"], bins=[zdbins, ebins])[0]
    off_histo = np.histogram2d(off_data["MPointingPos.fZd"], off_data["energy"], bins=[zdbins, ebins])[0]

    exc_histo = on_histo - (1 / 5) * off_histo
    exc_histo_err = np.sqrt(on_histo + (1 / 25) * off_histo)

    overall_significance = li_ma_significance(np.sum(on_histo), np.sum(off_histo))

    # Calculate the effective area:

    a_eff = calc_a_eff_parallel.calc_a_eff_parallel(ebins, zdbins, use_mc=use_mc, theta_square_cut=str(thetasq),
                                                    path=base_path)

    print(on_time_per_zd)
    print("On-Time:", np.sum(on_time_per_zd), "s")
    print("On-Time:", np.sum(on_time_per_zd) / (60 * 60), "h")

    # Calculate an effective effective area, that scales the effective area per on time
    a_eff = (a_eff * on_time_per_zd[:, np.newaxis]) / np.sum(on_time_per_zd)
    flux = np.divide(np.sum(exc_histo, axis=0), np.sum(a_eff, axis=0))
    flux = np.divide(flux, (np.sum(on_time_per_zd)))
    flux_err = np.ma.divide(np.sqrt(np.sum(on_histo, axis=0)+ (1/25) * np.sum(off_histo, axis=0)),
                              np.sum(a_eff, axis=0)) / np.sum(on_time_per_zd)
    sig = li_ma_significance(on_histo, off_histo)
    # flux_e = np.ma.average(flux2d, axis=0, weights=sig)
    #flux_e_err = np.sqrt(np.ma.average((flux2d_err ** 2), axis=0, weights=sig))

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

    flux_de = (flux / (bin_width / 1000))
    flux_de_err = ((flux_err / (bin_width / 1000)) / (flux_de * np.log(10)))
    flux_de_l = flux_de - np.power(10, (np.log10(flux_de) - flux_de_err))
    flux_de_h = np.power(10, (np.log10(flux_de) + flux_de_err)) - flux_de

    ax1 = plt.subplot(121)

    plt.errorbar(x=bin_centers, y=flux_de, yerr=[flux_de_l, flux_de_h], fmt="o", label="FACT")
    # plt.errorbar(x=hess_x * 1000, y=hess_y, yerr=[hess_yl, hess_yh], fmt=".")
    plt.errorbar(x=crab_do_x, y=crab_do_y, yerr=crab_do_y_err, fmt="o", label="Crab Dortmund")
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
    plt.errorbar(x=bin_centers, y=li_ma_significance(np.sum(exc_histo, axis=0), np.sum(exc_histo_err, axis=0)), fmt="o",
                 label="LiMa Significance")
    plt.xscale("log")
    plt.grid(True)
    plt.xlabel("Energy [GeV]")
    plt.ylabel(" $\mathrm{\sigma}$")
    plt.legend()

    ZD_min = data_cut["MPointingPos.fZd"].min()
    ZD_max = data_cut["MPointingPos.fZd"].max()

    fig = plt.figure()

    gs = gspec.GridSpec(2,4, height_ratios=[1,3])
    ax4 = fig.add_subplot(gs[0,:])
    ax4.text(0.0, 0.0, "On Time: " + str(np.sum(on_time_per_zd) / (60 * 60)) + " h\n" + "$\mathrm{\sigma_{LiMa}}$: "
             + str(overall_significance) + "\n" + "ZD min: " + str(ZD_min) + " $\deg$\n"
             + "ZD max: " + str(ZD_max) + " $\deg$")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1,0])
    ax5.imshow(a_eff)

    ax6 = fig.add_subplot(gs[1,1])
    ax6.imshow(exc_histo)

    ax7 = fig.add_subplot(gs[1,2])
    ax7.imshow(on_histo)

    ax8 = fig.add_subplot(gs[1,3])
    ax8.imshow(off_histo)

    plt.show()
