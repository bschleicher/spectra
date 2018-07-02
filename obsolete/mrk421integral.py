import ROOT
import pandas as pd
import os
from astropy import time as at
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from fact.analysis.statistics import li_ma_significance
import multiprocessing as mp

result = ROOT.gSystem.Load('libmars.so')
if result != 0:
    raise ImportError('Could not load libmars, Make sure to set your "LD_LIBRARY_PATH"')


def read_mars_fast(filename, tree='Events', leaves=[]):
    """Return a Pandas DataFrame of a star or ganymed output root file.

    A faster (~factor 15) version of read_mars. In addition, the useless leaves fBits and fUniqueID are omitted. 
    Keyword arguments:
    tree -- Set, which tree to read. (Default = "Events")
    leaves -- Specify a list of the read in leaves. (Default is [] what reads in all leaves)
    """

    f = ROOT.TFile(filename)
    tree = f.Get(tree)
    if not leaves:
        leaves = [l.GetName() for l in tree.GetListOfLeaves() if
                  not (l.GetName().endswith('.') or l.GetName().endswith('fBits') or l.GetName().endswith('fUniqueID'))]

    n_events = tree.GetEntries()
    tree.SetEstimate(n_events + 1)
    df = pd.DataFrame(np.empty([n_events, len(leaves)]), columns=leaves)
    b = np.empty([n_events])

    for leaf in leaves:
        tree.Draw(leaf, "", "goff")
        v1 = tree.GetV1()
        v1.SetSize(n_events + 1)
        for i in range(n_events):
            b[i] = v1[i]
        df[leaf] = b
    f.Close()

    return df



def calc_a_eff(ebins, zdbins):
    MCFiles = ROOT.TChain("OriginalMC")
    Events = ROOT.TChain("Events")

    Events.Add("/media/michi/523E69793E69574F/gamma/star_new/hzd_gammasall-analysis.root")

    ceres_list = list(open("/media/michi/523E69793E69574F/gamma/ceres/li_ceresall.txt", "r"))
    for entry in ceres_list:
        MCFiles.Add(entry[:-1])

    simulated = ROOT.TH2D("simulated", "Original MC", len(ebins) - 1, ebins, len(zdbins) - 1, zdbins)
    surviving = ROOT.TH2D("surviving", "Events Surviving Cuts", len(ebins) - 1, ebins, len(zdbins) - 1, zdbins)

    MCFiles.SetAlias("Zd", "(MMcEvtBasic.fTelescopeTheta*360)/(2*TMath::Pi())")
    Events.SetAlias("Zd", "MPointingPos.fZd")
    MCFiles.SetAlias("Mc", "MMcEvtBasic.fEnergy")
    Events.SetAlias("Mc", "MMcEvt.MMcEvtBasic.fEnergy")
    Events.SetAlias("E", "(pow(29.65*MHillas.fSize,(0.77/cos((MPointingPos.fZd * 1.35 * TMath::Pi())/360))))")

    MCFiles.Draw("Zd:Mc>>simulated", "(Zd<30)*4+(Zd>=30)", "goff")
    Events.Draw("Zd:Mc>>surviving", "(DataType.fVal>0.5)&&(ThetaSquared.fVal<0.085)", "goff")

    surviving.Divide(simulated)
    # surviving.Scale(np.pi * (54000.0 * 54000.0))

    a_eff = np.zeros((len(zdbins) - 1, len(ebins) - 1))
    for i in range(len(ebins) - 1):
        for j in range(len(zdbins) - 1):
            a_eff[j, i] = surviving.GetBinContent(i + 1, j + 1)  # +1, da in ROOT bin 0 der underflow bin ist.
    np.save("/media/michi/523E69793E69574F/daten/a_eff", a_eff)
    return a_eff


def calc_num_mc_entries(ebins, zdbins, n_chunks, chunk):
    MCFiles = ROOT.TChain("OriginalMC")

    ceres_list = list(open("/media/michi/523E69793E69574F/gamma/ceres/li_ceresall.txt", "r"))
    parts = (len(ceres_list) / n_chunks * np.arange(n_chunks + 1)).astype("int_")
    list_part = ceres_list[parts[chunk]:parts[chunk + 1]]

    for entry in list_part:
        MCFiles.Add(entry[:-1])

    simulated = ROOT.TH2D("simulated", "Original MC", len(ebins) - 1, ebins, len(zdbins) - 1, zdbins)

    MCFiles.SetAlias("Zd", "(MMcEvtBasic.fTelescopeTheta*360)/(2*TMath::Pi())")
    MCFiles.SetAlias("Mc", "MMcEvtBasic.fEnergy")

    MCFiles.Draw("Zd:Mc>>simulated", "(Zd<30)*4+(Zd>=30)", "goff")

    n_mc_part = np.zeros((len(zdbins) - 1, len(ebins) - 1))
    for i in range(len(ebins) - 1):
        for j in range(len(zdbins) - 1):
            n_mc_part[j, i] = simulated.GetBinContent(i + 1, j + 1)  # +1, da in ROOT bin 0 der underflow bin ist.

    return n_mc_part


def calc_a_eff_parallel(ebins, zdbins, use_mc=True, theta_square_cut="0.085"):
    name = "/media/michi/523E69793E69574F/daten/a_eff/" + "ebins" + str(len(ebins) - 1) + "_zdbins" + str(
        len(zdbins) - 1) + "emc_" + str(use_mc) + "theta_sq" + theta_square_cut

    if os.path.isfile(name + ".npy"):
        a_eff = np.load(name + ".npy")
        if not (a_eff.shape == (len(zdbins) - 1, len(ebins) - 1)):
            print("Shape of effective area is wrong, delete the old file:", name)
    else:
        Events = ROOT.TChain("Events")
        Events.Add("/media/michi/523E69793E69574F/gamma/star_new/hzd_gammasall-analysis.root")

        surviving = ROOT.TH2D("surviving", "Events Surviving Cuts", len(ebins) - 1, ebins, len(zdbins) - 1, zdbins)

        Events.SetAlias("Zd", "MPointingPos.fZd")
        Events.SetAlias("Mc", "MMcEvt.MMcEvtBasic.fEnergy")
        Events.SetAlias("E", "(pow(29.65*MHillas.fSize,(0.77/cos((MPointingPos.fZd * 1.35 * TMath::Pi())/360))))")

        cut = "(DataType.fVal>0.5)&&(ThetaSquared.fVal<" + theta_square_cut + ")"
        if use_mc:
            Events.Draw("Zd:Mc>>surviving", cut, "goff")
        else:
            Events.Draw("Zd:E>>surviving", cut, "goff")

        n_surviving = np.zeros((len(zdbins) - 1, len(ebins) - 1))
        for i in range(len(ebins) - 1):
            for j in range(len(zdbins) - 1):
                n_surviving[j, i] = surviving.GetBinContent(i + 1, j + 1)  # +1, da in ROOT bin 0 der underflow bin ist.

        mc_name = "/media/michi/523E69793E69574F/daten/a_eff/n_mc_histo" + "_e" + str(len(ebins) - 1) + "_zd" + str(
            len(zdbins - 1))
        if os.path.isfile(mc_name + ".npy"):
            n_mc = np.load(mc_name + ".npy")
        else:
            n_chunks = 8
            n_mc_parts = np.zeros([n_chunks, len(zdbins) - 1, len(ebins) - 1])

            pool = mp.Pool()

            result = [pool.apply_async(calc_num_mc_entries, args=(ebins, zdbins, n_chunks, i)) for i in range(n_chunks)]

            pool.close()
            pool.join()
            for r in range(n_chunks):
                n_mc_parts[r] = result[r].get()
            n_mc = np.sum(n_mc_parts, axis=0)
            np.save(mc_name, n_mc)

        a_eff = np.divide(n_surviving, n_mc) * (np.pi * (54000.0 * 54000.0))

        np.save(name, a_eff)

    return a_eff


def calc_on_time_chunks(zdbins, zdlabels, filelist):
    select_leaves_rates = ["MTimeRates.fMjd", "MTimeRates.fTime.fMilliSec", "MTimeRates.fNanoSec",
                           'MReportRates.fElapsedOnTime']
    select_leaves_drive = ["MTimeDrive.fMjd", "MTimeDrive.fTime.fMilliSec", "MTimeDrive.fNanoSec",
                           "MReportDrive.fNominalZd"]

    rates = pd.DataFrame(columns=select_leaves_rates)
    drive = pd.DataFrame(columns=select_leaves_drive)

    for entry in filelist:
        print(entry.strip().replace(" ", ""))
        drive_part = read_mars_fast(entry.strip().replace(" ", ""), tree="Drive", leaves=select_leaves_drive)
        rates_part = read_mars_fast(entry.strip().replace(" ", ""), tree="Rates", leaves=select_leaves_rates)
        rates_part = rates_part.assign(time=lambda x: x["MTimeRates.fMjd"] + (x["MTimeRates.fTime.fMilliSec"] + x[
            "MTimeRates.fNanoSec"] / (1000 * 1000)) / (1000 * 60 * 60 * 24))
        drive_part = drive_part.assign(time=lambda x: x["MTimeDrive.fMjd"] + (x["MTimeDrive.fTime.fMilliSec"] + x[
            "MTimeDrive.fNanoSec"] / (1000 * 1000)) / (1000 * 60 * 60 * 24))
        drive_part['Zdbin'] = pd.cut(drive_part["MReportDrive.fNominalZd"], zdbins, labels=zdlabels,
                                     include_lowest=True)
        rates_part["Zdbin"] = drive_part["Zdbin"][
            np.argmin(np.abs(drive_part["time"].values[:, np.newaxis] - rates_part['time'].values), axis=0)].values
        rates = pd.concat([rates, rates_part.sort_values("time")], axis=0)
        drive = pd.concat([drive, drive_part.sort_values("time")], axis=0)

    adj_check = (rates["Zdbin"] != rates["Zdbin"].shift()).cumsum()

    timeranges = rates.groupby([adj_check], as_index=False, sort=False).agg(
        {'Zdbin': ['min'], "MReportRates.fElapsedOnTime": ['sum']}).values

    on_time_per_zd = np.zeros((len(zdlabels),))

    for (zd, on_times) in timeranges:
        on_time_per_zd[int(zd)] += on_times

    return on_time_per_zd


if __name__ == '__main__':

    # Setup ThetaSqare Cut.
    # Zenith Distance Bins, MC are available from 0 to 60 deg.
    # Energy Bins, MC are av. from 0.2 to 50 TeV
    # If False, effective area is calculated with estimated energy and not MC energy.


    thetasq = 0.07
    zdbins = np.linspace(0, 60, 11)
    ebins = np.logspace(np.log10(200.0), np.log10(50000.0), 11)
    use_mc = True
    ganymed_result = "/media/michi/523E69793E69574F/daten/hzd_421_flare_integral_ed-analysis.root"
    star_list = list(open("/media/michi/523E69793E69574F/daten/hzd_421_flare_integral_ed.txt", "r"))
    #ganymed_all_events = "/media/michi/523E69793E69574F/daten/crab2-summary.root"

    # Read in the rates for the on-time calculations:
    rates = read_mars_fast(ganymed_result, tree='Rates')
    rates = rates.assign(time=lambda x: x["MTimeRates.fMjd"] + (x["MTimeRates.fTime.fMilliSec"] + x[
        "MTimeRates.fNanoSec"] / (1000 * 1000)) / (1000 * 60 * 60 * 24))

    # Calcualtion of on time:
    zdlabels = range(len(zdbins) - 1)



    n_chunks = 8
    parts = (len(star_list) / n_chunks * np.arange(n_chunks + 1)).astype("int_")

    on_time_parts = np.empty([n_chunks, len(zdbins) - 1])

    pool = mp.Pool()
    result = [pool.apply_async(calc_on_time_chunks, args=(zdbins, zdlabels, star_list[parts[i]:parts[i + 1]])) for i in
              range(n_chunks)]
    pool.close()
    pool.join()
    for r in range(n_chunks):
        on_time_parts[r] = result[r].get()

    on_time_per_zd = np.sum(on_time_parts, axis=0)

    ##


    select_leaves = ['DataType.fVal', 'MPointingPos.fZd', 'FileId.fVal', 'MTime.fMjd', 'MTime.fTime.fMilliSec',
                     'MTime.fNanoSec', 'MHillas.fSize', 'ThetaSquared.fVal', 'MNewImagePar.fLeakage2']
    data_cut = read_mars_fast(ganymed_result, leaves=select_leaves)
    data_cut = data_cut.assign(energy=lambda x: (
        np.power(29.65 * x["MHillas.fSize"], (0.77 / np.cos((x["MPointingPos.fZd"] * 1.35 * np.pi) / 360))) + x[
            "MNewImagePar.fLeakage2"] * 13000))

    data_cut['Zdbin'] = pd.cut(data_cut["MPointingPos.fZd"], zdbins, labels=zdlabels, include_lowest=True)

    elabels = range(len(ebins) - 1)

    data_cut['Ebin'] = pd.cut(data_cut["energy"], ebins, labels=elabels, include_lowest=True)
    data_cut['theta'] = pd.cut(data_cut["ThetaSquared.fVal"], [0, thetasq, 10], labels=[0, 1], include_lowest=True)

    theta_data = data_cut.groupby('theta').get_group(0)
    on_data = theta_data.groupby('DataType.fVal').get_group(1.0)
    off_data = theta_data.groupby('DataType.fVal').get_group(0.0)

    on_histo = np.histogram2d(on_data["MPointingPos.fZd"], on_data["energy"], bins=[zdbins, ebins])[0]
    off_histo = np.histogram2d(off_data["MPointingPos.fZd"], off_data["energy"], bins=[zdbins, ebins])[0]

    exc_histo = on_histo - (1 / 5) * off_histo
    exc_histo_err = np.sqrt(on_histo + (1 / 25) * off_histo)

    a_eff = calc_a_eff_parallel(ebins, zdbins, use_mc=use_mc, theta_square_cut=str(thetasq))
    # if os.path.isfile("/media/michi/523E69793E69574F/daten/a_eff.npy"):
    #    a_eff = np.load("/media/michi/523E69793E69574F/daten/a_eff.npy")
    #    if not (a_eff.shape == (len(zdbins)-1,len(ebins)-1)):
    #        a_eff = calc_a_eff(ebins, zdbins)
    # else:
    #    a_eff = calc_a_eff(ebins, zdbins)

    print(on_time_per_zd)
    print("On-Time:", np.sum(on_time_per_zd), "s")
    print("On-Time:", np.sum(on_time_per_zd) / (60 * 60), "h")
    flux2d = np.ma.divide(exc_histo, a_eff) / (on_time_per_zd)[:, np.newaxis]
    flux2d_err = np.ma.divide(exc_histo_err, a_eff) / (on_time_per_zd)[:, np.newaxis]
    sig = li_ma_significance(on_histo, off_histo)
    flux_e = np.ma.average(flux2d, axis=0, weights=sig)
    flux_e_err = np.sqrt(np.ma.average((flux2d_err ** 2), axis=0, weights=sig))

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

    flux_de = (flux_e / (bin_width / 1000))
    flux_de_err = ((flux_e_err / (bin_width / 1000)) / (flux_de * np.log(10)))
    flux_de_l = flux_de - np.power(10, (np.log10(flux_de) - flux_de_err))
    flux_de_h = np.power(10, (np.log10(flux_de) + flux_de_err)) - flux_de



    ax1 = plt.subplot(121)
    plt.title("MRK421 Flare Integral")
    plt.errorbar(x=bin_centers, y=flux_de, yerr=[flux_de_l, flux_de_h], fmt="o", label="FACT")
    #plt.errorbar(x=hess_x * 1000, y=hess_y, yerr=[hess_yl, hess_yh], fmt=".", label="MRK501 HESS 24.06.2014")
    #plt.errorbar(x=crab_do_x, y=crab_do_y, yerr=crab_do_y_err, fmt="o", label="Crab Dortmund")
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

    plt.show()