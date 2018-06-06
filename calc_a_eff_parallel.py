import ROOT
import numpy as np
import multiprocessing as mp
import os
import pandas as pd
from tqdm import tqdm
from read_mars import read_mars
from read_data import calc_onoffhisto

def calc_num_mc_entries(ebins, zdbins, n_chunks, chunk, path):
    MCFiles = ROOT.TChain("OriginalMC")

    ceres_list = list(open(path + "gamma/li_ceresall.txt", "r"))
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


def make_hist(df, bins, weights):
    return np.histogram2d((df["MMcEvtBasic.fTelescopeTheta"].values*360)/(2*np.pi),
                          df["MMcEvtBasic.fEnergy"].values,
                          bins=bins,
                          weights=weights)[0]


def calc_num_mc_entries_hd5(ebins, zdbins, path):
    parts = []
    for entry in tqdm(path):
        df = pd.read_hdf(entry, "table")
        selection = ((df["MMcEvtBasic.fTelescopeTheta"].values * 360) / (2 * np.pi) < 30)
        part1 = df[selection]
        part2 = df[np.bitwise_not(selection)]
        histo = make_hist(part1, [zdbins, ebins], np.ones(len(part1))*4)
        histo += make_hist(part2, [zdbins, ebins], np.ones(len(part2)))
        parts.append(histo)
    return np.sum(np.array(parts), axis=0)


def calc_a_eff_parallel_hd5(ebins,
                            zdbins,
                            correction_factors=True,
                            theta_square_cut=0.085,
                            path="/home/guest/mblank/",
                            list_of_hdf_ceres_files="/home/michi/read_mars/ceres_part",
                            energy_function=None):

    leafs = ["DataType.fVal",
             "MPointingPos.fZd",
             "MMcEvt.MMcEvtBasic.fEnergy",
             "MHillas.fSize",
             "ThetaSquared.fVal",
             "MNewImagePar.fLeakage2",
             'MHillas.fWidth',
             'MHillasSrc.fDist',
             'MHillasExt.fM3Long',
             'MHillasExt.fSlopeLong']

    df = read_mars(path, leaf_names=leafs)
    if correction_factors:
        def energy_function(x):
            return x["MMcEvt.MMcEvtBasic.fEnergy"]

    hist, theta = calc_onoffhisto(df, zdbins, ebins, theta_square_cut, energy_function=energy_function)
    n_surviving = hist[0]

    """
    theta_square_cut = str(theta_square_cut)
    print("\n Calculating the effective area. ---------")

    print("    Loading surviving gammas.")
    Events = ROOT.TChain("Events")
    Events.Add(path)

    surviving = ROOT.TH2D("surviving", "Events Surviving Cuts", len(ebins) - 1, ebins, len(zdbins) - 1, zdbins)

    Events.SetAlias("Zd", "MPointingPos.fZd")
    Events.SetAlias("Mc", "MMcEvt.MMcEvtBasic.fEnergy")
    Events.SetAlias("E", "(pow(29.65*MHillas.fSize,(0.77/cos((MPointingPos.fZd * 1.35 * TMath::Pi())/360))))")

    cut = "(DataType.fVal>0.5)&&(ThetaSquared.fVal<" + theta_square_cut + ")"
    if not correction_factors:
        Events.Draw("Zd:Mc>>surviving", cut, "goff")
    else:
        Events.Draw("Zd:E>>surviving", cut, "goff")

    n_surviving = np.zeros((len(zdbins) - 1, len(ebins) - 1))
    for i in range(len(ebins) - 1):
        for j in range(len(zdbins) - 1):
            n_surviving[j, i] = surviving.GetBinContent(i + 1, j + 1)  # +1, da in ROOT bin 0 der underflow bin ist.
    """

    n_mc = calc_num_mc_entries_hd5(ebins, zdbins, list_of_hdf_ceres_files)

    a_eff = np.divide(n_surviving, n_mc) * (np.pi * (54000.0 * 54000.0))

    print("--------- Returning the effective area.")

    return a_eff


def calc_a_eff_parallel(ebins,
                        zdbins,
                        correction_factors=False,
                        theta_square_cut=0.085,
                        path="/home/guest/mblank/",
                        n_chunks=8):

    """ Reads or calculates the effective area.
    
    Returns the effective area(energy, zenith distance) in cm² with binning specified in the one dimensional
    arrays of the bin-edges ebins and zdbins and saves it to a numpy file with a filename reflecting
    the parameters in "path"+a_eff/.
    
    The value of each bin is calculated from the number of events surviving the analysis chain n and from the
    number of simulated events, as well es the used area (max impact parameter) of the simulation:
    a_eff(E, Zd) = n(E, Zd)/n_sim(E, Zd) * 54000 * 54000 * Pi
    
    It also saves a numpy file of a histogram of the simulated MC events in the specified binning.

    The surviving events are loaded from path+"gamma/hzd_gammasall-analysis.root", while the number of simulated
    events is calculated from the according list of ceres-files in path+"gamma/li_ceresall.txt".
    
    A theta square cut (theta is the angular distance of events to the source position) is applied.    
    
    correction_factors = False is the standard option. It means, that the histogram of surviving events is filled 
    using the true MC energy. If correction factors = False is used, the histogram of surviving events is filled 
    with the estimated energy of the events according to the energy estimation formula, implementing the method of
    correction factors in a simple way.
    
    The n_chunks parameter specifies, into how many chunks the list of the ceres-files for the calculation of
    the histogram of the simulated events is split. The chunks are then processed by a pool of workers via 
    multiprocessing.Pool.apply_async().
    
    :param ebins: ndarray, Array of energy bin edges. 
    :param zdbins: ndarray, Array of zenith distance bin edges.
    :param correction_factors: bool, True or False.(Standard = True. If False, correction factor method is used.)
    :param theta_square_cut: str, Maximum squared angular distance from source. (Standard = 0.085) 
    :param path: str, Base path of the folder, where the arrays are saved to and loaded from.
    :param n_chunks: int, Number of chunks, into which the read-in of simulated events is divided to. (Standard = 8)
    :return: ndarray, histogram of 2D effective area(energy, zenith distance)
    """

    theta_square_cut = str(theta_square_cut)
    print("\n Calculate the effective area. ---------")

    name = path + "a_eff/" + "ebins" + str(len(ebins) - 1) + "_zdbins" + str(
        len(zdbins) - 1) + "emc_" + str(correction_factors) + "theta_sq" + theta_square_cut

    if os.path.isfile(name + ".npy"):
        print("    Numpy file of effective area exists. Loading it.")
        a_eff = np.load(name + ".npy")
        print(np.mean(a_eff))
        print(np.median(a_eff))
        if not (a_eff.shape == (len(zdbins) - 1, len(ebins) - 1)):
            print("    Shape of effective area is wrong, removing", name, "...")
            os.remove(name + ".npy")
            print("    Please run script again.")
        elif (np.isinf(np.mean(a_eff))):
            print("    Median of effective area is Inf, removing",name, "...")
            os.remove(name + ".npy")
            print("    Please run script again.")
        elif (np.isnan(np.median(a_eff))):
            print("    Median of effective area is NaN, removing",name, "...")
            os.remove(name + ".npy")
            print("    Please run script again.")
    else:
        print("    Loading surviving gammas.")
        Events = ROOT.TChain("Events")
        Events.Add(path + "gamma/hzd_gammasall-analysis.root")

        surviving = ROOT.TH2D("surviving", "Events Surviving Cuts", len(ebins) - 1, ebins, len(zdbins) - 1, zdbins)

        Events.SetAlias("Zd", "MPointingPos.fZd")
        Events.SetAlias("Mc", "MMcEvt.MMcEvtBasic.fEnergy")
        Events.SetAlias("E", "(pow(29.65*MHillas.fSize,(0.77/cos((MPointingPos.fZd * 1.35 * TMath::Pi())/360))))")

        cut = "(DataType.fVal>0.5)&&(ThetaSquared.fVal<" + theta_square_cut + ")"
        if not correction_factors:
            Events.Draw("Zd:Mc>>surviving", cut, "goff")
        else:
            Events.Draw("Zd:E>>surviving", cut, "goff")

        n_surviving = np.zeros((len(zdbins) - 1, len(ebins) - 1))
        for i in range(len(ebins) - 1):
            for j in range(len(zdbins) - 1):
                n_surviving[j, i] = surviving.GetBinContent(i + 1, j + 1)  # +1, da in ROOT bin 0 der underflow bin ist.

        mc_name = path + "a_eff/n_mc_histo" + "_e" + str(len(ebins) - 1) + "_zd" + str(
            len(zdbins - 1))
        if os.path.isfile(mc_name + ".npy"):
            print("    Loading existing histogram of simulated Gammas in correct binning:", mc_name+".npy")
            n_mc = np.load(mc_name + ".npy")
            if not (n_mc.shape == (len(zdbins) - 1, len(ebins) - 1)):
                print("    Shape of existing histogram is wrong, removing", mc_name, ".npy ...")
                os.remove(mc_name + ".npy")
                print("    Please run script again.")
            elif (np.isinf(np.mean(n_mc))):
                print("    Median of histogram is Inf, removing", mc_name, ".npy ...")
                os.remove(mc_name + ".npy")
                print("    Please run script again.")
            elif (np.isnan(np.median(n_mc))):
                print("    Median of histogram is NaN, removing", mc_name, ".npy ...")
                os.remove(mc_name + ".npy")
                print("    Please run script again.")
            elif (np.mean(n_mc) == 0):
                print("    Mean of histogram is 0, removing", mc_name, ".npy ...")
                os.remove(mc_name+".npy")
                print("    Please run script again.")

        else:
            print("    Calculating histogram of simulated Gammas from callisto root files.")
            n_mc_parts = np.zeros([n_chunks, len(zdbins) - 1, len(ebins) - 1])

            pool = mp.Pool()

            result = [pool.apply_async(calc_num_mc_entries, args=(ebins, zdbins, n_chunks, i, path))
                      for i in range(n_chunks)]

            pool.close()
            pool.join()
            for r in range(n_chunks):
                n_mc_parts[r] = result[r].get()
            n_mc = np.sum(n_mc_parts, axis=0)
            np.save(mc_name, n_mc)

        a_eff = np.divide(n_surviving, n_mc) * (np.pi * (54000.0 * 54000.0))
        print("    Saving effective area to", name)
        np.save(name, a_eff)

    print("--------- Returning the effective area.")

    return a_eff