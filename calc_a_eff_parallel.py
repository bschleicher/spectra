import ROOT
import numpy as np
import multiprocessing as mp
import os


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


def calc_a_eff_parallel(ebins, zdbins, use_mc=True, theta_square_cut="0.085", path="home/mblank/", n_chunks=8):
    name = path + "a_eff/" + "ebins" + str(len(ebins) - 1) + "_zdbins" + str(
        len(zdbins) - 1) + "emc_" + str(use_mc) + "theta_sq" + theta_square_cut

    if os.path.isfile(name + ".npy"):
        a_eff = np.load(name + ".npy")
        if not (a_eff.shape == (len(zdbins) - 1, len(ebins) - 1)):
            print("Shape of effective area is wrong, delete the old file:", name)
    else:
        Events = ROOT.TChain("Events")
        Events.Add(path + "gamma/hzd_gammasall-analysis.root")

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

        mc_name = path + "n_mc_histo" + "_e" + str(len(ebins) - 1) + "_zd" + str(
            len(zdbins - 1))
        if os.path.isfile(mc_name + ".npy"):
            n_mc = np.load(mc_name + ".npy")
        else:
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

        np.save(name, a_eff)

    return a_eff