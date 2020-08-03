import numpy as np
import pandas as pd
import read_mars
from multiprocessing import Pool


def read_data_and_bg_cut(entry, tree_name, leafs, cut_function=None):
    if cut_function is None:
        def cut_function(x):
            return (np.pi * x['MHillas.fLength'] * x['MHillas.fWidth']) < \
                   (np.log10(x['MHillas.fSize']) * 898 - 1535)

    temp = read_mars.read_mars(entry, tree=tree_name, leaf_names=leafs)

    # Apply Johannes Cut: Area < log10(Size) * 898 -1535
    temp = pd.DataFrame(temp[temp.apply(cut_function, axis=1)], columns=leafs)

    return temp

def calc_time(x, time_name):
    return x[time_name + ".fMjd"] + (x[time_name + ".fTime.fMilliSec"] + x[
                      time_name + ".fNanoSec"] / (1000 * 1000)) / (1000 * 60 * 60 * 24)

def calc_zd_for_star_file(events, star_file):
    select_leaves_drive = ["MTimeDrive.fMjd", "MTimeDrive.fTime.fMilliSec", "MTimeDrive.fNanoSec",
                           "MReportDrive.fNominalZd"]
    drive = read_mars.read_mars(star_file, tree='Drive', leaf_names=select_leaves_drive)
    drive["time"] = calc_time(drive, 'MTimeDrive').values
    event_times = calc_time(events, 'MTime').values
    return drive["MReportDrive.fNominalZd"][
        np.argmin(np.abs(drive["time"].values[:, np.newaxis] - event_times), axis=0)].values

def disp(data):
    mm2deg = (180 / np.pi) / (4.889 * 1000)
    xi = 1.39252 + 0.154247 * (
    data["MHillasExt.fSlopeLong"] * np.sign(data["MHillasSrc.fCosDeltaAlpha"]) / mm2deg) + 1.67972 * (
    1 - (1 / (1 + 4.86232 * data['MNewImagePar.fLeakage1'])))
    blubb = xi * (1 - np.divide(data['MHillas.fWidth'], data['MHillas.fLength']))

    sign1 = (data["MHillasExt.fM3Long"] * (np.sign(data["MHillasSrc.fCosDeltaAlpha"]) * mm2deg)) < -0.07

    sign2 = (data["MHillasExt.fSlopeLong"] * (np.sign(data["MHillasSrc.fCosDeltaAlpha"]) / mm2deg)) > (
    (data["MHillasSrc.fDist"] * mm2deg - 0.5) * 7.2)

    sign = np.logical_or(sign1.values, sign2.values)
    blubb = sign * blubb + np.logical_not(sign) * -blubb
    blubb *= np.sign(data["MHillasSrc.fCosDeltaAlpha"])
    return blubb.values



def read_and_select_data(entry, tree_name, leafs, zdbinsr, ebinsr, thetasqr, efunc, cut_function):
    temp = read_mars.read_mars(entry, tree=tree_name, leaf_names=leafs)
    temp["MPointingPos.fZd"] = calc_zd_for_star_file(temp, entry)
    return calc_onoffhisto(temp, zdbinsr, ebinsr, thetasqr, energy_function=efunc, cut=cut_function)


def histos_from_list_of_mars_files(file_list, leaf_names, zdbins, ebins, thetasq,
                                   efunc=None, cut_function=None, use_multiprocessing=True):
    file_list = [entry.strip().replace(" ", "/") for entry in
                 file_list if not entry.startswith('#')]

    if use_multiprocessing:
        parts = [[]] * len(file_list)
        pool = Pool()
        result = [pool.apply_async(read_and_select_data, args=(file_list[i], 'Events', leaf_names, zdbins,
                                                       ebins, thetasq, efunc, cut_function)) for i in range(len(file_list))]
        pool.close()
        pool.join()

        for i in range(len(file_list)):
            parts[i] = result[i].get()
    else:

        parts = [read_and_select_data(file_list[i], 'Events', leaf_names, zdbins,
                                                       ebins, thetasq, efunc, cut_function) for i in range(len(file_list))]

    pew = np.array([0]), [[np.array[0], np.array[0]], [np.array[0], np.array[0]]]

    for part in parts:
        onoff, theta = part
        pew[0] = pew[0] + onoff
        pew[1][0] = pew[1][0][0] + theta[0][0]
        pew[1][1] = pew[1][1][0] + theta[1][0]
        pew[1][0][1] = theta[0][1]
        pew[1][1][1] = theta[1][1]

    return pew


def calc_onoffhisto(data,
                    zdbins,
                    ebins,
                    thetasq,
                    energy_function=None,
                    slope_goal=None,
                    energy_function2=None,
                    cut=None,
                    is_star_file=False):

    if cut is not None:
        if hasattr(cut, '__call__'):
            data = data.loc[cut(data).values]
        else:
            raise ValueError('Cut is not callable, please check if it is a function of DataFrame columns')

    if energy_function is None:
        def energy_function(x):
            return ((LOG10(x["MHillas.fSize"])+0.7)-3/1.22+3.05
            #return (np.power(29.65 * x["MHillas.fSize"],
           #                  (0.77 / np.cos((x["MPointingPos.fZd"] * 1.35 * np.pi) / 360))) +
           #         x["MNewImagePar.fLeakage2"] * 13000)
            

    data = data.dropna().copy()
    on_histo = np.zeros([len(zdbins)-1, len(ebins)-1])
    off_histo = np.zeros([len(zdbins)-1, len(ebins)-1])

    if data["ThetaSquared.fVal"].min() < thetasq:
        try:
            if is_star_file:
                pass
            else:
                source_data = data.loc[np.less_equal(data["ThetaSquared.fVal"].values, thetasq)].copy()
                select = source_data["DataType.fVal"].values.astype(np.bool)


            source_data["energy"] = energy_function(source_data)
            if energy_function2 is not None:
                source_data["energy2"] = energy_function2(source_data)
            on_data = source_data.loc[select]
            off_data = source_data.loc[np.bitwise_not(select)]

            if slope_goal is None:
                weights_on = np.ones(len(on_data))
                weights_off = np.ones(len(off_data))
            else:
                m = 2.7
                factor = 50000 ** (-m + slope_goal)
                exponent = -slope_goal + m
                weights_on = np.power(on_data["MMcEvt.MMcEvtBasic.fEnergy"].values, exponent) * factor
                weights_off = np.power(off_data["MMcEvt.MMcEvtBasic.fEnergy"].values, exponent) * factor

            on_histo = np.histogram2d(on_data["MPointingPos.fZd"].values,
                                      on_data["energy"].values,
                                      bins=[zdbins, ebins],
                                      weights=weights_on)[0]
            off_histo = np.histogram2d(off_data["MPointingPos.fZd"].values,
                                       off_data["energy"].values,
                                       bins=[zdbins, ebins],
                                       weights=weights_off)[0]

            if energy_function2 is not None:
                energy_migration = np.histogramdd((on_data["MPointingPos.fZd"].values,
                                                   on_data["energy2"].values,
                                                   on_data["energy"].values),
                                                  bins=(zdbins, ebins, ebins))

            datr = data[["DataType.fVal", "ThetaSquared.fVal"]]
            select = datr["DataType.fVal"].values.astype(np.bool)
            on_thetasq = np.histogram(datr.loc[select]["ThetaSquared.fVal"].values,
                                      bins=40,
                                      range=(0.0, 0.3))
            off_thetasq = np.histogram(datr.loc[np.bitwise_not(select)]["ThetaSquared.fVal"].values,
                                       bins=40,
                                       range=(0.0, 0.3))

        except:
            on_histo = np.zeros([len(zdbins)-1, len(ebins)-1])
            off_histo = np.zeros([len(zdbins)-1, len(ebins)-1])

            datr = data[["DataType.fVal", "ThetaSquared.fVal"]]
            select = datr["DataType.fVal"].values.astype(np.bool)
            on_thetasq = np.histogram(datr.loc[select]["ThetaSquared.fVal"].values,
                                      bins=40,
                                      range=(0.0, 0.3))
            off_thetasq = np.histogram(datr.loc[np.bitwise_not(select)]["ThetaSquared.fVal"].values,
                                       bins=40,
                                       range=(0.0, 0.3))
            if energy_function2 is not None:
                raise ValueError('Calculation of histograms has failed.')

    if energy_function2 is not None:
        return np.array([on_histo, off_histo]), [on_thetasq, off_thetasq], energy_migration[0]
    return np.array([on_histo, off_histo]), [on_thetasq, off_thetasq]
