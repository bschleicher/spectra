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


def read_and_select_data(entry, tree_name, leafs, zdbinsr, ebinsr, thetasqr):
    temp = read_data_and_bg_cut(entry, tree_name, leafs)

    return calc_onoffhisto(temp, zdbinsr, ebinsr, thetasqr)


def histos_from_list_of_mars_files(file_list, leaf_names, zdbins, ebins, thetasq):
    file_list = [entry.strip().replace(" ", "/").replace("star", "ganymed_run").replace("_I", "-summary") for entry in
                 file_list if not entry.startswith('#')]
    leaf_names.append('MHillas.fLength')
    leaf_names.append('MHillas.fWidth')
    histos = np.zeros([2, len(zdbins)-1, len(ebins)-1])
    pool = Pool()
    result = [pool.apply_async(read_and_select_data, args=(file_list[i], 'Events', leaf_names, zdbins,
                                                           ebins, thetasq)) for i in range(len(file_list))]
    pool.close()
    pool.join()
    for res in result:
        histos += res.get()

    return np.array(histos)


def calc_onoffhisto(data,
                    zdbins,
                    ebins,
                    thetasq,
                    energy_function=None,
                    slope_goal=None,
                    energy_function2=None,
                    cut=None):

    if cut is not None:
        if hasattr(cut, '__call__'):
            data = data.loc[cut(data).values].copy()
        else:
            raise ValueError('Cut is not callable, please check if it is a function of DataFrame columns')

    if energy_function is None:
        def energy_function(x):
            return (np.power(29.65 * x["MHillas.fSize"],
                             (0.77 / np.cos((x["MPointingPos.fZd"] * 1.35 * np.pi) / 360))) +
                    x["MNewImagePar.fLeakage2"] * 13000)
    data = data.dropna().copy()
    data = data.assign(energy=energy_function)
    if energy_function2 is not None:
        data = data.assign(energy2=energy_function2)

    on_histo = np.zeros([len(zdbins)-1, len(ebins)-1])
    off_histo = np.zeros([len(zdbins)-1, len(ebins)-1])

    if data["ThetaSquared.fVal"].min() < thetasq:
        try:

            source_data = data.loc[np.less_equal(data["ThetaSquared.fVal"].values, 0.07)]
            select = source_data["DataType.fVal"].values.astype(np.bool)
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
        return np.array([on_histo, off_histo]), [on_thetasq, off_thetasq], energy_migration[0]
    return np.array([on_histo, off_histo]), [on_thetasq, off_thetasq]
