import numpy as np
import pandas as pd
import read_mars
from multiprocessing import Pool


def read_data_and_bg_cut(entry, tree_name, leafs):
    temp = read_mars.read_mars(entry, tree=tree_name, leaf_names=leafs)
    # Apply Johannes Cut: Area < log10(Size) * 898 -1535
    temp = pd.DataFrame(temp[temp.apply(
        lambda x: (np.pi * x['MHillas.fLength'] * x['MHillas.fWidth']) < (np.log10(x['MHillas.fSize']) * 898 - 1535),
        axis=1)], columns=leafs)

    return temp


def read_and_select_data(entry, tree_name, leafs, zdbinsr, zdlabelsr, ebinsr, elabelsr, thetasqr):
    temp = read_data_and_bg_cut(entry, tree_name, leafs)

    return calc_onoffhisto(temp, zdbinsr, zdlabelsr, ebinsr, elabelsr, thetasqr)


def histos_from_list_of_mars_files(file_list, leaf_names, zdbins, zdlabels, ebins, elabels, thetasq):
    file_list = [entry.strip().replace(" ", "/").replace("star", "ganymed_run").replace("_I", "-summary") for entry in
                 file_list if not entry.startswith('#')]
    leaf_names.append('MHillas.fLength')
    leaf_names.append('MHillas.fWidth')
    histos = np.zeros([2, len(zdlabels), len(elabels)])
    pool = Pool()
    result = [pool.apply_async(read_and_select_data, args=(file_list[i], 'Events', leaf_names, zdbins, zdlabels,
                                                           ebins, elabels, thetasq)) for i in range(len(file_list))]
    pool.close()
    pool.join()
    for res in result:
        histos += res.get()

    return np.array(histos)


def calc_onoffhisto(data, zdbins, zdlabels, ebins, elabels, thetasq):
    data = data.assign(energy=lambda x: (np.power(29.65 * x["MHillas.fSize"],
                                                  (0.77 / np.cos((x["MPointingPos.fZd"] * 1.35 * np.pi) / 360))) +
                                         x["MNewImagePar.fLeakage2"] * 13000))
    on_histo = np.zeros([len(zdlabels), len(elabels)])
    off_histo = np.zeros([len(zdlabels), len(elabels)])

    data['Zdbin'] = pd.cut(data["MPointingPos.fZd"], zdbins, labels=zdlabels, include_lowest=True)
    data['Ebin'] = pd.cut(data["energy"], ebins, labels=elabels, include_lowest=True)
    if data["ThetaSquared.fVal"].min() < thetasq:
        try:
            data['theta'] = pd.cut(data["ThetaSquared.fVal"], [0, thetasq, 10], labels=["source", "notsource"],
                                   include_lowest=True)

            source_data = data.groupby('theta').get_group("source")
            on_data = source_data.groupby("DataType.fVal").get_group(1.0)
            off_data = source_data.groupby("DataType.fVal").get_group(0.0)

            on_histo = np.histogram2d(on_data["MPointingPos.fZd"], on_data["energy"], bins=[zdbins, ebins])[0]
            off_histo = np.histogram2d(off_data["MPointingPos.fZd"], off_data["energy"], bins=[zdbins, ebins])[0]
            on_thetasq = np.histogram(data.groupby("DataType.fVal").get_group(1.0)["ThetaSquared.fVal"],
                                      bins=40, range=(0.0, 0.3))
            off_thetasq = np.histogram(data.groupby("DataType.fVal").get_group(0.0)["ThetaSquared.fVal"],
                                       bins=40, range=(0.0, 0.3))
        except:
            on_histo = np.zeros([len(zdlabels), len(elabels)])
            off_histo = np.zeros([len(zdlabels), len(elabels)])
    return np.array([on_histo, off_histo]), [on_thetasq, off_thetasq]
