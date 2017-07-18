import read_mars
import pandas as pd
import numpy as np
import multiprocessing as mp


def calc_on_time_chunks(zdbins, zdlabels, filelist):
    select_leaves_rates = ["MTimeRates.fMjd", "MTimeRates.fTime.fMilliSec", "MTimeRates.fNanoSec",
                           'MReportRates.fElapsedOnTime']
    select_leaves_drive = ["MTimeDrive.fMjd", "MTimeDrive.fTime.fMilliSec", "MTimeDrive.fNanoSec",
                           "MReportDrive.fNominalZd"]

    rates_chunk = pd.DataFrame(columns=select_leaves_rates)
    drive_chunk = pd.DataFrame(columns=select_leaves_drive)

    for entry in filelist:
        drive_part = read_mars.read_mars(entry, tree="Drive", leaf_names=select_leaves_drive)
        rates_part = read_mars.read_mars(entry, tree="Rates", leaf_names=select_leaves_rates)
        rates_part = rates_part.assign(time=lambda x: x["MTimeRates.fMjd"] + (x["MTimeRates.fTime.fMilliSec"] + x[
            "MTimeRates.fNanoSec"] / (1000 * 1000)) / (1000 * 60 * 60 * 24))
        drive_part = drive_part.assign(time=lambda x: x["MTimeDrive.fMjd"] + (x["MTimeDrive.fTime.fMilliSec"] + x[
            "MTimeDrive.fNanoSec"] / (1000 * 1000)) / (1000 * 60 * 60 * 24))
        drive_part['Zdbin'] = pd.cut(drive_part["MReportDrive.fNominalZd"], zdbins, labels=zdlabels,
                                     include_lowest=True)
        rates_part["Zdbin"] = drive_part["Zdbin"][
            np.argmin(np.abs(drive_part["time"].values[:, np.newaxis] - rates_part['time'].values), axis=0)].values
        rates_chunk = pd.concat([rates_chunk, rates_part.sort_values("time")], axis=0)
        drive_chunk = pd.concat([drive_chunk, drive_part.sort_values("time")], axis=0)

    adj_check = (rates_chunk["Zdbin"] != rates_chunk["Zdbin"].shift()).cumsum()

    timeranges = rates_chunk.groupby([adj_check], as_index=False, sort=False).agg(
        {'Zdbin': ['min'], "MReportRates.fElapsedOnTime": ['sum']}).values

    on_time_per_zd_chunk = np.zeros((len(zdlabels),))

    for (zd, on_times) in timeranges:
        on_time_per_zd_chunk[int(zd)] += on_times

    return on_time_per_zd_chunk


def calc_on_time(ganymed_input_list, zdbins, zdlabels, n_chunks=8):

    ganymed_input_list = [entry.strip().replace(" ", "/") for entry in ganymed_input_list if not entry.startswith('#')]
    parts = (len(ganymed_input_list) / n_chunks * np.arange(n_chunks + 1)).astype("int_")

    on_time_parts = np.empty([n_chunks, len(zdbins) - 1])

    pool = mp.Pool()
    result = [pool.apply_async(calc_on_time_chunks, args=(zdbins, zdlabels, ganymed_input_list[parts[i]:parts[i + 1]]))
              for i in range(n_chunks)]
    pool.close()
    pool.join()
    for r in range(n_chunks):
        on_time_parts[r] = result[r].get()

    return np.sum(on_time_parts, axis=0)
