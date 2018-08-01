import read_mars
import pandas as pd
import numpy as np
import multiprocessing as mp


def calc_on_time_chunks(zdbins, filelist):
    select_leaves_rates = ["MTimeRates.fMjd", "MTimeRates.fTime.fMilliSec", "MTimeRates.fNanoSec",
                           'MReportRates.fElapsedOnTime']
    select_leaves_drive = ["MTimeDrive.fMjd", "MTimeDrive.fTime.fMilliSec", "MTimeDrive.fNanoSec",
                           "MReportDrive.fNominalZd"]

    rates_chunk = pd.DataFrame(columns=select_leaves_rates)
    drive_chunk = pd.DataFrame(columns=select_leaves_drive)

    zdlabels = np.arange(len(zdbins)-1)

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

    on_time_per_zd_chunk = np.zeros((len(zdbins)-1,))

    for (zd, on_times) in timeranges:
        if np.isnan(zd):
            pass
        else:
            on_time_per_zd_chunk[int(zd)] += on_times

    return on_time_per_zd_chunk


def calc_on_time(ganymed_input_list, zdbins, use_multiprocessing=True, n_chunks=8):
    """Calculate the observation on-time per zenith distance bin in parallel.
    
    Divide the input list of star files to n_chunks and start the on-time calculation per zenith distance bin
    with a multiprocessing.Pool of workers.
    On-Time information is read from the "Rates" tree of the Star output root files. Pointing information is read from
    the "Drives" tree.
    It matches the time of the entry in rates to the closest time of the entry in the drive information and
    sums the elapsed on time per zenith distance bin.
    
      There might be a small error in the zenith distance assignment, because the drive information is 
      more frequent than the rates. Each entry of fElapsedOnTime contains the sum of on time since the last
      entry. During that time, a change in zenith distance might have occured.
    
    :param ganymed_input_list: array of strings containing the star_files
    :param zdbins: ndarray
    :param use_multiprocessing: bool, if True, use multiprocessing
    :param n_chunks: int, number of chunks to divide the ganymed_input_list into
    :return: ndarray, on-time of the observation per zenith distance bin
    """

    ganymed_input_list = [entry.strip().replace(" ", "/") for entry in ganymed_input_list if not entry.startswith('#')]

    if len(ganymed_input_list) == 0:
        raise ValueError("No Files in input_list")

    if len(ganymed_input_list) < n_chunks:
        n_chunks = len(ganymed_input_list)
        print("Number of runs smaller than specified number of chunks, use {} chunks instead.".format(n_chunks))

    parts = (len(ganymed_input_list) / n_chunks * np.arange(n_chunks + 1)).astype("int_")

    print("\nOn time calculation ---------")
    print("Calculate the on time in", n_chunks, "chunks with a total of", len(ganymed_input_list), "entries.")

    on_time_parts = np.zeros([n_chunks, len(zdbins) - 1])

    if use_multiprocessing:
        pool = mp.Pool()
        result = [pool.apply_async(calc_on_time_chunks,
                                   args=(zdbins, ganymed_input_list[parts[i]:parts[i + 1]]))
                  for i in range(n_chunks)]
        pool.close()
        pool.join()
        for r in range(n_chunks):
            on_time_parts[r] = result[r].get()
    else:
        on_time_parts = [calc_on_time_chunks(zdbins, ganymed_input_list[parts[i]:parts[i + 1]])
                         for i in range(n_chunks)]

    print("--------- Finished on time calculation.")
    return np.sum(on_time_parts, axis=0)
