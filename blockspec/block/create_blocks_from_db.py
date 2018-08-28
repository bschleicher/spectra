from blockspec.block.temp_database_model import get_qla_data
from blockspec.block.makeblocks import makeblocks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def makenightly(checked2, binning_function=None, **kwargs):
    if binning_function is None:
        groupby_key = 'night'
    else:
        checked2['bins'] = binning_function(checked2, **kwargs)
        groupby_key = 'bins'

    plotdf = checked2.groupby(groupby_key).aggregate(
        {'humidity': 'mean', 'n_on': 'sum', 'n_off': 'sum', 'ontime': 'sum', 'threshold_median': 'mean', 'dust': 'mean',
         'run_start': 'min', 'run_stop': 'max', 'dewpoint': 'mean', 'temp': 'mean', 'zd': 'mean'})

    plotdf['excess_rate'] = (plotdf.n_on - plotdf.n_off * 0.2) / plotdf.ontime * 3600
    plotdf['excess_rate_err'] = np.sqrt(plotdf.n_on + 0.2 ** 2 * plotdf.n_off) / plotdf.ontime * 3600
    plotdf['ontime'] /= 3600
    plotdf['time_width'] = plotdf.run_stop - plotdf.run_start
    plotdf['time_mean'] = plotdf.run_start + 0.5 * plotdf.time_width
    plotdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    plotdf.dropna(inplace=True)
    return plotdf


def get_data(source="Mrk 501",
                         first_night=20130101,
                         last_night=20180418,
                         datacheck_frac=0.93,
                         hum_value=80,
                         ):
    dfall = get_qla_data(last_night=last_night, first_night=first_night, sources=[source])
    df = dfall.loc[dfall.zd < 60]
    df = df.copy()
    df.drop('windgust_rms', axis=1, inplace=True)
    df.dropna(inplace=True)
    checked2 = df.loc[(df.r750cor/df.r750ref > datacheck_frac)].dropna()
    checked2["time_mean"] = checked2.run_start + (checked2.run_start - checked2.run_stop) / 2
    checked = df.loc[(df.r750cor/df.r750ref > datacheck_frac)
                     & (df.r750cor/df.r750ref < 1.2)
                     & (df.humidity < hum_value)].dropna()
    return checked, checked2


def apply_binning_and_timecut(checked,
                              checked2,
                              time=0.1,
                              thresh_value=653,
                              binning_function=None):
    plotdf = makenightly(checked2, binning_function=binning_function)
    timecut = plotdf.where(plotdf.ontime > time).dropna()

    humcut = makenightly(checked, binning_function=binning_function)
    humcut.where((humcut.threshold_median < thresh_value) & (humcut.ontime > time), inplace=True)
    humcut.dropna(inplace=True)

    return timecut, humcut


def save_runlist_star(filename,
                      df,
                      basepath="/media/michi/523E69793E69574F/daten/star/",
                      use_ganymed_convention=True):

    test = np.array([df.night.values.astype(np.int).astype(np.str),
                     df.run_id.values.astype(np.int).astype(np.str)]).T
    output = []
    for [night, run_id] in test:
        run_id = '{:03d}'.format(int(run_id))
        outputstring = basepath + night[:4] + '/' + night[4:6] + '/' + night[6:]
        if use_ganymed_convention:
            outputstring += " "
        else:
            outputstring += "/"
        outputstring += night + '_' + run_id + '_I.root'
        output.append(outputstring)

    np.savetxt(filename, output, fmt='%s')
    print('Saved runlist to {:s} with the basepath {:s}'.format(filename, basepath))
    print('Number of entries: {:d}'.format(len(df)))
    return


def blocks_from_df(timecut, prior=5.1):
    return makeblocks(timecut.time_mean.values, timecut.excess_rate.values, timecut.excess_rate_err.values, prior)


def plot_dataframe(timecut, source, alpha=0.6):
    plt.errorbar(x=timecut.time_mean.values,
                 y=timecut.excess_rate.values,
                 yerr=timecut.excess_rate_err.values,
                 fmt=".",
                 label=source,
                 alpha=alpha)


def plot_dataframe_as_block(timecut):
    plt.errorbar(x=timecut.time_mean.values,
                 xerr=timecut.time_width.values/2,
                 y=timecut.excess_rate.values,
                 fmt=" ",
                 label="Blocks")
    pew = timecut[['run_start', 'run_stop']].copy()
    pew["mid"] = pew.run_stop + (pew.run_start.shift(-1) - pew.run_stop)
    pew["mid"].iloc[-1] = pew["mid"].iloc[-2]
    x = pew.values.flatten()
    y = (timecut['excess_rate'].values[:, np.newaxis] * np.ones(3)).flatten()
    y_err = (timecut['excess_rate_err'].values[:, np.newaxis] * np.ones(3)).flatten()
    where = np.ones(len(x))
    where[2::3] = 0

    plt.fill_between(x, y - y_err,
                     y + y_err,
                     where=where,
                     color="green",
                     alpha=0.5,
                     linewidth=0,
                     label="Binning with ")


def plot_blocks(timecut, blocks, prior):
    s_cp = blocks[4]
    s_amplitudes = blocks[5]
    s_amplitudes_err = blocks[6]
    plt.plot(timecut.time_mean.iloc[s_cp].values, s_amplitudes, "green")
    plt.fill_between(timecut.time_mean.iloc[s_cp].values,
                     s_amplitudes-s_amplitudes_err,
                     s_amplitudes+s_amplitudes_err,
                     color="green",
                     alpha=0.5,
                     linewidth=0,
                     label="Bayesian Blocks, prior {:.1f}".format(prior))


def plot_nightly(timecut, blocks=None, prior=None, source='Fact-Source', as_block=False):
    plt.figure()
    plot_dataframe(timecut, source, alpha=0.6)

    if as_block:
        plot_dataframe_as_block(blocks)
    else:
        plot_blocks(timecut, blocks, prior=prior)

    plt.xlabel("Time")
    plt.ylabel("Excess rate Evts/h")
    plt.legend()
    plt.grid()
    plt.tight_layout()


def save_block_files(timecut,
                     checked2,
                     basepath_of_starfiles,
                     prior=5.1,
                     basepath="/media/michi/523E69793E69574F/daten/Mrk501/blocks/",
                     dryrun=False):
    blocks = blocks_from_df(timecut, prior)
    s_cp = blocks[4]
    # s_amplitudes = blocks[5]
    # s_amplitudes_err = blocks[6]
    print("Number of Blocks:", len(s_cp)/2)
    array = timecut.index[s_cp].to_series()

    datetimes = pd.to_datetime(array, format="%Y%m%d")

    datetimes[datetimes.duplicated(keep='last')] -= pd.to_timedelta(1, unit='d')

    ranges = datetimes.dt.strftime(date_format="%Y%m%d").values.astype('int64')
    rangeframe = pd.DataFrame(ranges.reshape(-1, 2), columns=["start", "stop"])
    print(rangeframe)
    # nights = timecut.loc[((timecut.index >= rangeframe.start[0]) & (timecut.index <= rangeframe.stop[0]))]
    destination = basepath
    namelist = []
    for ranges in rangeframe.itertuples():
        nights = timecut.loc[((timecut.index >= ranges[1]) & (timecut.index <= ranges[2]))]
        start = nights.iloc[0].run_start
        stop = nights.iloc[-1].run_stop
        filename = destination + ranges[1].astype('str') + "_" + ranges[2].astype('str') + ".txt"
        if not dryrun:
            save_runlist_star(filename, checked2[checked2.night.isin(nights.index)], basepath=basepath_of_starfiles)
        namelist.append([ranges[0], filename, start, stop])
    namelist = np.array(namelist)
    mapping = pd.DataFrame(namelist, columns=["block", "filepath", "start", "stop"])
    mapping.set_index("block", inplace=True)
    if not dryrun:
        mapping.to_json(destination + "mapping.json")
    return mapping, blocks


def save_block_files_df(timecut,
                        checked2,
                        blocks,
                        basepath_of_starfiles,
                        basepath="/media/michi/523E69793E69574F/daten/Mrk501/blocks/",
                        dryrun=False,
                        use_block_number_as_name=True,
                        block_binning=None):
    if hasattr(block_binning, "__call__"):
        checked2["bin"] = block_binning(checked2)

    rangeframe = blocks[["run_start", "run_stop"]]
    print(rangeframe)

    destination = basepath
    namelist = []
    for i in range(len(rangeframe)):
        ranges = [rangeframe.index[i], rangeframe.run_start.iloc[i], rangeframe.run_stop.iloc[i]]
        select = (checked2.time_mean >= ranges[1]) & (checked2.time_mean <= ranges[2])
        if hasattr(block_binning, "__call__"):
            select = checked2["bin"] == ranges[0]

        if use_block_number_as_name:
            filename = destination + str(ranges[0]) + ".txt"
        else:
            filename = destination + ranges[1].astype('str') + "_" + ranges[2].astype('str') + ".txt"
        if not dryrun:
            save_runlist_star(filename, checked2[select], basepath=basepath_of_starfiles)
        namelist.append([ranges[0], filename, ranges[1], ranges[2]])
    namelist = np.array(namelist)
    mapping = pd.DataFrame(namelist, columns=["block", "filepath", "start", "stop"])
    mapping.set_index("block", inplace=True)
    if not dryrun:
        mapping.to_json(destination + "mapping.json")
        mapping.to_pickle(destination + "mapping.pkl")
    return mapping, blocks


def create_blocks(source="Mrk 501",
                  destination_path="/media/michi/523E69793E69574F/daten/Mrk501/blocks/",
                  basepath_of_starfiles="/media/michi/523E69793E69574F/daten/star/",
                  start=20130101,
                  stop=20180418,
                  time=0.5,
                  frac=0.93,
                  use_thresh_and_hum=False,
                  prior=5.1,
                  dryrun=False,
                  start_binning=None,
                  block_binning='makeblocks',
                  blocks_runwise=True):

    """ 
    This function reads the runwise entries from the FACT QLA database and either performs a binning to individual 
    blocks as specified by block_binning. It creates a DataFrame called mapping.json and the ganymed txt files at 
    the destination path.
    The entries of the runlists for ganymed are prepended with the basepath_of_starfiles option in front of the
    usual "YYYY/MM/DD YYYYMMDD_RRR.root" folder structure. 
    Use the path, where the star files are located on your device.
    
    Parameters
    ----------
    source : str
        Identifier of the source as used by FACT, e.g. "Mrk 501", "Mrk 421", "Crab".
    
    destination_path : str
        Path, where the mapping.json DataFrame and runlists will be saved
    
    basepath_of_starfiles : str
        Path to be prepended in front of the "YYYY/MM/DD YYYYMMDD_RRR.root" folder structure
        of the star files. Should be the location of the star files on your device.
        
    start : int
        Date from which to start. Should be of the form YYYYMMDD, e.g. 20130223.
    
    stop : int
        End date of the query. Should be of the form YYYYMMDD, e.g. 20130223.
    
    time : float
        A cut for minimum on-time in one block in hours. 
        Example: 0.5 for a minimum on time per block of 30 minutes.
    
    frac : float
        Minimum fraction for the DataCheck: 
        (rate of the run)/(nominal rate of the period) > frac
    
    use_thresh_and_hum : bool
        Apply a runwise cut of humidity > 80 % and a block-wise cut of median-threshold < 653.
    
    prior : float
        Prior for the bayesian block binning. Has no effect if block_binning is different from "makeblocks".
    
    dryrun : bool
        If True, don't save mapping DataFrame and runlist text-files. If false, save them to the path
        specified by destination_path.
    
    start_binning : function or None
        Pass a binning function that is applied before the block-binning. Only has an effect if blocks_runwise is False.
        If blocks_runwise is True, the block_binning will be applied runwise.
        If blocks_runwise is False, start_binning is applied before the block-binning and if start_binning is None,
        the block_binning will be applied to a nightly binning.
            
    block_binning : function or 'makeblocks' or None
        Binning for the individual blocks. If 'makeblocks', the bayesian block algorithm with a prior set by 
        the prior parameter is used. If None, nightly binning is used.
        Depending on the blocks_runwise and start_binning parameters, the block binning is either applied runwise 
        or on the binning specified by start_binning.
        
    blocks_runwise : bool
        If True, block_binning is applied runwise.
        If False, block_binning is applied on binned runs with a binning specified by start_binning.
        
    
    """


    checked, checked2 = get_data(source=source,
                                 first_night=start,
                                 last_night=stop,
                                 datacheck_frac=frac)

    timecut, humcut = apply_binning_and_timecut(checked,
                                                checked2,
                                                time=time,
                                                thresh_value=653,
                                                binning_function=start_binning)
    if use_thresh_and_hum:
        timecut = humcut

    if block_binning == 'makeblocks':
        mapping, blocks = save_block_files(timecut,
                                           checked2,
                                           basepath_of_starfiles,
                                           prior,
                                           basepath=destination_path,
                                           dryrun=dryrun)
        plot_nightly(timecut, blocks, prior=prior, source=source)
    else:
        if blocks_runwise:
            pass_frame = checked
            pass_frame2 = checked2
        else:
            pass_frame = timecut
            pass_frame2 = humcut

        blocks, block_humcut = apply_binning_and_timecut(pass_frame,
                                                         pass_frame2,
                                                         time=time,
                                                         thresh_value=653,
                                                         binning_function=block_binning)

        if use_thresh_and_hum:
            blocks = block_humcut

        mapping, blocks = save_block_files_df(timecut, checked2, blocks, basepath_of_starfiles,
                                              basepath=destination_path, dryrun=dryrun, block_binning=block_binning)

        plot_nightly(timecut, blocks, source=source, as_block=True)

    return timecut, mapping, blocks
