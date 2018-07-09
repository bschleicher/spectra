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


def get_and_prepare_data(source="Mrk 501",
                         first_night=20130101,
                         last_night=20180418,
                         time=0.1,
                         datacheck_frac=0.93,
                         thresh_value=653,
                         hum_value=80,
                         binning_function=None):
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

    plotdf = makenightly(checked2, binning_function=binning_function)
    humcut = makenightly(checked, binning_function=binning_function)
    humcut.where((humcut.threshold_median < thresh_value) & (humcut.ontime > time), inplace=True)
    humcut.dropna(inplace=True)
    timecut = plotdf.where(plotdf.ontime > time).dropna()
    return timecut, humcut, checked2


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


def plot_dataframe(timecut, source):
    plt.errorbar(x=timecut.time_mean.values,
                 y=timecut.excess_rate.values,
                 yerr=timecut.excess_rate_err.values,
                 fmt=".",
                 label=source)


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
    y = (pew['excess_rate'].values[:, np.newaxis] * np.ones(3)).flatten()
    y_err = (pew['excess_rate_err'].values[:, np.newaxis] * np.ones(3)).flatten()
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
    plt.fill_between(timecut.time_mean.iloc[s_cp].values,
                     s_amplitudes-s_amplitudes_err,
                     s_amplitudes+s_amplitudes_err,
                     color="green",
                     alpha=0.5,
                     linewidth=0,
                     label="Bayesian Blocks, prior {:.1f}".format(prior))


def plot_nightly(timecut, blocks=None, prior=None, source='Fact-source', as_block=False):
    plt.figure()
    plot_dataframe(timecut, source)

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
                        use_block_number_as_name=True):

    rangeframe = blocks[["run_start", "run_stop"]]
    print(rangeframe)

    destination = basepath
    namelist = []
    for i in range(len(rangeframe)):
        ranges = [rangeframe.index[i], rangeframe.run_start.iloc[i], rangeframe.run_stop.iloc[i]]
        select = (checked2.time_mean >= ranges[1]) & (checked2.time_mean <= ranges[2])

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
                  block_binning='makeblocks'):
    timecut, humcut, checked2 = get_and_prepare_data(source=source,
                                                     first_night=start,
                                                     last_night=stop,
                                                     time=time,
                                                     datacheck_frac=frac,
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
        blocks, block_humcut, block_checked2 = get_and_prepare_data(source=source,
                                                                    first_night=start,
                                                                    last_night=stop,
                                                                    time=time,
                                                                    datacheck_frac=frac,
                                                                    binning_function=block_binning)

        if use_thresh_and_hum:
            blocks = block_humcut

        mapping, blocks = save_block_files_df(timecut, block_checked2, blocks, basepath_of_starfiles,
                                              basepath=destination_path, dryrun=dryrun)

        plot_nightly(timecut, blocks, source=source, as_block=True)

    return timecut, mapping, blocks
