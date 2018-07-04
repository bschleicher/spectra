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


def get_and_prepare_data(source="Mrk 501", first_night=20130101, last_night=20180418, time=0.1, frac=0.93):
    dfall = get_qla_data(last_night=last_night, first_night=first_night, sources=[source])
    df = dfall.loc[dfall.zd < 60]
    df = df.copy()
    df.drop('windgust_rms', axis=1, inplace=True)
    df.dropna(inplace=True)
    checked2 = df.loc[(df.r750cor/df.r750ref > frac)].dropna()
    checked = df.loc[(df.r750cor/df.r750ref > frac) & (df.r750cor/df.r750ref < 1.2) & (df.humidity < 80)].dropna()
    plotdf = makenightly(checked2)
    humcut = makenightly(checked)
    humcut.where((humcut.threshold_median < 653) & (humcut.ontime > time), inplace=True)
    humcut.dropna(inplace=True)
    timecut = plotdf.where(plotdf.ontime > time).dropna()
    return timecut, humcut, checked2


def save_runlist_star(filename, df, basepath="/media/michi/523E69793E69574F/daten/star/"):
    test = np.array([df.night.values.astype(np.int).astype(np.str),
                     df.run_id.values.astype(np.int).astype(np.str)]).T
    output = []
    for [night, run_id] in test:
        run_id = '{:03d}'.format(int(run_id))
        output.append(basepath + night[:4] + '/' + night[4:6] + '/' + night[6:] +
                      " " + night + '_' + run_id + '_I.root')
    np.savetxt(filename, output, fmt='%s')
    print('Saved runlist to {:s} with the basepath {:s}'.format(filename, basepath))
    print('Number of entries: {:d}'.format(len(df)))
    return


def blocks_from_df(timecut, prior=5.1):
    return makeblocks(timecut.time_mean.values, timecut.excess_rate.values, timecut.excess_rate_err.values, prior)


def plot_nightly(timecut, blocks, prior, source):
    plt.figure()
    s_cp = blocks[4]
    s_amplitudes = blocks[5]
    s_amplitudes_err = blocks[6]
    plt.errorbar(x=timecut.time_mean.values,
                 y=timecut.excess_rate.values,
                 yerr=timecut.excess_rate_err.values,
                 fmt=".",
                 label=source)
    plt.fill_between(timecut.time_mean.iloc[s_cp].values,
                     s_amplitudes-s_amplitudes_err,
                     s_amplitudes+s_amplitudes_err,
                     color="green",
                     alpha=0.4,
                     linewidth=0,
                     label="Bayesian Blocks, prior {:.1f}".format(prior))
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


def create_blocks(source="Mrk 501",
                  destination_path="/media/michi/523E69793E69574F/daten/Mrk501/blocks/",
                  basepath_of_starfiles="/media/michi/523E69793E69574F/daten/star/",
                  start=20130101,
                  stop=20180418,
                  time=0.5,
                  frac=0.93,
                  use_thresh_and_hum=False,
                  prior=5.1,
                  dryrun=False):
    timecut, humcut, checked2 = get_and_prepare_data(source=source,
                                                     first_night=start,
                                                     last_night=stop,
                                                     time=time,
                                                     frac=frac)
    if use_thresh_and_hum:
        timecut = humcut
    mapping, blocks = save_block_files(timecut,
                                       checked2,
                                       basepath_of_starfiles,
                                       prior,
                                       basepath=destination_path,
                                       dryrun=dryrun)
    plot_nightly(timecut, blocks, prior, source)
    return timecut, mapping, blocks
