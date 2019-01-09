import numpy as np
import pandas as pd
from blockspec.block.makeblocks import makeblocks


def multiple_day_binning(df, n_days=7, **kwargs):
    """Binning function that allows for grouping in fixed-length-intervals of multiple days.
    
    As opposed to the on-time binning functions, it bins to intervals with fixed length, 
    even if there is no data in one of the intervals.
    It uses the resampling capabilities of Pandas and allows the passing of **kwargs to 
    pandas.DataFrame.resample.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be binned.
    n_days : int, optional
        Number of days to bin to.
    
    Returns
    -------
    
    pandas.Series
        Returns a Series with the bin numbers of each entry.
    
    """
    df["time_mid"] = df.run_start + (df.run_stop - df.run_start) / 2
    frequency = str(n_days*24) + 'h'

    print("Resampling with:", frequency)

    resampler = df.resample(frequency, on='time_mid', base=-12, **kwargs)

    print("Number of blocks:", len(resampler.groups))

    df["bin"] = np.zeros(len(df))
    for name, group in resampler:
        df.loc[group.index, 'bin'] = [name] * len(group)
    df['bin'] = (df.bin != df.bin.shift()).cumsum()

    print("Number of filled blocks:", df["bin"].iloc[-1])
    print("Number of runs per block:")
    print(resampler['n_on'].count())

    return df['bin']


def bayesian_block_binning(timecut, prior=5.1):
    """Binning function for binning in bayesian blocks.
       
    Parameters
    ----------
    timecut : pandas.DataFrame
        DataFrame to be binned.
    prior : float, optional
        Prior to be used for the bayesian block calculation.
    
    Returns
    -------
    pandas.Series
        Returns a Series with the bin numbers of each entry.
    
    """

    blocks = makeblocks(timecut.time_mean.values,
                        timecut.excess_rate.values,
                        timecut.excess_rate_err.values,
                        prior)
    s_cp = blocks[4]

    print("Number of Blocks:", len(s_cp)/2)

    timecut['bin'] = 0
    timecut.loc[s_cp, 'bin'] = 1

    return timecut['bin'].cumsum()


def threshold_binning(df, bins):
    df["bin"] = pd.cut(df["threshold_median"], bins, )
    return df["bin"]


def threshold_binning_ontime(df, time_in_hours):
    """Bin runs by "fThresholdMinSet" until a combined ontime of more than time_in_hours is reached. 
       The algorithm sorts by threshold in ascending order. Starting at bin number 0, 
       one run is added at a time, until the goal of an combined ontime of time_in_hours of the bin is met. """
    time = time_in_hours * 60 * 60
    threshold_column = "threshold_minset"
    data = df[[threshold_column, "ontime"]].copy()
    sel = data.sort_values(threshold_column)
    data['bin'] = None
    bin_number = 0
    ontime_sum = 0
    for tuples in sel.itertuples():
        ontime_sum += tuples[2]
        data.loc[tuples[0], "bin"] = bin_number
        if ontime_sum > time:
            bin_number += 1
            ontime_sum = 0

    for i in range(data["bin"].max() + 1):
        sel = data.loc[data["bin"] == i, threshold_column]
        minimum, maximum = sel.min(), sel.max()
        data.loc[data["bin"] == i, "bin"] = "{:3.0f} to {:3.0f}".format(minimum, maximum)

    return data["bin"]


def zenith_binning_ontime(df, time_in_hours):
    """Bin runs by "zd" until a combined ontime of more than time_in_hours is reached. 
       The algorithm sorts by zd in ascending order. Starting at bin number 0, 
       one run is added at a time, until the goal of an combined ontime of time_in_hours of the bin is met. """
    time = time_in_hours * 60 * 60
    threshold_column = "zd"
    data = df[[threshold_column, "ontime"]].copy()
    sel = data.sort_values(threshold_column)
    data['bin'] = None
    bin_number = 0
    ontime_sum = 0
    for tuples in sel.itertuples():
        ontime_sum += tuples[2]
        data.loc[tuples[0], "bin"] = bin_number
        if ontime_sum > time:
            bin_number += 1
            ontime_sum = 0

    for i in range(data["bin"].max() + 1):
        sel = data.loc[data["bin"] == i, threshold_column]
        minimum, maximum = sel.min(), sel.max()
        data.loc[data["bin"] == i, "bin"] = "{:3.0f} to {:3.0f}".format(minimum, maximum)

    return data["bin"]


def exclude_periods(df, array_of_periods=None):
    """Bin runs by excluding nights of specified periods.
       array_of_periods: two times nested array
            example: [['2014-05-30', '2014-06-07'], ['2014-08-01', '2014-08-14']]"""
    if array_of_periods is None:
        array_of_periods = [['2014-05-30', '2014-06-07'], ['2014-08-01', '2014-08-14']]
    exclude = array_of_periods
    data = df[['run_start', 'run_stop']].copy()
    data['bin'] = 0
    for index, period in enumerate(exclude, 1):
        a = pd.to_datetime(period)
        data.loc[np.bitwise_not(np.bitwise_or((df['run_stop'] < a[0]), (df['run_start'] > a[1]))), 'bin'] = index
    return data['bin']
