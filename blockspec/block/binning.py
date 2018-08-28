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