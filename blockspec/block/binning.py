import numpy as np


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
