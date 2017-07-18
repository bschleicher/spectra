if os.path.isfile("/media/michi/523E69793E69574F/daten/data.pkl"):
    data = pd.read_pickle("/media/michi/523E69793E69574F/daten/data.pkl")
else:
    select_leaves = ['DataType.fVal', 'MPointingPos.fZd', 'FileId.fVal', 'MTime.fMjd', 'MTime.fTime.fMilliSec',
                     'MTime.fNanoSec']
    data = read_mars_fast(ganymed_all_events, leaves=select_leaves)

    data = data.assign(
        time=lambda x: x["MTime.fMjd"] + (x["MTime.fTime.fMilliSec"] + x["MTime.fNanoSec"] / (1000 * 1000)) / (
            1000 * 60 * 60 * 24))
    data.to_pickle("/media/michi/523E69793E69574F/daten/data.pkl")

zdlabels = range(len(zdbins) - 1)
data['Zdbin'] = pd.cut(data["MPointingPos.fZd"], zdbins, labels=zdlabels, include_lowest=True)

on_data = data.groupby("DataType.fVal").get_group(1.0)

on_time_per_zd = np.zeros((len(zdlabels),))
adj_check = (on_data["Zdbin"] != on_data["Zdbin"].shift()).cumsum()
timeranges = on_data.groupby(["FileId.fVal", adj_check], as_index=False, sort=False).agg(
    {'time': ['min', 'max'], 'Zdbin': ['min']}).values

for (file_id, min_time, max_time, zd) in timeranges:
    on_time_per_zd[int(zd)] += rates['MReportRates.fElapsedOnTime'][rates.time.between(min_time, max_time)].sum(
        axis=0) / (60 * 60)