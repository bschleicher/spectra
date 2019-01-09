# this is an adaption of get_qla_data() from pyfact
# https://github.com/fact-project/pyfact/blob/master/fact/qla.py
# In the future, I want to add this function to pyfact


from peewee import (
    FloatField, IntegerField, DateTimeField, CompositeKey
)


from fact.factdb.models import FactDataModel


from fact.factdb import (
    read_into_dataframe,
    AnalysisResultsRunLP as QLA,
    AnalysisResultsRunISDC as ISDC,
    Source

)


class RunInfoWithRate(FactDataModel):

    fazimuthmax = FloatField(db_column='fAzimuthMax', null=True)
    fazimuthmean = FloatField(db_column='fAzimuthMean', null=True)
    fazimuthmin = FloatField(db_column='fAzimuthMin', null=True)
    fnight = IntegerField(db_column='fNight')
    fnumevents = IntegerField(db_column='fNumEvents', null=True)
    fontime = FloatField(db_column='fOnTime', null=True)
    foutsidetempmean = FloatField(db_column='fOutsideTempMean', null=True)
    foutsidetemprms = FloatField(db_column='fOutsideTempRms', null=True)
    frunid = IntegerField(db_column='fRunID')
    frunstart = DateTimeField(db_column='fRunStart', null=True)
    frunstop = DateTimeField(db_column='fRunStop', null=True)
    fsequenceid = IntegerField(db_column='fSequenceID', null=True)
    fsourcekey = IntegerField(db_column='fSourceKEY', null=True)
    ftngdust = FloatField(db_column='fTNGDust', null=True)
    fthresholdavgmean = FloatField(db_column='fThresholdAvgMean', null=True)
    fthresholdmax = IntegerField(db_column='fThresholdMax', null=True)
    fthresholdmedmean = FloatField(db_column='fThresholdMedMean', null=True)
    fthresholdmedrms = FloatField(db_column='fThresholdMedRms', null=True)
    fthresholdmedian = FloatField(db_column='fThresholdMedian', null=True)
    fthresholdminset = IntegerField(db_column='fThresholdMinSet', null=True)
    fthresholdmintimediff = IntegerField(db_column='fThresholdMinTimeDiff', null=True)
    fzenithdistancemax = FloatField(db_column='fZenithDistanceMax', null=True)
    fzenithdistancemean = FloatField(db_column='fZenithDistanceMean', null=True)
    fzenithdistancemin = FloatField(db_column='fZenithDistanceMin', null=True)

    fr750cor = FloatField(db_column='fR750Cor', null=True)
    fr750ref = FloatField(db_column='fR750Ref', null=True)
    fhumiditymean = FloatField(db_column='fHumidityMean', null=True)
    fhumidityrms = FloatField(db_column='fHumidityRms', null=True)
    fairpressuremean = FloatField(db_column='fAirPressureMean', null=True)
    fairpressurerms = FloatField(db_column='fAirPressureRms', null=True)
    fdewpointmean = FloatField(db_column='fDewPointMean', null=True)
    fdewpointrms = FloatField(db_column='fDewPointRms', null=True)
    foutsidetempmean = FloatField(db_column='fOutsideTempMean', null=True)
    foutsidetemprms = FloatField(db_column='fOutsideTempRms', null=True)
    fwindgustmean = FloatField(db_column='fWindGustMean', null=True)
    fwindgustrms = FloatField(db_column='fWindGustRms', null=True)
    fwindspeedmean = FloatField(db_column='fWindSpeedMean', null=True)
    fwindspeedrms = FloatField(db_column='fWindSpeedRms', null=True)

    class Meta:
        db_table = 'RunInfo'
        indexes = ((('fnight', 'frunid'), True),)
        primary_key = CompositeKey('fnight', 'frunid')


def get_qla_data(
        first_night=None,
        last_night=None,
        sources=None,
        database_engine=None,
        run_database='QLA'
        ):
    """
    Request QLA results from our database
    first_night: int or None
        If given, first night to query as FACT night integer.
    last_night: int or None
        If given, last night to query as FACT night integer.
    sources: iterable[str]
        If given, only these sources will be requested.
        Names have to match Source.fSourceName in our db.
    database_engine: sqlalchmey.Engine
        If given, the connection to use for the query.
        Else, `fact.credentials.create_factdb_engine` will be used to create it.
    database: str
        Can be either 'QLA' or 'ISDC'
    """
    if run_database == 'QLA':
        run_db = QLA
    elif run_database == 'ISDC':
        run_db = ISDC
    else:
        raise ValueError("run_database must be either 'QLA' or 'ISDC'")

    query = run_db.select(run_db.frunid.alias('run_id'),
                          run_db.fnight.alias('night'),
                          run_db.fnumexcevts.alias('n_excess'),
                          run_db.fnumsigevts.alias('n_on'),
                          (run_db.fnumbgevts * 5).alias('n_off'),
                          run_db.fontimeaftercuts.alias('ontime'),
        RunInfoWithRate.frunstart.alias('run_start'),
        RunInfoWithRate.frunstop.alias('run_stop'),
        Source.fsourcename.alias('source'),
        Source.fsourcekey.alias('source_key'),

        RunInfoWithRate.fr750cor.alias('r750cor'),
        RunInfoWithRate.fr750ref.alias('r750ref'),
        RunInfoWithRate.ftngdust.alias('dust'),
        RunInfoWithRate.fthresholdmedian.alias('threshold_median'),
        RunInfoWithRate.fthresholdminset.alias('threshold_minset'),
        RunInfoWithRate.fhumiditymean.alias('humidity'),
        RunInfoWithRate.fhumidityrms.alias('humidity_rms'),
        RunInfoWithRate.fairpressuremean.alias('pressure'),
        RunInfoWithRate.fairpressurerms.alias('pressure_rms'),
        RunInfoWithRate.fdewpointmean.alias('dewpoint'),
        RunInfoWithRate.fdewpointrms.alias('dewpoint_rms'),
        RunInfoWithRate.foutsidetempmean.alias('temp'),
        RunInfoWithRate.foutsidetemprms.alias('temp_rms'),
        RunInfoWithRate.fwindgustmean.alias('windgust'),
        RunInfoWithRate.fwindgustrms.alias('windgust_rms'),
        RunInfoWithRate.fwindspeedmean.alias('windspeed'),
        RunInfoWithRate.fwindspeedrms.alias('windspeed_rms'),
        RunInfoWithRate.fazimuthmean.alias('az'),
        RunInfoWithRate.fzenithdistancemean.alias('zd'),
        RunInfoWithRate.fnumevents.alias('n')
    )

    on = (RunInfoWithRate.fnight == run_db.fnight) & (RunInfoWithRate.frunid == run_db.frunid)
    query = query.join(RunInfoWithRate, on=on)
    query = query.join(Source, on=RunInfoWithRate.fsourcekey == Source.fsourcekey)

    if first_night is not None:
        query = query.where(run_db.fnight >= first_night)
    if last_night is not None:
        query = query.where(run_db.fnight <= last_night)

    if sources is not None:
        query = query.where(Source.fsourcename.in_(sources))

    runs = read_into_dataframe(query, engine=database_engine)

    # drop rows with NaNs from the table, these are unfinished qla results
    # runs.dropna(inplace=True)

    runs.sort_values('run_start', inplace=True)

    return runs
