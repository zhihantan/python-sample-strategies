import pandas
import pytz

from matplotlib.dates import date2num
from pandas.stats.moments import ewma
from dateutil.parser import parse

def load_eod_data(file):
    eod_data = pandas.read_csv(file, delimiter=’;’,
                               parse_dates=True,
                               date_parser=lambda x: parse(x).replace(tzinfo=pytz.utc),
                               index_col=’date’)

    if len(eod_data) == 0:
        raise SystemExit

    eod_data[’dt’] = eod_data.index.map(lambda x: x.replace(tzinfo=pytz.utc))

    eod_data[’date’] = eod_data.index.map(date2num)

    return eod_data

def compute_macd(data):
    data[’ewma5’] = ewma(data.close, span=5)
    data[’ewma35’] = ewma(data.close, span=35)

    data[’macd’] = data.ewma5 - data.ewma35

    data[’macd_signal’] = ewma(data.macd, span=5)

    data[’histogram’] = data.macd - data.macd_signal