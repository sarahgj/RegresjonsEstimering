import pytest
import pytz

from shyft.api import utctime_now  # To time the reading from SMG
from pandas.util.testing import assert_frame_equal
from regression_modules import *

start_time = utctime_now()

@pytest.fixture
def global_var():
    pytest.df_week = pd.read_csv('df_week_for_test.csv',index_col=[0])
    pytest.df_week.index = pd.to_datetime(pytest.df_week.index, utc=True)
    #pytest.df_week.tz_convert('Etc/GMT-1')
    pytest.MagKap_list = {'/NVE.-Randsfjord....-D1004A0V': 409.2}

def test_get_timeperiods():
    period, forecast_time, read_start, last_true_value = get_timeperiods('tilsig','mandag')
    assert forecast_time == '2019.06.17'
    assert read_start == '2015.06.08'
    assert last_true_value == '2019.06.17'
    period, forecast_time, read_start, last_true_value = get_timeperiods('tilsig','onsdag')
    assert forecast_time == '2019.06.24'
    assert read_start == '2015.06.08'
    assert last_true_value == '2019.06.17'
    #Magasin
    period, forecast_time, read_start, last_true_value = get_timeperiods('magasin','mandag')
    assert forecast_time == '2019.06.24'
    assert read_start == '2015.06.08'
    assert last_true_value == '2019.06.24'
    period, forecast_time, read_start, last_true_value = get_timeperiods('magasin','onsdag')
    assert forecast_time == '2019.07.01'
    assert read_start == '2015.06.08'
    assert last_true_value == '2019.06.24'

def test_GWh2percentage(global_var):
    df = GWh2percentage(pytest.df_week, pytest.MagKap_list)
    df.to_csv('df_for_test.csv')
    print(df)

def test_read_and_setup(global_var):
    df_week, MagKap_list, period, forecast_time, read_start = read_and_setup('magasin','mandag')
    #df_week.to_csv('df_week_for_test.csv')
    assert MagKap_list['/NVE.-Randsfjord....-D1004A0V'] == 409.2
    df_week.index = pd.to_datetime(df_week.index, utc=True) #converts to UTC for checking with output from cvs
    assert_frame_equal(pytest.df_week,df_week)

def test_printout():
    print('\nThe test script used a total of %.0f seconds\n' %(utctime_now() - start_time))

