import pytest
import pandas as pd
import math
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from pandas.util.testing import assert_frame_equal

from modules import regression_modules as reg


@pytest.fixture
def global_var():
    pytest.df_week = pd.read_csv('df_week_for_test.csv',index_col=[0])
    pytest.df_week.index = pd.to_datetime(pytest.df_week.index, utc=True)
    #pytest.df_week.tz_convert('Etc/GMT-1')
    pytest.MagKap_list = {'/NVE.-Randsfjord....-D1004A0V': 409.2}
    pytest.var = ['tilsig', 'magasin']
    pytest.reg = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4']

def test_get_timeperiods():
    period, forecast_time, read_start = reg.get_timeperiods('tilsig', 'mandag')
    assert forecast_time == '2019.06.17'
    assert read_start == '2015.06.08'
    period, forecast_time, read_start = reg.get_timeperiods('tilsig', 'onsdag')
    assert forecast_time == '2019.06.17'
    assert read_start == '2015.06.08'
    #Magasin
    period, forecast_time, read_start = reg.get_timeperiods('magasin', 'mandag')
    assert forecast_time == '2019.06.24'
    assert read_start == '2015.06.08'
    period, forecast_time, read_start = reg.get_timeperiods('magasin', 'onsdag')
    assert forecast_time == '2019.06.24'
    assert read_start == '2015.06.08'

def test_GWh2percentage(global_var):
    df = reg.GWh2percentage(pytest.df_week, pytest.MagKap_list)
    df.to_csv('df_for_test.csv')
    for ts in df:
        print(ts)
        for value in df[ts]:
            # Some overshooting is possible, hence 105
            if not (value >=0 and value <=105):
                print(value)
                assert False, "Found one ore more numbers in df that is not in percentage."

#This function also tests read_import_SMG(variable, list_dict, list_names, period)
def test_read_and_setup(global_var):
    df_week, MagKap_list, period, forecast_time, read_start = reg.read_and_setup('magasin', 'mandag')
    assert MagKap_list['/NVE.-Randsfjord....-D1004A0V'] == 409.2
    df_week.index = pd.to_datetime(df_week.index, utc=True) #converts to UTC for checking with output from cvs
    #df_week.to_csv('df_week_for_test.csv')
    assert_frame_equal(pytest.df_week, df_week)


def test_make_fasit_key(global_var):
    for variable in pytest.var:
        period, forecast_time, read_start = reg.get_timeperiods(variable, test="mandag")
        for region in ['NO2']:
            fasit_key = reg.make_fasit_key(variable, region)
            df = period.read([fasit_key])
            if variable == 'magasin':
                if df.isnull().sum()[0] > 0:
                    assert False, 'More NaNs in {} than expected ({}) for {}, {}.'.format(fasit_key,df.isnull().sum()[0],region, variable)
            if variable == 'tilsig':
                if df.isnull().sum()[0] > 1:
                    print(df)
                    assert False, 'More NaNs in {} than expected ({}) for {}, {}.'.format(fasit_key,df.isnull().sum()[0],region, variable)


def test_get_input_variables_from_file(global_var):
    for variable in pytest.var:
        for region in pytest.reg:
            output = reg.get_input_variables_from_file(variable, region, backup=False, test=True)
            for x in output[:-1]:
                if math.isnan(x):
                    assert False, 'Something is wrong in the backup input variable file.'
            output2 = reg.get_input_variables_from_file(variable, region, backup=True, test=True)
            for x in output2[:-1]:
                if math.isnan(x):
                    assert False, 'Something is wrong in the tuning input variable file.'
