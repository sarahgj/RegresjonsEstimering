import sys
import pandas as pd
import time
import pytz
import math
from pandas import Timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from statkraft.ssa.wrappers import ReadWrapper

from imports import import_from_SMG
from tests import import_from_SMG_test

#Global variables
#today = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')  # now
#max_final_numb_kandidater = 25
#max_input_series = 196
#nb_weeks_tipping = 10  # number of weeks to do tipping back in time
tz = pytz.timezone('Etc/GMT-1')
#columns = ['ant_kandidater', 'ant_serier', 'r2_modelled', 'r2_tippet', 'r2_samlet', 'short_period', 'max_p']
#first_period = 216  # Length of the long regression in weeks
#min_kandidater = 6

def read_and_setup(variable: str, test: str=False) -> [pd.DataFrame, list, ReadWrapper, str, str]:
    """This function is the head function for reading and setting up the series used for the regression

    Args:
        variable: must be either 'magasin' or 'tilsig'

    Returns:
        df_week: DataFrame with all the time series in weekly values
        MagKap_list: a list of the MagKap numbers to each series (0 if none).
        period: ReadWrapper with period of which the series are read in (to later read the fasit with the same period)
        forecast_time: Time of "true" forecast
        read_start: Time of the start of the regression and reading in

    Examples:
        >> df_week, MagKap, period, forecast_time, read_start = read_and_setup('tilsig')
    """

    period, forecast_time, read_start = get_timeperiods(variable, test)

    if variable == 'tilsig':
        print('---------------------------------------------------------------')
        print('                        TILSIG                                 ')
        print('---------------------------------------------------------------')
        if test:
            list_dict, list_names_dict = import_from_SMG_test.import_tilsig()
        else:
            list_dict, list_names_dict = import_from_SMG.import_tilsig()
        df_all, MagKap_list = read_import_SMG(variable, list_dict, list_names_dict, period)
        df_week = index2week(df_all, variable).loc[:forecast_time]

    else:
        print('---------------------------------------------------------------')
        print('                        MAGASIN                                ')
        print('---------------------------------------------------------------')
        if test:
            list_dict, list_names_dict = import_from_SMG_test.import_magasiner()
        else:
            list_dict, list_names_dict = import_from_SMG.import_magasiner()
        df_all, MagKap_list = read_import_SMG(variable, list_dict, list_names_dict, period)
        corrected_mag = GWh2percentage(df_all, MagKap_list)
        df_week = index2week(corrected_mag, variable).loc[:forecast_time]

    # Printing information:
    print('Mandag det tippes for: ', forecast_time)
    first_true = True
    for key in df_week:
        if not df_week[key].loc[forecast_time] > 0:
            if first_true:
                print('\n-------------------Feil i siste verdi for kjente %s--------------------' % variable)
            print(df_week[key].loc[forecast_time], key)
            first_true = False
    print('\n\n')
    return df_week, MagKap_list, period, forecast_time, read_start


def get_timeperiods(variable: str, test: str = False) -> [ReadWrapper, str, str]:
    """This function finds what day it is today and chooses from that information the end of the regression
    and the time period of which the series should be read. It is used for read_and_setup().

    Args:
        variable: magasin or tilsig

    Returns:
        period: time period of which the series should be read (using the ReadWrapper from statkraft.ssa.wrappers)
        forecast_time: Time of last forecast
        read_start: start time of the regression on the period etc
    """
    if test == 'mandag':
        now = pd.to_datetime("2019.06.24 11:00", format="%Y.%m.%d %H:%M", errors='ignore')
    elif test == 'onsdag':
        now = pd.to_datetime("2019.06.27 14:00", format="%Y.%m.%d %H:%M", errors='ignore')
    else:
        now = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')

    read_start = '2015.06.08'
    read_end = now + Timedelta(days=7)

    # getting the period from the ReadWrapper from statkraft.ssa.wrappers
    period = ReadWrapper(start_time=read_start, end_time=read_end, read_from='SMG_PROD', tz=tz)

    # The fasit value (unknown) appears on wednesday 14 o'clock limiting the end time of the regression.
    # The forecast time is the time of the last true value of kjente magasin and tilsig.
    if (0 <= now.weekday() <= 1) or (now.weekday() == 2 and now.hour < 14):  # before wednesday 2pm
        # Since we get the values for the tilsig series one week later than the magasin series, some adjustment
        # is necessary.
        if variable == 'tilsig':
            reg_mandag = now - Timedelta(days=now.weekday()) - Timedelta(days=14)
        else:
            reg_mandag = now - Timedelta(days=now.weekday()) - Timedelta(days=7)
    else:
        if variable == 'tilsig':
            reg_mandag = now - Timedelta(days=now.weekday()) - Timedelta(days=7)
        else:
            reg_mandag = now - Timedelta(days=now.weekday())
    # calculating forecast time and start of regression
    if (0 <= now.weekday() <= 1) or (now.weekday() == 2 and now.hour < 14):  # before wednesday 2pm
        forecast_time = (reg_mandag + Timedelta(days=7)).strftime('%Y.%m.%d')
    else:
        forecast_time = reg_mandag.strftime('%Y.%m.%d')

    return period, forecast_time, read_start

def GWh2percentage(df, MagKap):
    """This function converts from GWh to percentage magasinfylling.
    Args:
        df: Dataframe with all series that are intended for the magasin regression.
        MagKap: List of magasine capacity for each magasinfylling-series (=0 for series that's already in percentage).

    Returns:
        df: df with all series en percentage magasinfylling
    """
    # start_ time.time()
    for key in MagKap:
        if MagKap[key] != 0:
            df[key] = (100 * df[key] / MagKap[key])
    # end_time time.time()
    #print('Time to run GWh2percentage: ', end_time - start_time)
    return df


def index2week(df, variable):
    """This function changes the index of a dataframe to weekly values. This is done differently for the
    magasin input series than the tilsig input series.

    df : DataFrame object which contain

    variable : byttes med variable str

    """
    # start_ time.time()
    if variable == 'magasin':
        diff = df.index[-1] - df.index[0]  # number of days in df
        weeks = math.floor(diff.days / 7)  # number of weeks in df
        start = False
        for date in df.index:
            if date.weekday() == 0 and date.hour == 0:
                start = date
                break
            if not start:
                sys.exit("Something went wrong with finding the first monday og the dataframe in the index2week")
        datoer = [start + Timedelta(days=7 * i) for i in range(weeks + 1)]
        df_week = df.loc[datoer]
    elif variable == 'tilsig':
        df_week = df.resample('W', label='left', closed='right').sum()
        df_week = df_week.shift(1, freq='D')
    else:
        sys.exit("wrong variable, must be either magasin or tilsig")
    # end_time time.time()
    #print('Time to run index2week: ', end_time - start_time)
    return df_week




def read_import_SMG(variable, list_dict, list_names, period):
    """This function reads time series from SMG_PROD, and is specially designed for the read_and_setup() module.

    Args:
        variable: Either 'tilsig' or 'magasin' dependent on what regression the series to be read should be used for.
        list_dict: A list of dictionaries containing the series to be read for the respective regression, and also
        MagKap if needed (otherwise MagKap = 0).
        list_names: A list of names of each dict in list_dict. It is used for printing aout the name for the dict of
        series hvile read.
        period: Time period that the series should be read in for. This is the output of the function: get_timeperiods().

    Returns:
        df: DataFrame with all series that are intended for the regression of ..variable..
        MagKap_dict: A dictionary of MagKaps for all series.

    Examples:
        >> df_inf, MagKap_inf = read_import_SMG('tilsig', inf_dict, inf_names, period)
        >> df_mag, MagKap_mag = read_import_SMG('magasin',mag_dict, mag_names, period)
    """
    # start_ utctime_now()  # for time taking
    if variable == 'tilsig':
        print('Forventet innlesingstid er +/-180 sekunder.')
    elif variable == 'magasin':
        print('Forventet innlesingstid er +/-6 sekunder.')

    def read_region(series, dict_name, period):
        """This function does the reading part for each dictionary in the list of dictionaries: list_dict"""
        keys_dict = []
        MagKap = {}
        print(f'Leser nå {dict_name}..')
        # separating the series keys and the MagKap number in the dictionaries
        for key in series:
            keys_dict.append(series[key][0])
            if len(series[key]) >= 2:
                MagKap[series[key][0]] = series[key][1]
        # reading all series in each dict.
        df = period.read(keys_dict)
        return df, MagKap

    MagKap_dict = {}
    df = pd.DataFrame()
    # Sending in one and one list of dictionaries with series to be read for reading
    for (series, dict_name) in zip(list_dict, list_names):
        df_new, MagKap = read_region(series, dict_name, period)
        MagKap_dict.update(MagKap)
        df = pd.concat([df, df_new], axis=1, sort=False)
    #print('\nInnlesning for %s tok totalt %.0f sekunder. \n' % (variable, utctime_now() - start_time))
    return df, MagKap_dict


def make_fasit(variable, region, reg_end, period):
    fasit_key = make_fasit_key(variable, region)
    fasit = period.read([fasit_key]).loc[:reg_end]
    return fasit, fasit_key



def make_fasit_key(variable, region):
    if ('N' in region) and (variable == 'tilsig'):
        fasit_key = '/Norg-No' + region[-1] + '.Fasit.....-U9100S0BT0105'
    elif ('S' in region) and (variable == 'tilsig'):
        fasit_key = '/Sver-Se' + region[-1] + '.Fasit.....-U9100S0BT0105'
    elif ('N' in region) and (variable == 'magasin'):
        fasit_key = '/Norg-NO' + region[-1] + '.Fasit.....-U9104A5R-0132'
    elif ('S' in region) and (variable == 'magasin'):
        fasit_key = '/Sver-SE' + region[-1] + '.Fasit.....-U9104A5R-0132'
    else:
        sys.exit("Could not make fasit_key from variable and region")
    return fasit_key


def get_input_variables_from_file(variable, region, backup=False, test=False):
    if test:
        if backup:
            input_file = r'..\imports\input_variables_backup.txt'
        else:
            input_file = r'..\imports\input_variables_from_tuning.txt'
    else:
        if backup:
            input_file = '..\imports\input_variables_backup.txt'
        else:
            input_file = '..\imports\input_variables_from_tuning.txt'
    string2find = '{:3} {:7}'.format(region,variable)
    with open(input_file,"r") as file:
        for line in file:
            if line.startswith(string2find):
                max_p = float(line[12:17])
                ant_kandidater: int = int(line[18:21])
                reg_period: int =int(line[22:25])
    return reg_period, max_p, ant_kandidater, input_file
