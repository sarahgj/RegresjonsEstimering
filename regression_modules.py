import sys
import pandas as pd
import time
import pytz
import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import Timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from shyft.api import utctime_now  # To time the reading from SMG
import statsmodels.api as sm
from statkraft.ssa.wrappers import ReadWrapper
from statkraft.ssa.timeseriesrepository import TimeSeriesRepositorySmg
from statkraft.ssa.environment import SMG_PROD
from statkraft.ssa.adapter import ts_from_pandas_series


import import_from_SMG
import import_from_SMG_test


#Global variables
today = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')  # now
max_final_numb_kandidater = 25
max_input_series = 196
nb_weeks_tipping = 10  # number of weeks to do tipping back in time
tz = pytz.timezone('Etc/GMT-1')
columns = ['ant_kandidater', 'ant_serier', 'r2_modelled', 'r2_tippet', 'r2_samlet', 'short_period', 'max_p']
max_p = 0.025
first_period = 216  # Length of the long regression in weeks

########################################################################################################################
#                                        READ AND SETUP                                                                #
#                                                                                                                      #
#           Here series from SMG is read and the rime period etc is .                                                  #
########################################################################################################################


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

    period, forecast_time, read_start, last_true_value = get_timeperiods(variable, test)

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
        if not df_week[key].loc[last_true_value] > 0:
            if first_true:
                print('\n-------------------Feil i kjente %s verdier--------------------' % variable)
            print(df_week[key].loc[last_true_value], key)
            first_true = False
    print('\n\n')
    return df_week, MagKap_list, period, forecast_time, read_start


def get_timeperiods(variable, test=False):
    """This function finds what day it is today and chooses from that information the end of the regression
    and the time period of which the series should be read. It is used for read_and_setup().

    Args:
        variable: magasin or tilsig

    Returns:
        period: time period of which the series should be read (using the ReadWrapper from statkraft.ssa.wrappers)
        forecast_time: Time of last forecast

    Examples:
        >> period, forecast_inf, reg_end_inf, reg_start_inf = get_timeperiods('tilsig')
        >> period, forecast_mag, reg_end_mag, reg_start_mag = get_timeperiods('magasin')
    """
    # start_ time.time()
    if test == 'mandag':
        today = pd.to_datetime("2019.07.22 11:00", format="%Y.%m.%d %H:%M", errors='ignore')
    elif test == 'onsdag':
        today = pd.to_datetime("2019.07.24 11:00", format="%Y.%m.%d %H:%M", errors='ignore')
    else:
        today = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')  # today/now
    read_start = '2015.06.08'
    read_end = today + Timedelta(days=7)
    # The fasit value appears on wednesday 14 o'clock limiting the end time of the regression.
    
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        # Since we get the values for the tilsig series one week later than the magasin series, some adjustment
        # is necessary.
        if variable == 'tilsig':
            reg_mandag = today - Timedelta(days=today.weekday()) - Timedelta(days=14)
        else:
            reg_mandag = today - Timedelta(days=today.weekday()) - Timedelta(days=7)
    else:
        if variable == 'tilsig':
            reg_mandag = today - Timedelta(days=today.weekday()) - Timedelta(days=7)
        else:
            reg_mandag = today - Timedelta(days=today.weekday())
    reg_end = reg_mandag.strftime('%Y.%m.%d')
    # getting the period from the ReadWrapper from statkraft.ssa.wrappers
    period = ReadWrapper(start_time=read_start, end_time=read_end, read_from='SMG_PROD', tz=tz)
    # calculating forecast time and start of regression
    forecast_time = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") + Timedelta(days=7)).strftime('%Y.%m.%d')
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        last_true_value = forecast_time
    else:
        last_true_value = (pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(days=7)).strftime('%Y.%m.%d')
    # end_time time.time()
    #print('Time to run get_timeperiods: ', end_time - start_time)
    return period, forecast_time, read_start, last_true_value

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
        for date in df.index:
            if date.weekday() == 0 and date.hour == 0:
                start = date
                break
        datoer = [start + Timedelta(days=7 * i) for i in range(weeks + 1)]
        df_week = df.loc[datoer]
    elif variable == 'tilsig':
        df_week = df.resample('W', label='left', closed='right').sum()
        df_week = df_week.shift(1, freq='D')
    else:
        print('wrong variable, must be either magasin or tilsig')
        sys.exit(1)
    # end_time time.time()
    #print('Time to run index2week: ', end_time - start_time)
    return df_week


def deletingNaNs(df):
    """This function drops columns of a DataFrame (df) that has one or more NaNs."""
    # start_ time.time()
    df_old = df.copy()
    df.dropna(axis=1, how='any', inplace=True)
    for key in df_old:
        if str(key) not in df:
            print('Deleted ', key)
    # end_time time.time()
    #print('Time to run deletingNaNs: ', end_time - start_time)
    return df


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



#################################################################################################################
#####                               GENERAL METHODS                                                         #####
#####-------------------------------------------------------------------------------------------------------#####
#####                                                                                                       #####
#################################################################################################################

def run_regression(auto_input,
               variables: list = ['magasin', 'tilsig'],
               regions: list = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4'],
               jupyter: bool = False,
               backup: bool = False,
               loop: bool = False) -> None:
    """This function is the head function for the regression and it also deals with the outputs.

    Args:
        variable: Must be either 'magasin' or 'tilsig'
        regions: Must be one or more of the default regions
        jupyter: Set to Tru if the code is runned on Jupyter Notebooks
        backup: Set to True if you would rather use the backup input variables (input_variables_backup.txt) than the
                automatically updated variables from the last tuning (input_variables_from_tuning.txt).
        loop: Set to True if you want to do a Tuning to update input_variavles_from_tuning.txt and run the loop. This
                takes approximately half an hour.

    Tuning example:
        >> run_regression(auto_input, loop=True)
    Updating SMG on Jupyter Notebooks exaple:
        >> run_regression(auto_input, jupyter=True)
    """

    start_tuning = utctime_now()

    for variable in variables:

        if not variable in ['magasin', 'tilsig']:
            sys.exit("Variable must be either 'tilsig' or 'magasin'")

        df_week, MagKap, period, forecast_time, read_start = auto_input[variable]
        reg_end = (pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(days=7)).strftime(
            '%Y.%m.%d')

        if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
            last_forecast = forecast_time
        else:
            last_forecast = reg_end

        df_cleaned = deletingNaNs(df_week.loc[:last_forecast])

        if loop:
            if variable == 'tilsig':
                print('---------------------------------------------------------------')
                print('                        TILSIG                                 ')
                print('---------------------------------------------------------------')
                max_kandidater = 196
                min_kandidater = 5

            else:
                print('---------------------------------------------------------------')
                print('                        MAGASIN                                ')
                print('---------------------------------------------------------------')
                max_kandidater = 135
                min_kandidater = 5

            max_weeks = 208
            min_weeks = 10
            print('max ant. kandidater: {}, min ant. kandidater: {}'.format(max_kandidater, min_kandidater))
            print('max ant. uker: {}, min ant. uker: {}'.format(max_weeks, min_weeks))

        for region in regions:

            if not region in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4']:
                sys.exit("Region must one out of: 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4'")

            start_time_loop = utctime_now()
            fasit, fasit_key = make_fasit(variable, region, reg_end, period)

            if fasit[fasit_key].isnull().any():
                print('OBS: Det mangler verdier på fasiten! Går videre til neste region i loopen..')
                continue

            sorted_r2 = get_R2_sorted(variable, df_cleaned, fasit, fasit_key)

            if loop:

                # First loop: Tuning number of candidates for best possible R2 combined
                df_ant_kandidater = pd.DataFrame(columns=columns)
                for antall in range(min_kandidater, max_kandidater + 1, 2):
                    if antall > len(sorted_r2):
                        chosen_r2 = sorted_r2
                    else:
                        chosen_r2 = sorted_r2[:antall]
                    output = make_estimate(df_cleaned, fasit, fasit_key, last_forecast, first_period, max_p, chosen_r2,
                                           loop=True)
                    df_ant_kandidater = df_ant_kandidater.append(
                        {columns[0]: output[0], columns[1]: output[1], columns[2]: output[2], columns[3]: output[3],
                         columns[4]: output[4], columns[5]: output[5], columns[6]: output[6]}, ignore_index=True)
                    if antall > len(sorted_r2):
                        print('Feilmelding: Ønsket antall kandidater overskrider maks (%i).\n' % len(sorted_r2))
                        break
                idx_max = df_ant_kandidater.r2_samlet.idxmax(skipna=True)
                ant_kandidater_beste = int(df_ant_kandidater.ant_kandidater.values[idx_max])
                print('Beste ant_kandidater loop 1: ', ant_kandidater_beste)

                # Second loop: tuning length of the short regression for best possible R2 combined, using the best number of
                # candidates found in the First loop.
                df_short_period = pd.DataFrame(columns=columns)
                for short_period in range(min_weeks, max_weeks + 1, 4):
                    short_period = int(short_period)
                    final_chosen_r2 = sorted_r2[:ant_kandidater_beste]
                    output = make_estimate(df_cleaned, fasit, fasit_key, last_forecast, short_period, max_p,
                                           final_chosen_r2, loop=True)
                    df_short_period = df_short_period.append(
                        {columns[0]: output[0], columns[1]: output[1], columns[2]: output[2], columns[3]: output[3],
                         columns[4]: output[4], columns[5]: output[5], columns[6]: output[6]}, ignore_index=True)
                idx_max = df_short_period.r2_samlet.idxmax(skipna=True)
                short_period_beste = int(df_short_period.short_period.values[idx_max])
                print('Beste short_period loop 2: ', short_period_beste)

                # Getting the best input variables from loop and write to input_variables_from_tuning.txt
                df_all_methods = pd.concat([df_ant_kandidater, df_short_period], ignore_index=True, sort=False)
                idx_max = df_all_methods.r2_samlet.idxmax(skipna=True)
                ant_kandidater_beste = int(df_all_methods.ant_kandidater.values[idx_max])
                chosen_r2_beste = sorted_r2[:ant_kandidater_beste]
                short_period_beste = df_all_methods.short_period.values[idx_max]
                write_input_variables_to_file(region, variable, max_p, ant_kandidater_beste, short_period_beste)

            else:
                #getting the best variables from input_variables_from_tuning.txt or input_variables_backup.txr
                short_period_beste, max_p, ant_kandidater_beste, input_file = get_input_variables_from_file(variable, region, backup)
                chosen_r2_beste = sorted_r2[:ant_kandidater_beste]
                print("Input variables was read from: ", input_file)

            # Show results
            input1 = make_estimate(df_cleaned, fasit, fasit_key, last_forecast, short_period_beste, max_p,
                                   chosen_r2_beste, loop=False)
            input2 = fasit_key, ant_kandidater_beste, max_p, reg_end, read_start

            if not loop:
                #Write results from the regression to SMG.
                fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_df, short_period, nb_weeks_tipping = input1

                # write to SMG:
                write_SMG_regresjon(variable, region, tipping_df)

                # write to SMG, virtual:
                write_V_SMG_Regresjon(short_results, chosen_p, fasit_key, r2_modelled, MagKap)

            if jupyter:
                show_result_jupyter(input1, input2)
            else:
                show_result(input1, input2)

            print('\nTuning for regionen tok %.0f minutter. \n' % ((utctime_now() - start_time_loop) / 60))

    print('---------------------------------------------------------------')
    print('                         SLUTT                                 ')
    print('---------------------------------------------------------------')
    print('\nRegresjon for alle regioner og variabler brukte totalt %.0f minutter. \n' % (
                (utctime_now() - start_tuning) / 60))




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
        print('Could not make fasit_key from variable and region')
        sys.exit(1)
    return fasit_key


def get_input_variables_from_file(variable, region, backup=False):
    # start_ time.time()
    if backup:
        input_file = 'input_variables_backup.txt'
    else:
        input_file = 'input_variables_from_tuning.txt'
    string2find = '{:3} {:7}'.format(region,variable)
    with open(input_file,"r") as file:
        for line in file:
            if line.startswith(string2find):
                max_p = float(line[12:17])
                ant_kandidater: int = int(line[18:21])
                reg_period: int =int(line[22:25])
    # end_time time.time()
    #print('Time to run get_input_variables_from_file: ',end_time-start_time)
    return reg_period, max_p, ant_kandidater, input_file



def write_input_variables_to_file(region, variable, max_p, ant_kandidater, reg_period):
    # start_ time.time()
    output_file = 'input_variables_from_tuning.txt'
    string2find = '{:3} {:7}'.format(region,variable)
    with open(output_file,'r') as file:
        data = file.readlines()
        i = 0
        for line in data:
            if line.startswith(string2find):
                now = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')
                data[i] = '{:10s} {:5.3f} {:3d} {:3d} {}\n'.format(string2find, max_p, int(ant_kandidater), int(reg_period), now)
                break
            i +=1
    with open(output_file, 'w') as file:
        file.writelines(data)
    # end_time time.time()
    #print('Time to run write_input_variables_to_file: ',end_time-start_time)


def make_fasit(variable, region, reg_end, period):
    fasit_key = make_fasit_key(variable, region)
    fasit = period.read([fasit_key]).loc[:reg_end]
    return fasit, fasit_key


def calc_R2(Fasit, Model):
    """This function calculates the correlation coefficient between a model and a fasit.
    Args:
        Fasit: A timeseries
        Model: A modelled timesries

    Returns:
        R2: the correlation coefficient bewteen the two series."""
    # Calculating
    R2 = 1 - sum(np.power(Fasit - Model, 2)) / sum(np.power(Fasit - np.mean(Fasit), 2))
    return R2


def get_R2_sorted(variable, df_cleaned, fasit, fasit_key):
    start_r2_time = time.time()
    r2_original = pd.Series()
    for key in df_cleaned:
        if variable == 'tilsig':
            if df_cleaned[key].mean() == 0:
                print('passed for ', key, 'mean = 0')
                pass
            else:
                scalefac = fasit[fasit_key].mean() / df_cleaned[key].mean()
                r2_original[key] = calc_R2(fasit[fasit_key], df_cleaned[key] * scalefac)
        elif variable == 'magasin':
            r2_original[key] = calc_R2(fasit[fasit_key], df_cleaned[key])
        else:
            print('wrong variable')
            sys.exit(1)

    # Chosing the chosen number of best r2 keys
    sorting = r2_original.sort_values(ascending=False)
    sorted_r2 = list(sorting.axes[0][:max_input_series])
    end_r2_time = time.time()
    # print('Time to get chosen_r2 from df_cleaned: ', end_r2_time - start_r2_time)
    return sorted_r2



def make_estimate(df_cleaned, fasit, fasit_key, last_forecast, short_period, max_p, chosen_r2, loop=False):
    # start_ time.time()
    ant_kandidater = len(chosen_r2)
    df_tot = df_cleaned.join(fasit)
  
    # Long Regression: for the whole period to pick out the best fitted series
    # Placing this outside the forecast loop means that you might get a bit wrong results, but it is unlikely,
    # and I therefore keep it here because of computational costs.
    reg_end = (pd.to_datetime(time.strftime(last_forecast), format="%Y.%m.%d") - Timedelta(days=7)).strftime('%Y.%m.%d')  # 6*52
    long_results, chosen_p, ant_break = regression(df_tot.loc[:reg_end], fasit_key, chosen_r2, max_p)

    #########################################################################################
    #Forecast loop setup
    start_tipping = 7 * nb_weeks_tipping
    reg_end_new = (pd.to_datetime(time.strftime(last_forecast), format="%Y.%m.%d") - Timedelta(days=start_tipping)).strftime('%Y.%m.%d') #6*52
    forecast_time_new = True
    tipping_times = []
    tipping_values = []
    while forecast_time_new != last_forecast:
        forecast_time_new = (pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") + Timedelta(days=7)).strftime('%Y.%m.%d')
        reg_start = (pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") - Timedelta(days=short_period * 7)).strftime('%Y.%m.%d')
        # Short Regression: for a short period defined by reg_start
        short_results, chosen_p, ant_break_short = regression(df_tot.loc[reg_start:reg_end_new], fasit_key, chosen_p, 1)
        r2_modelled = calc_R2(df_tot.loc[reg_start:reg_end_new][fasit_key],short_results.predict(df_tot.loc[reg_start:reg_end_new][chosen_p]))
        ant_break_short += ant_break
        prediction = short_results.predict(df_cleaned[chosen_p]).loc[reg_end_new:forecast_time_new]
        reg_end_new = (pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") + Timedelta(days=7)).strftime('%Y.%m.%d')
        tipping_times.append(prediction.index[-1])
        tipping_values.append(prediction[-1])

    tipping_df = pd.Series(tipping_values, index=tipping_times)
    r2_tippet = calc_R2(fasit[fasit_key].loc[tipping_df.index[0]:], tipping_df[:fasit[fasit_key].index[-1]])
    r_samlet = (r2_modelled * 0.5 + r2_tippet * 0.5)
    ant_serier = len(chosen_p)
    if loop:
        print(ant_kandidater, ant_serier, r2_modelled, r2_tippet, r_samlet, short_period, max_p)
        # end_time time.time()
        #print('Time to run make_estimate_while_looping: ',end_time-start_time)
        return ant_kandidater, ant_serier, r2_modelled, r2_tippet, r_samlet, short_period, max_p
    else:
        # end_time = time.time()
        #print('Time to run make_estimate(no loop): ',end_time-start_time)
        return fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_df, short_period, nb_weeks_tipping



def regression(df_tot, fasit_key, chosen, max_p):
    """This function runs several regressions so that the optimal set of series in the result model are given in return. 
    Each time the regression is run, the series with the highest (worst) p-value is dropped out of the chosen list, 
    until the highest p in p-values in the results equal max_p, or only 6 series are left.
    Args:
        df_tot: A DataFrame with the series used for the regression.
        fasit_key: The key of the fasit in the df_tot.
        chosen: A list of keys in df_tot based on the best correlattion with the fasit.
        max_p: The maximum allowed p-value in the model, except for when the model ends up with too few series.

    Returns:
        results: The results of the final regression.
        chosen_p: A list of keys of the final chosen set of timeseries for the regression model.
        ant_break: 1 if the loop picking out p-values stopped because of minimum number of series limit."""
    
    # First regression
    first_model = sm.OLS(df_tot[fasit_key], df_tot[chosen])
    
    # Initializing loop
    results = first_model.fit()
    chosen_p = chosen.copy()
    ant_break = 0
    
    # Looping through until final model is chosen
    while max(results.pvalues) > max_p or len(results.pvalues) >= max_final_numb_kandidater:
        if len(results.pvalues) <= min_kandidater:
            ant_break = 1  # count
            break
        chosen_p.remove(results.pvalues.idxmax())  # updating the chosen list
        results = sm.OLS(df_tot[fasit_key], df_tot[chosen_p]).fit()  # regression
        
    return results, chosen_p, ant_break




def show_result(input1, input2, variable_file=False):
    """This function prints out and plots the results from the regression."""
    fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_df, short_period, nb_weeks_tipping = input1
    fasit_key, ant_kandidater, max_p, reg_end, read_start = input2
    # start_ time.time()
    plt.interactive(False)
    reg_start = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") - Timedelta(days=short_period * 7)).strftime('%Y.%m.%d')
    print('\n-----------------------------------------------------------------------')
    print('RESULTATER FOR %s\n' % fasit_key)
    print('Regresjonsperiode fra: %s til: %s.' % (reg_start, reg_end))
    if variable_file:
        print('Input variablene (reg_period og ant_kandidater) ble hentet fra: ',variable_file)
    else:
        print('Input variablene (reg_period og ant_kandidater) ble funnet ved tuning.')
    print('Valgte %.2f kandidater til regresjonen utifra korrelasjon med fasitserien.' % (ant_kandidater))
    print('Valgte så ut de med p-value < %.5f, som var %i stk.' % (max_p, len(long_results.pvalues)))
    print('R2 for regresjonen (kort periode): %.5f' % r2_modelled)
    print('R2 mellom fasit og tipping: %.5f\n' % (calc_R2(fasit[fasit_key].loc[tipping_df.index[0]:], tipping_df[:fasit[fasit_key].index[-1]])))
    print('Fasit:\n', fasit[fasit_key][-4:])
    print('\nModdelert/Tippet:\n', tipping_df[-5:])
    # end_time time.time()
    #print('Time to run show_result: ',end_time-start_time)


def show_result_jupyter(input1, input2, variable_file=False):
    fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_df, short_period, nb_weeks_tipping = input1
    fasit_key, ant_kandidater, max_p, reg_end, read_start = input2
    """This function prints out and plots the results from the regression."""
    plt.interactive(False)
    reg_start = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") - Timedelta(days=short_period * 7)).strftime('%Y.%m.%d')
    print('\n-----------------------------------------------------------------------')
    print('RESULTATER FOR %s\n' % fasit_key)
    print('Regresjonsperiode brukt til setup for siste tipping: %s til: %s.' % (read_start, reg_end))
    if variable_file:
        print('Input variablene (reg_period og ant_kandidater) ble hentet fra: ',variable_file)
    else:
        print('Input variablene (reg_period og ant_kandidater) ble funnet ved tuning.')
    print('Regresjonsperiode brukt på modellen for siste tipping: %s til: %s, satt til %d uker.' % (reg_start, reg_end, int(short_period)))
    print('Valgte %d kandidater til regresjonen utifra korrelasjon med fasitserien.'%(int(ant_kandidater)))
    print('Valgte så ut de med p-value < %.5f, som var %i stk.' % (max_p, len(long_results.pvalues)))
    print('R2 for regresjonen (kort periode): %.5f' % r2_modelled)
    print('R2 mellom fasit og tipping: %.5f\n'%(calc_R2(fasit[fasit_key].loc[tipping_df.index[0]:], tipping_df[:fasit[fasit_key].index[-1]])))
    print('Fasit:\n', fasit[fasit_key][-4:])
    print('\nModdelert/Tippet:\n', tipping_df[-5:])
    if fasit_key[-3:] == '105':
        color_tipping = 'blue'
    elif fasit_key[-3:] == '132':
        color_tipping = 'lightblue'
    else:
        print("The fasit key should end with 105 or 132")
        sys.exit(1)
    
    # Plot with regression:    
    plt.figure(figsize=(16, 10))
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        plt.plot(fasit[fasit_key].loc[:reg_end], color='k', linewidth=2.0, label='fasit')
    else:
        plt.plot(fasit[fasit_key].loc[:], color='k', linewidth=2.0, label='fasit')
    plt.plot(short_results.predict(df_tot[chosen_p].loc[reg_start:reg_end]), color='orange', label='regresjon på historie(kort periode)')
    plt.plot(long_results.predict(df_tot[chosen_p].loc[:reg_start]), color='cyan', label='regresjon på historie (lang periode)')
    plt.plot(short_results.predict(df_tot[chosen_p].loc[:reg_start]), color='deeppink', label='modell på historie (kort periode)')
    plt.plot(tipping_df, label='tipping', color=color_tipping)  # , marker='o')
    plt.title('Regresjon for: %s' % fasit_key)
    plt.legend()

    # Plot just prediction:
    plt.figure(figsize=(16, 10))
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        plt.plot(fasit[fasit_key].loc[tipping_df.index[0]:], color='k', linewidth=2.0, label='fasit')
    else:
        plt.plot(fasit[fasit_key].loc[tipping_df.index[0]:reg_end], color='k', linewidth=2.0, label='fasit')
    plt.plot(tipping_df, label='tipping', color=color_tipping)  # , marker='o')
    plt.title('Tipping for: %s' % fasit_key)
    plt.legend()

    # Plot input series:
    plt.figure(figsize=(16, 10))
    plt.plot(fasit[fasit_key], color='k', linewidth=3.0, label='fasit')
    for key in chosen_p:
        if fasit_key[-3:] == '105':
            sfac = df_tot[fasit_key].mean() / df_tot[key].mean()
            plt.plot(df_tot[key]* sfac)  # , marker='o')
        elif fasit_key[-3:] == '132':
            plt.plot(df_tot[key])  # , marker='o')
    plt.plot(tipping_df, label='tipping', color=color_tipping)  # , marker='o')
    plt.title('Regresjonsserier for: %s' % fasit_key)
    plt.legend()
    plt.show()


def write_SMG_regresjon(variable, region, df):
    """This function writes pandas series (df) to smg series (ts) chosen according to the chosen region."""
    smg = TimeSeriesRepositorySmg(SMG_PROD)

    # define ts to write to
    if variable == 'tilsig':
        ts = 'RegEstimatTilsig' + region + '-U9100S0BT0105'
    if variable == 'magasin':
        ts = 'RegEstimatMag' + region + '...-U9104S0BT0132'
    print('\nSkriver estimerte verdier til: ', ts)

    # get metainfo from ts
    minfo = smg.get_meta_info_by_name([ts])

    # we need to attach meta information to the Pandas series since it is needed to create the TimeSeries object
    df.meta_info = minfo[0]

    # convert the Pandas series to a time series
    time_series_to_write = ts_from_pandas_series(df, filter_out_nans=True)

    # writing
    result = smg.write([time_series_to_write])


def write_V_SMG_Regresjon(results, chosen_p, fasit_key, r2_modelled, MagKap_mag=False):
    start_time = time.time()
    now = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')
    expression = str('! Sist oppdatert {}\n!R2 med {} serier: {}\n'.format(now, len(chosen_p), r2_modelled))
    vekt_serie = "F{tall}={serie}*{vekt}\n"
    region = str(fasit_key[6:9])

    if str(fasit_key[-3:]) == '132':
        sum_av_vekter = '## = ({alle_vekter})/100'
        for tall, serie, vekt in sorted(zip(range(len(chosen_p)), chosen_p, results.params)):
            MagKap = MagKap_mag[serie]
            if ('-U' not in serie) and (MagKap != 0):
                serie = "(100*@TRANSFORM(%'{}','WEEK','FIRST')/{})".format(serie, MagKap)
            elif ('-U' in serie) and (MagKap != 0):
                serie = "(100*%'{}'/{})".format(serie, MagKap)
            elif ('-U' not in serie) and (MagKap == 0):
                serie = "@TRANSFORM(%'{}','WEEK','FIRST')".format(serie)
            elif ('-U' in serie) and (MagKap == 0):
                serie = "%'{}'".format(serie)
            expression += (str((vekt_serie.format(tall=tall + 1, serie=serie, vekt=vekt))))
        expression += str(
            sum_av_vekter.format(alle_vekter='F' + '+F'.join([str(tall + 1) for tall in range(len(chosen_p))])))
        expression += '\n@SET_TS_VALUNIT(##,132)'
        ts = 'RegresjonMag' + region + '.......-U9104S0BT0132'

    if str(fasit_key[-3:]) == '105':
        sum_av_vekter = '## = {alle_vekter}'
        for tall, serie, vekt in sorted(zip(range(len(chosen_p)), chosen_p, results.params)):
            serie = "@TRANSFORM(%'{}','WEEK','SUM')".format(serie)
            expression += (str((vekt_serie.format(tall=tall + 1, serie=serie, vekt=vekt))))
        expression += str(sum_av_vekter.format(alle_vekter='F' + '+F'.join([str(tall + 1) for tall in range(len(chosen_p))])))
        expression += '\n@SET_TS_VALUNIT(##,105)'
        if 'No' in region:
            ts = 'RegresjonTilsigNO' + region[-1] + '.-U9100S0BT0105'
        elif 'Se' in region:
            ts = 'RegresjonTilsigSE' + region[-1] + '.-U9100S0BT0105'

    smg = TimeSeriesRepositorySmg(SMG_PROD)
    print('Skriver formel for siste regresjon til: ', ts)
    info = smg.get_meta_info_by_name([ts])
    #print('-----------------------------------------------------------------------')
    #print(expression)
    #print('-----------------------------------------------------------------------')
    smg.update_virtual({info[0]: expression})
    end_time = time.time()
    print('Time to run write_V_SMG_Regresjon: ',end_time-start_time)


