import sys
import pandas as pd
import time
import pytz
import numpy as np
from pandas import Timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import statsmodels.api as sm

from shyft.api import utctime_now  # To time the reading from SMG

from modules import read_and_setup as rs
from modules import write_and_show as ws


#Global variables
today = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')  # now
max_final_numb_kandidater = 20 #25
max_input_series = 496 #196
nb_weeks_tipping = 5  # number of weeks to do tipping back in time
tz = pytz.timezone('Etc/GMT-1')
columns = ['ant_kandidater', 'ant_serier', 'r2_modelled', 'r2_tippet', 'r2_samlet', 'short_period', 'max_p']
first_period = 220 #219  # Length of the long regression in weeks
min_kandidater = 1


def run_regression(auto_input,
                   variables: list = ['magasin', 'tilsig'],
                   regions: list = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4'],
                   jupyter: bool = False,
                   backup: bool = False,
                   loop: bool = False,
                   write: bool = True,
                   week_nb: int = False,
                  year: int = False) -> None:
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

    for region in regions:

        if not region in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4']:
            sys.exit("Region must be one out of: 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4'")

        for variable in variables:

            if not variable in ['magasin', 'tilsig']:
                sys.exit("Variable must be either 'tilsig' or 'magasin'")

            print('---------------------------------------------------------------')
            print('                          {}, {}                                  '.format(region, variable))
            print('---------------------------------------------------------------')

            
            
            df_week, MagKap = auto_input[variable]
            
            period, forecast_time, read_start = rs.get_timeperiods(variable, week_nb, year)
            
            reg_end = (pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(days=7)).strftime(
                '%Y.%m.%d')

            if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
                last_forecast = forecast_time
            else:
                last_forecast = forecast_time

            df_cleaned = deletingNaNs(df_week.loc[:last_forecast])

            if loop:
                if variable == 'tilsig':
                    max_kandidater = 196
                    min_kandidater = 1

                else:
                    max_kandidater = 171
                    min_kandidater = 1

                max_weeks = 220 #288
                min_weeks = 11
                print('max ant. kandidater: {}, min ant. kandidater: {}'.format(max_kandidater, min_kandidater))
                print('max ant. uker: {}, min ant. uker: {}'.format(max_weeks, min_weeks))

            start_time_loop = utctime_now()
            fasit, fasit_key = rs.make_fasit(variable, region, reg_end, period)
            print('Fasit er lest inn.\n')

            if fasit[fasit_key].isnull().any():
                print('OBS: Det mangler verdier på fasiten! Går videre til neste region i loopen..')
                continue

            sorted_r2 = get_R2_sorted(variable, df_cleaned, fasit, fasit_key)

            if loop:
                max_p = 0.025

                # First loop: Tuning number of candidates for best possible R2 combined
                df_ant_kandidater = pd.DataFrame(columns=columns)
                for antall in range(min_kandidater, max_kandidater + 1, 1):
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
                for short_period in range(min_weeks, max_weeks + 1, 1):
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
                ws.write_input_variables_to_file(region, variable, max_p, ant_kandidater_beste, short_period_beste)

                print('\nTuning for regionen tok %.0f minutter. \n' % ((utctime_now() - start_time_loop) / 60))

            else:
                # getting the best variables from input_variables_from_tuning.txt or input_variables_backup.txr
                short_period_beste, max_p, ant_kandidater_beste, input_file = rs.get_input_variables_from_file(variable,region,backup)
                chosen_r2_beste = sorted_r2[:ant_kandidater_beste]
                print("Input variables was read from: ", input_file)

            # SHOW RESULTS
            input1 = make_estimate(df_cleaned, fasit, fasit_key, last_forecast, short_period_beste, max_p,
                                   chosen_r2_beste, loop=False)
            input2 = fasit_key, ant_kandidater_beste, max_p, reg_end, read_start


            #WRITE RESULTS
            if write:
                # Write results from the regression to SMG.
                fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_ps, short_period, nb_weeks_tipping = input1

                # write to SMG:
                ws.write_SMG_regresjon(variable, region, tipping_ps[-1:])

                # write to SMG, virtual:
                ws.write_V_SMG_Regresjon(short_results, chosen_p, fasit_key, r2_modelled, MagKap)

            if jupyter:
                ws.show_result_jupyter(input1, input2)
            else:
                ws.show_result(input1, input2)

    print('---------------------------------------------------------------')
    print('                         SLUTT                                 ')
    print('---------------------------------------------------------------')
    print('\nRegresjon for alle regioner og variabler brukte totalt %.0f minutter. \n' % (
            (utctime_now() - start_tuning) / 60))


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
            sys.exit("wrong variable")

    # Chosing the chosen number of best r2 keys
    sorting = r2_original.sort_values(ascending=False)
    sorted_r2 = list(sorting.axes[0][:max_input_series])
    return sorted_r2


def make_estimate(df_cleaned, fasit, fasit_key, last_forecast, short_period, max_p, chosen_r2, loop=False):
    # start_ time.time()
    ant_kandidater = len(chosen_r2)
    df_tot = df_cleaned.join(fasit)

    # Long Regression: for the whole period to pick out the best fitted series
    # Placing this outside the forecast loop means that you might get a bit wrong results, but it is unlikely,
    # and I therefore keep it here because of computational costs.
    reg_end = (pd.to_datetime(time.strftime(last_forecast), format="%Y.%m.%d") - Timedelta(days=7)).strftime(
        '%Y.%m.%d')  # 6*52
    long_results, chosen_p, ant_break = regression(df_tot.loc[:reg_end], fasit_key, chosen_r2, max_p)

    #########################################################################################
    # Forecast loop setup
    start_tipping = 7 * nb_weeks_tipping
    reg_end_new = (pd.to_datetime(time.strftime(last_forecast), format="%Y.%m.%d") - Timedelta(
        days=start_tipping)).strftime('%Y.%m.%d')  # 6*52
    forecast_time_new = True
    tipping_times = []
    tipping_values = []
    while forecast_time_new != last_forecast:
        forecast_time_new = (
                    pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") + Timedelta(days=7)).strftime(
            '%Y.%m.%d')
        reg_start = (pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") - Timedelta(
            days=short_period * 7)).strftime('%Y.%m.%d')
        # Short Regression: for a short period defined by reg_start
        short_results, chosen_p, ant_break_short = regression(df_tot.loc[reg_start:reg_end_new], fasit_key, chosen_p, 1)
        r2_modelled = calc_R2(df_tot.loc[reg_start:reg_end_new][fasit_key],
                              short_results.predict(df_tot.loc[reg_start:reg_end_new][chosen_p]))
        ant_break_short += ant_break
        prediction = short_results.predict(df_cleaned[chosen_p]).loc[reg_end_new:forecast_time_new]
        reg_end_new = (pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") + Timedelta(days=7)).strftime(
            '%Y.%m.%d')
        tipping_times.append(prediction.index[-1])
        tipping_values.append(prediction[-1])

    tipping_ps = pd.Series(tipping_values, index=tipping_times)
    r2_tippet = calc_R2(fasit[fasit_key].loc[tipping_ps.index[0]:], tipping_ps[:fasit[fasit_key].index[-1]])
    r_samlet = (r2_modelled * 0.5 + r2_tippet * 0.5)
    ant_serier = len(chosen_p)
    if loop:
        #print(ant_kandidater, ant_serier, r2_modelled, r2_tippet, r_samlet, short_period, max_p)
        # end_time time.time()
        # print('Time to run make_estimate_while_looping: ',end_time-start_time)
        return ant_kandidater, ant_serier, r2_modelled, r2_tippet, r_samlet, short_period, max_p
    else:
        # end_time = time.time()
        # print('Time to run make_estimate(no loop): ',end_time-start_time)
        return fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_ps, short_period, nb_weeks_tipping


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

    with np.errstate(divide='ignore'):
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

        with np.errstate(divide='ignore'):
            results = sm.OLS(df_tot[fasit_key], df_tot[chosen_p]).fit()  # regression

    return results, chosen_p, ant_break


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
