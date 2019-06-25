import pandas as pd
import time
import pytz
import math
import numpy as np
import matplotlib.pyplot as plt
#plt.interactive(False)
#from datetime import datetime
from pandas import Timedelta

from shyft.api import utctime_now  # To time the reading from SMG

import statsmodels.api as sm
from statkraft.ssa.wrappers import ReadWrapper
from statkraft.ssa.timeseriesrepository import TimeSeriesRepositorySmg
from statkraft.ssa.environment import SMG_PROD
#from statkraft.ssa.timeseries import MetaInfo, TimeStepConstraint, PointInterpretation, Calendar, TimeSeries
from statkraft.ssa.adapter import ts_from_pandas_series
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from import_from_SMG import *
from default_input_variables import *


today = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')  # today


def get_timeperiods(variable):
    """This function finds what day it is today and chooses from that information the end of the regression
    and the time period of which the series should be read.

    Args:
        variable: magasin or tilsig

    Returns:
        period: time period of which the series should be read (using the ReadWrapper from statkraft.ssa.wrappers)
        forecast_time: Time of last forecast

    Examples:
        >>> period, forecast_inf, reg_end_inf, reg_start_inf = get_timeperiods('tilsig')
        >>> period, forecast_mag, reg_end_mag, reg_start_mag = get_timeperiods('magasin')
    """
    read_start = '2015.06.08'
    read_end = today + Timedelta(days=7)
    # The fasit value appears on wednesday 14 o'clock limiting the end time of the regression.
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        # Since we get the values for the tilsig series one week later than the magasin series, some adjustment
        # is neccessary.
        if variable == 'tilsig':
            reg_mandag = today - Timedelta(days=today.weekday()) - Timedelta(days=14)
        elif variable == 'magasin':
            reg_mandag = today - Timedelta(days=today.weekday()) - Timedelta(days=7)
    else:
        if variable == 'tilsig':
            reg_mandag = today - Timedelta(days=today.weekday()) - Timedelta(days=7)
        elif variable == 'magasin':
            reg_mandag = today - Timedelta(days=today.weekday())
    reg_end = reg_mandag.strftime('%Y.%m.%d')
    tz = pytz.timezone('Etc/GMT-1')
    # getting the period from the ReadWrapper from statkraft.ssa.wrappers
    period = ReadWrapper(start_time=read_start, end_time=read_end, read_from='SMG_PROD', tz=tz)
    # calculating forecast time and start of regression
    forecast_time = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") + Timedelta(days=7)).strftime('%Y.%m.%d')
    # printing out information about the chosen times
    if variable == 'tilsig:':
        print('---------------------------------------------------------------')
        print('                        TILSIG                                 ')
        print('---------------------------------------------------------------')
    elif variable == 'magasin':
        print('---------------------------------------------------------------')
        print('                        MAGASIN                                ')
        print('---------------------------------------------------------------')
    print(variable, ' tipping: ', forecast_time)
    print(period)
    return period, forecast_time, read_start


def read_and_setup(variable):
    """This function is the head function for reading and seting up the series used for the regression

    Args:
        variable: magasin or tilsig

    Returns:
        df_week: dataframe with all the time series in weekly values
        MagKap_list: a list of the MagKap numbers to each series (0 if none).
        period: period of which the series are read in (to later read the fasit with the same period)
        forecast_time: Time of "true" forecast

    Examples:
        >>> inf_week, MagKap_inf, period, forecast_inf = read_and_setup('tilsig')
        >>> mag_week, MagKap_mag, period, forecast_mag = read_and_setup('magasin')
    """
    if variable == 'tilsig':
        list_dict, list_names_dict = import_tilsig()
    elif variable == 'magasin':
        list_dict, list_names_dict = import_magasiner()
    period, forecast_time, read_start = get_timeperiods(variable)
    df_all, MagKap_list = read_import_SMG(variable, list_dict, list_names_dict, period)
    if variable == 'magasin':
        corrected_mag = GWh2percentage(df_all, MagKap_list)
        df_week = index2week(corrected_mag, variable).loc[:forecast_time]
    elif variable == 'tilsig':
        df_week = index2week(df_all, variable).loc[:forecast_time]

    # feil på siste verdi printes ut
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        last_true_value = forecast_time
    else:
        last_true_value = (
                    pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(days=7)).strftime(
            '%Y.%m.%d')
    for key in df_week:
        if not df_week[key].loc[last_true_value] > 0:
            print('\n-------------------Feil i kjente %s verdier---------------------------' % variable)
            print(df_week[key].loc[last_true_value], key)
            print('\n\n')

    return df_week, MagKap_list, period, forecast_time, read_start


def read_import_SMG(variable, list_dict, list_names, period):
    """This function reads timeseries from SMG_PROD, and is specially designed for regresjonstipping.py

    Args:
        variable: Either 'tilsig' or 'magasin' dependent on what regression the series to be read should be used for.
        list_dict: A list of dictionaries containing the series to be read for the respective regression, and also MagKap if
                   needed (othervise MagKap = 0).
        list_names: A list of names of each dict in list_dict. It is used for printing aout the name for the dict of series hvile read.
        period: Time period that the series should be read in for. This is the output of the function: get_timeperiods().

    Returns:
        df: Dataframe with all series that are intended for the regression of ..variable..
        MagKap_dict: A dictionary of MagKaps for all series.

    Examples:
        >>> df_inf, MagKap_inf = read_import_SMG('tilsig', inf_dict, inf_names, period)
        >>> df_mag, MagKap_mag = read_import_SMG('magasin',mag_dict, mag_names, period)
    """
    start_time = utctime_now()  # for time taking
    if variable == 'tilsig':
        print('Forventet innlesingstid er +/-180 sekunder.')
    elif variable == 'magasin':
        print('Forventet innlesingstid er +/-6 sekunder.')

    def read_region(series, dict_name, period):
        """This function does the reading part for each dictionary in the list of dictionaries: list_dict"""
        keys_dict = []
        MagKap = {}
        print(f'Leser nå {dict_name}..')
        # seperating the series keys and the MagKap number in the dictionaries
        for key in series:
            keys_dict.append(series[key][0])
            if len(series[key]) >= 2:
                MagKap[series[key][0]] = series[key][1]
        read_start = utctime_now()
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
    print('\nInnlesning for %s tok totalt %.0f sekunder. \n' % (variable, utctime_now() - start_time))
    return df, MagKap_dict


def GWh2percentage(df, MagKap):
    """This function converts from GWh to percentage magasinfylling.
    Args:
        df: Dataframe with all series that are intended for the magasin regression.
        MagKap: List of magasine capacity for each magasinfylling-series (=0 for series that's already in percentage).

    Returns:
        df: df with all series en percentage magasinfylling
    """
    for key in MagKap:
        if MagKap[key] != 0:
            df[key] = (100 * df[key] / MagKap[key])
    return df


def index2week(df, variable):
    """This function changes the index of a dataframe to weekly values. This is done differently for the
    magasin input series than the tilsig input series.

    df : DataFrame object which contain

    variable : byttes med variable str

    """
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
        df_week = pd.DataFrame()
        df_week = df.resample('W', label='left', closed='right').sum()
        df_week = df_week.shift(1, freq='D')
    return df_week


def deletingNaNs(df):
    """This function drops columns of a DataFrame (df) that has one or more NaNs."""
    df_old = df.copy()
    df.dropna(axis=1, how='any', inplace=True)
    for key in df_old:
        if str(key) not in df:
            print('Deleted ', key)
    return df


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


def show_result(show_result_input):
    """This function prints out and plots the results from the regression."""
    fasit_key, ant_kandidater, max_p, fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_df, reg_end, reg_period, ant_break_long, nb_weeks_tipping = show_result_input
    plt.interactive(False)
    reg_start = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") - Timedelta(days=reg_period * 7)).strftime(
        '%Y.%m.%d')
    print('\n-----------------------------------------------------------------------')
    print('RESULTATER FOR %s\n' % fasit_key)
    print('Regresjonsperiode fra: %s til: %s.' % (reg_start, reg_end))
    print('Valgte %.2f kandidater til regresjonen utifra korrelasjon med fasitserien.' % (ant_kandidater))
    print('Valgte så ut de med p-value < %.5f, som var %i stk.' % (max_p, len(long_results.pvalues)))
    print('Antall stopp av loopen som luker ut for høye p pga minimum antall serier i den lange regresjonen: %i/%i' % (
    ant_break_long, nb_weeks_tipping))
    print('R2 for regresjonen (kort periode): %.5f' % r2_modelled)
    # start_tipping = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") - Timedelta(days=7*(nb_weeks_tipping-2))).strftime('%Y.%m.%d')
    print('R2 mellom fasit og tipping: %.5f\n' % (
        calc_R2(fasit[fasit_key].loc[tipping_df.index[0]:], tipping_df[:fasit[fasit_key].index[-1]])))

    print('Fasit:\n', fasit[fasit_key][-4:])
    print('\nModdelert/Tippet:\n', tipping_df[-5:])

    if fasit_key[-3:] == '105':
        color_tipping = 'blue'
    elif fasit_key[-3:] == '132':
        color_tipping = 'lightblue'


def show_result_jupyter(show_result_input):
    """This function prints out and plots the results from the regression."""
    fasit_key, ant_kandidater, max_p, fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_df, reg_end, reg_period, ant_break_long, nb_weeks_tipping, read_start = show_result_input
    plt.interactive(False)
    reg_start = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") - Timedelta(days=reg_period * 7)).strftime(
        '%Y.%m.%d')
    print('\n-----------------------------------------------------------------------')
    print('RESULTATER FOR %s\n' % fasit_key)
    print('Regresjonsperiode brukt til setup for siste tipping: %s til: %s.' % (read_start, reg_end))
    print('Regresjonsperiode brukt på modellen for siste tipping: %s til: %s.' % (reg_start, reg_end))
    print('Valgte %.2f kandidater til regresjonen utifra korrelasjon med fasitserien.' % (ant_kandidater))
    print('Valgte så ut de med p-value < %.5f, som var %i stk.' % (max_p, len(long_results.pvalues)))
    print('Antall stopp av loopen som luker ut for høye p pga minimum antall serier i den lange regresjonen: %i/%i' % (
    ant_break_long, nb_weeks_tipping))
    print('R2 for regresjonen (kort periode): %.5f' % r2_modelled)
    # start_tipping = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") - Timedelta(days=7*(nb_weeks_tipping-2))).strftime('%Y.%m.%d')
    print('R2 mellom fasit og tipping: %.5f\n' % (
        calc_R2(fasit[fasit_key].loc[tipping_df.index[0]:], tipping_df[:fasit[fasit_key].index[-1]])))

    print('Fasit:\n', fasit[fasit_key][-4:])
    print('\nModdelert/Tippet:\n', tipping_df[-5:])

    if fasit_key[-3:] == '105':
        color_tipping = 'blue'
    elif fasit_key[-3:] == '132':
        color_tipping = 'lightblue'

    # Plot with regression:
    plt.figure(figsize=(16, 10))
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        plt.plot(fasit[fasit_key].loc[:reg_end], color='k', linewidth=2.0, label='fasit')
    else:
        plt.plot(fasit[fasit_key].loc[:], color='k', linewidth=2.0, label='fasit')
    plt.plot(short_results.predict(df_tot[chosen_p].loc[reg_start:reg_end]), color='orange',
             label='regresjon på historie(kort periode)')
    plt.plot(long_results.predict(df_tot[chosen_p].loc[:reg_start]), color='cyan',
             label='regresjon på historie (lang periode)')
    plt.plot(short_results.predict(df_tot[chosen_p].loc[:reg_start]), color='deeppink',
             label='modell på historie (kort periode)')
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
            plt.plot(df_tot[key] * sfac)  # , marker='o')
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


def write_V_SMG_Regresjon(df_tot, results, chosen_p, fasit_key, r2_modelled, MagKap_mag=False):
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
        expression += str(
            sum_av_vekter.format(alle_vekter='F' + '+F'.join([str(tall + 1) for tall in range(len(chosen_p))])))
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


def regression(df_tot, fasit_key, chosen, max_p):
    """This function runs several regressions so that the optimal set of series in the result model are given in return. Each time the regression is run, the series with the highest (worst) p-value is dropped out of the chosen list, until the highest p in p-values in the results equal max_p, or only 6 series are left.
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
    while max(results.pvalues) > max_p or len(results.pvalues) >= 25:
        if len(results.pvalues) <= 1:
            ant_break = 1  # count
            break
        chosen_p.remove(results.pvalues.idxmax())  # updating the chosen list
        results = sm.OLS(df_tot[fasit_key], df_tot[chosen_p]).fit()  # regression
    return results, chosen_p, ant_break


def make_estimate(variable, region, auto_input, fasit_key, reg_period, max_p, ant_kandidater):
    df_week, MagKap, period, forecast_time, read_start = auto_input
    reg_end = (pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(days=7)).strftime('%Y.%m.%d')
    fasit = period.read([fasit_key]).loc[:reg_end]

    # df_tot = deletingNaNs(df_week.loc[:(pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") + Timedelta(days=7)).strftime('%Y.%m.%d')]).join(fasit)
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        last_forecast = forecast_time
    else:
        last_forecast = reg_end
    df_cleaned = deletingNaNs(df_week.loc[:last_forecast])
    df_tot = df_cleaned.join(fasit)

    #########################################################################################
    nb_weeks_tipping = 10  # number of weeks to do tipping back in time
    start_tipping = 7 * nb_weeks_tipping
    reg_end_new = (pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(
        days=start_tipping)).strftime('%Y.%m.%d')  # 6*52
    forecast_time_new = True
    tipping_times = []
    tipping_values = []
    ant_break_long = 0
    ant_break_short = 0
    ant_kandidater_error = False
    max_input_error = False
    while forecast_time_new != last_forecast:
        forecast_time_new = (
                pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") + Timedelta(days=7)).strftime(
            '%Y.%m.%d')
        reg_start = (pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") - Timedelta(
            days=reg_period * 7)).strftime('%Y.%m.%d')
        df_tot_new = df_tot[:reg_end_new]
        r2_original = pd.Series()
        for key in df_cleaned:
            if variable == 'tilsig':
                if df_tot_new[key].mean() == 0:
                    print('passed for ', key, 'mean = 0')
                    pass
                else:
                    scalefac = df_tot_new[fasit_key].mean() / df_tot_new[key].mean()
                    r2_original[key] = calc_R2(df_tot_new[fasit_key], df_tot_new[key] * scalefac)
            elif variable == 'magasin':
                r2_original[key] = calc_R2(df_tot_new[fasit_key], df_tot_new[key])

        # Chosing the chosen number of best r2 keys
        sorted_r2 = r2_original.sort_values(ascending=False)
        max_input_series = 196
        maks = max_input_series if len(sorted_r2) >= max_input_series else len(sorted_r2)
        if ant_kandidater > maks:
            ant_kandidater_error = '\nFeilmelding: Ønsket antall kandidater overskrider maks (%i).' % maks
            chosen_r2 = list(sorted_r2.axes[0][:maks])
        else:
            chosen_r2 = list(sorted_r2.axes[0][:ant_kandidater])
        # Regresjon
        long_results, chosen_p, ant_break = regression(df_tot_new, fasit_key, chosen_r2, max_p)
        ant_break_long += ant_break

        short_results, chosen_p, ant_break_short = regression(df_tot_new.loc[reg_start:reg_end_new], fasit_key,
                                                              chosen_p, 1)
        r2_modelled = calc_R2(df_tot_new.loc[reg_start:reg_end_new][fasit_key],
                              short_results.predict(df_tot_new.loc[reg_start:reg_end_new][chosen_p]))
        ant_break_short += ant_break
        prediction = short_results.predict(df_cleaned[chosen_p]).loc[reg_end_new:forecast_time_new]
        # print(df_cleaned['/Glom-Krv.NO1.EMPS..-U9104S0BT0132'].loc[reg_end_new:forecast_time_new])
        reg_end_new = (pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") + Timedelta(days=7)).strftime(
            '%Y.%m.%d')
        tipping_times.append(prediction.index[-1])
        # print('tipping: ', prediction.index[-1])
        tipping_values.append(prediction[-1])
        ####################################################################################
    tipping_df = pd.Series(tipping_values, index=tipping_times)
    if ant_kandidater_error:
        print(ant_kandidater_error)

    return fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_df, reg_end, reg_period, ant_break_long, nb_weeks_tipping, read_start


def make_fasit_key(variable, region):
    if ('N' in region) and (variable == 'tilsig'):
        fasit_key = '/Norg-No' + region[-1] + '.Fasit.....-U9100S0BT0105'
    elif ('S' in region) and (variable == 'tilsig'):
        fasit_key = '/Sver-Se' + region[-1] + '.Fasit.....-U9100S0BT0105'
    if ('N' in region) and (variable == 'magasin'):
        fasit_key = '/Norg-NO' + region[-1] + '.Fasit.....-U9104A5R-0132'
    elif ('S' in region) and (variable == 'magasin'):
        fasit_key = '/Sver-SE' + region[-1] + '.Fasit.....-U9104A5R-0132'
    return fasit_key


def make_estimate_while_looping(variable, region, auto_input, reg_period, max_p, ant_kandidater):
    fasit_key = make_fasit_key(variable, region)
    df_week, MagKap, period, forecast_time, read_start = auto_input
    reg_end = (pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(days=7)).strftime('%Y.%m.%d')
    fasit = period.read([fasit_key]).loc[:reg_end]

    # df_tot = deletingNaNs(df_week.loc[:(pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") + Timedelta(days=7)).strftime('%Y.%m.%d')]).join(fasit)
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        last_forecast = forecast_time
    else:
        last_forecast = reg_end
    df_cleaned = deletingNaNs(df_week.loc[:last_forecast])
    df_tot = df_cleaned.join(fasit)

    #########################################################################################
    nb_weeks_tipping = 10  # number of weeks to do tipping back in time
    start_tipping = 7 * nb_weeks_tipping
    reg_end_new = (pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(
        days=start_tipping)).strftime('%Y.%m.%d')  # 6*52
    forecast_time_new = True
    tipping_times = []
    tipping_values = []
    ant_break_long = 0
    ant_break_short = 0
    ant_kandidater_error = False
    max_input_error = False
    while forecast_time_new != last_forecast:
        forecast_time_new = (
                pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") + Timedelta(days=7)).strftime(
            '%Y.%m.%d')
        reg_start = (pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") - Timedelta(
            days=reg_period * 7)).strftime('%Y.%m.%d')
        df_tot_new = df_tot[:reg_end_new]
        r2_original = pd.Series()
        for key in df_cleaned:
            if variable == 'tilsig':
                if df_tot_new[key].mean() == 0:
                    print('passed for ', key, 'mean = 0')
                    pass
                else:
                    scalefac = df_tot_new[fasit_key].mean() / df_tot_new[key].mean()
                    r2_original[key] = calc_R2(df_tot_new[fasit_key], df_tot_new[key] * scalefac)
            elif variable == 'magasin':
                r2_original[key] = calc_R2(df_tot_new[fasit_key], df_tot_new[key])

        # Chosing the chosen number of best r2 keys
        sorted_r2 = r2_original.sort_values(ascending=False)
        max_input_series = 196
        if ant_kandidater > max_input_series:
            ant_kandidater_error = '\nFeilmelding: Ønsket antall kandidater overskrider maks (%i), passed.' % maks
            pass
            chosen_r2 = list(sorted_r2.axes[0][:maks])
        else:
            chosen_r2 = list(sorted_r2.axes[0][:ant_kandidater])
        # Regresjon
        long_results, chosen_p, ant_break = regression(df_tot_new, fasit_key, chosen_r2, max_p)
        ant_break_long += ant_break

        short_results, chosen_p, ant_break_short = regression(df_tot_new.loc[reg_start:reg_end_new], fasit_key,
                                                              chosen_p, 1)
        r2_modelled = calc_R2(df_tot_new.loc[reg_start:reg_end_new][fasit_key],
                              short_results.predict(df_tot_new.loc[reg_start:reg_end_new][chosen_p]))
        ant_break_short += ant_break
        prediction = short_results.predict(df_cleaned[chosen_p]).loc[reg_end_new:forecast_time_new]
        # print(df_cleaned['/Glom-Krv.NO1.EMPS..-U9104S0BT0132'].loc[reg_end_new:forecast_time_new])
        reg_end_new = (pd.to_datetime(time.strftime(reg_end_new), format="%Y.%m.%d") + Timedelta(days=7)).strftime(
            '%Y.%m.%d')
        tipping_times.append(prediction.index[-1])
        # print('tipping: ', prediction.index[-1])
        tipping_values.append(prediction[-1])
        ####################################################################################
    tipping_df = pd.Series(tipping_values, index=tipping_times)
    r2_tippet = calc_R2(fasit[fasit_key].loc[tipping_df.index[0]:], tipping_df[:fasit[fasit_key].index[-1]])
    if ant_kandidater_error:
        print(ant_kandidater_error)
    r_samlet = (r2_modelled * 0.5 + r2_tippet * 0.5)
    ant_serier = len(chosen_p) if len(chosen_p) >= 6 else 6
    print(ant_kandidater, ant_serier, r2_modelled, r2_tippet, r_samlet, reg_period, max_p)
    return ant_kandidater, ant_serier, r2_modelled, r2_tippet, r_samlet, reg_period, max_p


def get_default_variables(variable, region, reg_period, max_p, ant_kandidater):
    print('\n')
    default = get_default(variable, region) #default=[fasit_key, reg_period, max_p, ant_kandidater]
    fasit_key = default[0]
    if ((reg_period, max_p, ant_kandidater) != default[1:]) and (reg_period, max_p, ant_kandidater) != (False,False,False):
        print('ENDRET INPUT:')
    if (ant_kandidater!=False) and (ant_kandidater != default[3]):
        print('ant_kandidater: %.2f'%ant_kandidater)
    else:
        ant_kandidater = default[3]
    if (max_p!=False) and (max_p != default[2]):
        print('max_p: %.5f'%max_p)
    else:
        max_p = default[2]
    if (0 < reg_period < 200) and (reg_period != default[1]):
        print('reg_period: %i uker'%reg_period)
    elif reg_period > 200:
        print('Antall uker (reg_period) valgt for den korte, siste regresjonen var for lang.')
    #elif (200 < reg_period <= 0):
    #    raise Exception('reg_period = {} er for høy eller lav, prøv med: 200 >= reg_period > 0.'.format(reg_period))
    else:
        reg_period = default[1]
    return fasit_key, reg_period, max_p, ant_kandidater



def make_estimate_and_write(variable, region, auto_input, reg_period=False, max_p=False, ant_kandidater=False):
    fasit_key, reg_period, max_p, ant_kandidater = get_default_variables(variable, region, reg_period, max_p,
                                                                         ant_kandidater)
    df_week, MagKap, period, forecast_time = auto_input
    tipping_output = make_estimate(variable, region, auto_input, fasit_key, reg_period, max_p, ant_kandidater)
    [fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_df, reg_end,
     reg_period, ant_break_long, nb_weeks_tipping] = tipping_output
    # write to SMG:
    write_SMG_regresjon(variable, region, tipping_df)
    # write to SMG, virtual:
    write_V_SMG_Regresjon(df_tot, short_results, chosen_p, fasit_key, r2_modelled, MagKap)
    show_results_input = [fasit_key, ant_kandidater, max_p, fasit, long_results, short_results, df_tot, chosen_p,
                          chosen_r2, r2_modelled, prediction, tipping_df, reg_end, reg_period, ant_break_long,
                          nb_weeks_tipping]
    return show_results_input
