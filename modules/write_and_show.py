import sys
import pandas as pd
import time
import pytz
import matplotlib.pyplot as plt
from pandas import Timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from statkraft.ssa.timeseriesrepository import TimeSeriesRepositorySmg
from statkraft.ssa.environment import SMG_PROD
from statkraft.ssa.adapter import ts_from_pandas_series

from modules import r2_and_regression as reg

#Global variables
today = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')  # now
#max_final_numb_kandidater = 25
#max_input_series = 196
#nb_weeks_tipping = 10  # number of weeks to do tipping back in time
#tz = pytz.timezone('Etc/GMT-1')
#columns = ['ant_kandidater', 'ant_serier', 'r2_modelled', 'r2_tippet', 'r2_samlet', 'short_period', 'max_p']
#first_period = 216  # Length of the long regression in weeks
#min_kandidater = 6


def show_result(input1, input2, variable_file=False):
    """This function prints out and plots the results from the regression."""
    fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_ps, short_period, nb_weeks_tipping = input1
    fasit_key, ant_kandidater, max_p, reg_end, read_start = input2
    plt.interactive(False)
    reg_start = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") - Timedelta(days=short_period * 7)).strftime(
        '%Y.%m.%d')
    print('\n-----------------------------------------------------------------------')
    print('RESULTATER FOR %s\n' % fasit_key)
    print('Regresjonsperiode fra: %s til: %s.' % (reg_start, reg_end))
    if variable_file:
        print('Input variablene (reg_period og ant_kandidater) ble hentet fra: ', variable_file)
    else:
        print('Input variablene (reg_period og ant_kandidater) ble funnet ved tuning.')
    print('Valgte %.2f kandidater til regresjonen utifra korrelasjon med fasitserien.' % (ant_kandidater))
    print('Valgte så ut de med p-value < %.5f, som var %i stk.' % (max_p, len(long_results.pvalues)))
    print('R2 for regresjonen (kort periode): %.5f' % r2_modelled)
    print('R2 mellom fasit og tipping: %.5f\n' % (reg.calc_R2(fasit[fasit_key].loc[tipping_ps.index[0]:], tipping_ps[:fasit[fasit_key].index[-1]])))
    print('Fasit:\n', fasit[fasit_key][-4:])
    print('\nModdelert/Tippet:\n', tipping_ps[-5:])


def show_result_jupyter(input1, input2, variable_file=False):
    fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, r2_modelled_long, prediction, tipping_ps, short_period, nb_weeks_tipping = input1
    fasit_key, ant_kandidater, max_p, reg_end, read_start = input2
    """This function prints out and plots the results from the regression."""
    plt.interactive(False)
    reg_start = (pd.to_datetime(time.strftime(reg_end), format="%Y.%m.%d") - Timedelta(days=short_period * 7)).strftime(
        '%Y.%m.%d')
    print('\n-----------------------------------------------------------------------')
    print('RESULTATER FOR %s\n' % fasit_key)
    print('Regresjonsperiode brukt til setup for siste tipping: %s til: %s.' % (read_start, reg_end))
    if variable_file:
        print('Input variablene (reg_period og ant_kandidater) ble hentet fra: ', variable_file)
    else:
        print('Input variablene (reg_period og ant_kandidater) ble funnet ved tuning.')
    print('Regresjonsperiode brukt på modellen for siste tipping: %s til: %s, satt til %d uker.' % (
    reg_start, reg_end, int(short_period)))
    print('Valgte %d kandidater til regresjonen utifra korrelasjon med fasitserien.' % (int(ant_kandidater)))
    print('Valgte så ut de med p-value < %.5f, som var %i stk.' % (max_p, len(long_results.pvalues)))
    print('R2 for regresjonen (kort periode): %.5f' % r2_modelled)
    print('R2 mellom fasit og tipping: %.5f\n' % (reg.calc_R2(fasit[fasit_key].loc[tipping_ps.index[0]:], tipping_ps[:fasit[fasit_key].index[-1]])))
    print('Fasit:\n', fasit[fasit_key][-4:])
    print('\nModdelert/Tippet:\n', tipping_ps[-5:])
    if fasit_key[-3:] == '105':
        color_tipping = 'blue'
    elif fasit_key[-3:] == '132':
        color_tipping = 'lightblue'
    else:
        sys.exit("The fasit key should end with 105 or 132")

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
    plt.plot(tipping_ps, label='tipping', color=color_tipping)  # , marker='o')
    plt.title('Regresjon for: %s' % fasit_key)
    plt.legend()

    # Plot just prediction:
    plt.figure(figsize=(16, 10))
    plt.plot(fasit[fasit_key].loc[tipping_ps.index[0]:], color='k', linewidth=2.0, label='fasit')
    plt.plot(tipping_ps, label='tipping', color=color_tipping)  # , marker='o')
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
    plt.plot(tipping_ps, label='tipping', color=color_tipping)  # , marker='o')
    plt.title('Regresjonsserier for: %s' % fasit_key)
    plt.legend()
    plt.show()


def write_SMG_regresjon(variable, region, ps):
    """This function writes pandas series (pandas series) to smg series (ts) chosen according to the chosen region."""
    global ts
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
    ps.meta_info = minfo[0]
    print(ps)
    print(ps[-1])

    # convert the Pandas series to a time series
    time_series_to_write = ts_from_pandas_series(ps, filter_out_nans=True)

    # writing
    result = smg.write([time_series_to_write])


def write_V_SMG_Regresjon(results, chosen_p, fasit_key, r2_modelled, r2_modelled_long, short_period_beste, MagKap_mag=False):
    start_time = time.time()
    now = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')
    expression = str('! Sist oppdatert {}\n!R2 (modell vs fasit) for periode tilbake til 08.06.2015: {:0.6f}\n!R2 (modell vs fasit) for periode tilbake {} uker: {:0.6f}\n'.format(now, r2_modelled_long, short_period_beste, r2_modelled))
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
    # print('-----------------------------------------------------------------------')
    # print(expression)
    # print('-----------------------------------------------------------------------')
    smg.update_virtual({info[0]: expression})
    end_time = time.time()
    print('Time to run write_V_SMG_Regresjon: ', end_time - start_time)



def write_input_variables_to_file(region, variable, max_p, ant_kandidater, reg_period):
    output_file = '../imports/input_variables_from_tuning.txt'
    string2find = '{:3} {:7}'.format(region, variable)
    with open(output_file, 'r') as file:
        data = file.readlines()
        i = 0
        for line in data:
            if line.startswith(string2find):
                now = pd.to_datetime(time.strftime("%Y.%m.%d %H:%M"), format="%Y.%m.%d %H:%M", errors='ignore')
                data[i] = '{:10s} {:5.3f} {:3d} {:3d} {}\n'.format(string2find, max_p, int(ant_kandidater), int(reg_period), now)
                break
            i += 1
    with open(output_file, 'w') as file:
        file.writelines(data)
