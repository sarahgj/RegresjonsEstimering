import socket
from datetime import datetime
import logging
import os
import traceback
from shyft.api import utctime_now

from regression_modules import *

start_time = utctime_now()
sti_til_logfilområde = ''

log_file = os.path.join(sti_til_logfilområde,  # folder with log files
                            '{}#{}#{}#{}.log'.format(
                            os.path.splitext(os.path.basename(__file__))[0],  # script file name
                            socket.gethostname().lower(),  # host_name
                            datetime.now().strftime('%Y%m%dT%H%M%S'),  # timestamp
                            os.getpid()  # process ID
                        ))
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info('autojob started.')


try:
    auto_input = {}
    auto_input['tilsig'] = read_and_setup('tilsig')
    auto_input['magasin'] = read_and_setup('magasin')

    for variable in ['magasin', 'tilsig']:
        df_week, MagKap, period, forecast_time, read_start = auto_input[variable]
        reg_end = (pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(days=7)).strftime(
            '%Y.%m.%d')
        if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
            last_forecast = forecast_time
        else:
            last_forecast = reg_end
        df_cleaned = deletingNaNs(df_week.loc[:last_forecast])
        for region in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4']:
            start_time_loop = utctime_now()
            
            fasit, fasit_key = make_fasit(variable, region, reg_end, period)
            if fasit[fasit_key].isnull().any():
                print('OBS: Det mangler verdier på fasiten! Går videre til neste region i loopen..')
                continue
            
            sorted_r2 = get_R2_sorted(variable, df_cleaned, fasit, fasit_key)
            short_period, max_p, ant_kandidater, input_file = get_input_variables_from_file(variable, region, backup=False)
            chosen_r2 = sorted_r2[:ant_kandidater]
            input1 = make_estimate(df_cleaned, fasit, fasit_key, last_forecast, short_period, max_p, chosen_r2, loop=False)
            fasit, long_results, short_results, df_tot, chosen_p, chosen_r2, r2_modelled, prediction, tipping_df, short_period, nb_weeks_tipping = input1
            input2 = fasit_key, ant_kandidater, max_p, reg_end, read_start
            # write to SMG:
            write_SMG_regresjon(variable, region, tipping_df)
            # write to SMG, virtual:
            write_V_SMG_Regresjon(df_tot, short_results, chosen_p, fasit_key, r2_modelled, MagKap)
            show_result(input1, input2)
    logging.info('\nThe script ran successfully and used a total of %.0f minutes\n' %((utctime_now() - start_time)/60))


except Exception as e:
    trace = traceback.format_exc()
    logging.exception(e)
    logging.error(e)
    logging.error(trace)
