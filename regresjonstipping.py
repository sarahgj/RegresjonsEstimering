import socket
from datetime import datetime
import logging
import os
import traceback

from regression_modules import *
from chosen_input_variables import *

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

    auto_input_tilsig = read_and_setup('Tilsig')
    auto_input_magasin = read_and_setup('Magasin')

    all_regions = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4']
    norge = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
    sverige = ['SE1', 'SE2', 'SE3', 'SE4']

    for region in all_regions:
        # Tilsig
        fasit_key, max_p, max_r2, regPeriod = input_Tilsig(region)
        show_result_input = make_estimate_and_write('Tilsig', region, auto_input_tilsig, fasit_key, max_p, max_r2,
                                                    regPeriod)
        # show_result(show_result_input)
        # Magasin
        fasit_key, max_p, max_r2, regPeriod = input_Magasin(region)
        show_result_input = make_estimate_and_write('Magasin', region, auto_input_magasin, fasit_key, max_p, max_r2,
                                                    regPeriod)
        # show_result(show_result_input)
    logging.info('Script ran successfully')


except Exception as e:
    trace = traceback.format_exc()
    logging.exception(e)
    logging.error(e)
    logging.error(trace)
