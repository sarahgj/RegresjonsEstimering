import socket
from datetime import datetime
import logging
import os
import traceback

from modules.regression_modules import *

start_time = utctime_now()
sti_til_logfilområde = '../logging'

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

    run_regression(auto_input)
    # def run_regression(auto_input,
    #               variables: list = ['magasin', 'tilsig'],
    #               regions: list = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4'],
    #               jupyter: bool = False,
    #               backup: bool = False,
    #               loop: bool = False) -> None:
    logging.info('\nThe script ran successfully and used a total of %.0f minutes\n' %((utctime_now() - start_time)/60))


except Exception as e:
    trace = traceback.format_exc()
    logging.exception(e)
    logging.error(e)
    logging.error(trace)
