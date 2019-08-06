from regression_modules import *

auto_input = {}
auto_input['tilsig'] = read_and_setup('tilsig')
auto_input['magasin'] = read_and_setup('magasin')


run_regression(auto_input, loop=True)
#def run_regression(auto_input,
#               variables: list = ['magasin', 'tilsig'],
#               regions: list = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4'],
#               jupyter: bool = False,
#               backup: bool = False,
#               loop: bool = False) -> None:
