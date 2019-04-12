from regression_modules import *

auto_input_tilsig = read_and_setup('Tilsig')
#auto_input_magasin = read_and_setup('Magasin')

all_regions = ['NO1','NO2','NO3','NO4','NO5','SE1','SE2','SE3','SE4']
norge = ['NO1','NO2','NO3','NO4','NO5']
sverige = ['SE2','SE3','SE4']

for region in ['NO1']:
    #Tilsig
    fasit_key, max_p, max_r2, regPeriod = input_Tilsig(region)
    regresjonstipping('Tilsig', region, auto_input_tilsig, fasit_key, max_p, max_r2, regPeriod)
    #Magasin
    #fasit_key, max_p, max_r2, regPeriod = input_Magasin(region)
    #regresjonstipping('Magasin', region, auto_input_magasin, fasit_key, max_p, max_r2, regPeriod)