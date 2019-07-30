from shyft.api import utctime_now  # To time the reading from SMG
from regression_modules import *
start_time = utctime_now()

auto_input = {}
auto_input['tilsig'] = read_and_setup('tilsig')
auto_input['magasin'] = read_and_setup('magasin')

columns = ['ant_kandidater', 'ant_serier', 'r2_modelled', 'r2_tippet', 'r2_samlet', 'short_period', 'max_p']
# Initializing
max_p = 0.025
first_period = 208  # Finn hele perioden

#['magasin', 'tilsig']
for variable in ['magasin', 'tilsig']:
    df_week, MagKap, period, forecast_time, read_start = auto_input[variable]
    reg_end = (pd.to_datetime(time.strftime(forecast_time), format="%Y.%m.%d") - Timedelta(days=7)).strftime('%Y.%m.%d')
    if (0 <= today.weekday() <= 1) or (today.weekday() == 2 and today.hour < 14):  # True for tipping
        last_forecast = forecast_time
    else:
        last_forecast = reg_end
    df_cleaned = deletingNaNs(df_week.loc[:last_forecast])
    if variable == 'tilsig':
        print('---------------------------------------------------------------')
        print('                        TILSIG                                 ')
        print('---------------------------------------------------------------')
        max_kandidater = 196
        min_kandidater = 5
        print('variable = {}'.format(variable))
    elif variable == 'magasin':
        print('---------------------------------------------------------------')
        print('                        MAGASIN                                ')
        print('---------------------------------------------------------------')
        max_kandidater = 135
        min_kandidater = 5
        print('variable = {}'.format(variable))
    else:
        print("variable in line 14 must be either 'tilsig' or 'magasin'")
        sys.exit(1)
    max_weeks = 208
    min_weeks = 10
    print('max ant. kandidater: {}, min ant. kandidater: {}'.format(max_kandidater, min_kandidater))
    print('max ant. uker: {}, min ant. uker: {}'.format(max_weeks, min_weeks))

    # ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4']
    for region in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4']:
        start_time_loop = utctime_now()
        fasit, fasit_key = make_fasit(variable, region, reg_end, period)
        
            
        if fasit[fasit_key].isnull().any():
            print('OBS: Det mangler verdier på fasiten! Går videre til neste region i loopen..')
            continue
            
        sorted_r2 = get_R2_sorted(variable, df_cleaned, fasit, fasit_key)


        # Første loop: Tuner antall kandidater som gir best R2 samlet
        start_time = time.time()
        df_ant_kandidater = pd.DataFrame(columns=columns)
        for antall in range(min_kandidater, max_kandidater+1, 1):
            if antall > len(sorted_r2):
                chosen_r2 = sorted_r2
            else:
                chosen_r2 = sorted_r2[:antall]
            output = make_estimate(df_cleaned, fasit, fasit_key, last_forecast, first_period, max_p, chosen_r2, loop=True)
            df_ant_kandidater = df_ant_kandidater.append(
                {columns[0]: output[0], columns[1]: output[1], columns[2]: output[2], columns[3]: output[3],
                 columns[4]: output[4], columns[5]: output[5], columns[6]: output[6]}, ignore_index=True)
            if antall > len(sorted_r2):
                print('Feilmelding: Ønsket antall kandidater overskrider maks (%i).\n' % len(sorted_r2))
                break
        idx_max = df_ant_kandidater.r2_samlet.idxmax(skipna=True)
        ant_kandidater_beste = int(df_ant_kandidater.ant_kandidater.values[idx_max])
        print('Beste ant_kandidater loop 1: ', ant_kandidater_beste)
        end_time = time.time()
        print('Time to run loop 1: ', end_time - start_time)

        # Andre loop: tuner lengden på den korte regresjonen som gir best R2 samlet
        start_time = time.time()
        df_short_period = pd.DataFrame(columns=columns)
        for short_period in range(min_weeks, max_weeks+1, 1):
            short_period = int(short_period)
            final_chosen_r2 = sorted_r2[:ant_kandidater_beste]
            output = make_estimate(df_cleaned, fasit, fasit_key, last_forecast, short_period, max_p, final_chosen_r2, loop=True)
            df_short_period = df_short_period.append(
                {columns[0]: output[0], columns[1]: output[1], columns[2]: output[2], columns[3]: output[3],
                 columns[4]: output[4], columns[5]: output[5], columns[6]: output[6]}, ignore_index=True)
        idx_max = df_short_period.r2_samlet.idxmax(skipna=True)
        short_period_beste = int(df_short_period.short_period.values[idx_max])
        print('Beste short_period loop 2: ', short_period_beste)
        end_time = time.time()
        print('Time to run loop 2: ', end_time - start_time)

        # FINAL RESULTS AFTER TUNING
        # Update with the input which gives the best R2 samlet
        df_all_methods = pd.concat([df_ant_kandidater, df_short_period], ignore_index=True, sort=False)
        idx_max = df_all_methods.r2_samlet.idxmax(skipna=True)
        ant_kandidater_beste = int(df_all_methods.ant_kandidater.values[idx_max])
        chosen_r2_beste = sorted_r2[:ant_kandidater_beste]
        short_period_beste = df_all_methods.short_period.values[idx_max]
        write_input_variables_to_file(region, variable, max_p, ant_kandidater_beste, short_period_beste)
        # Show results
        input1 = make_estimate(df_cleaned, fasit, fasit_key, last_forecast, short_period_beste, max_p, chosen_r2_beste, loop=False)
        input2 = fasit_key, ant_kandidater_beste, max_p, reg_end, read_start
        show_result(input1, input2)
        print('\nRegresjonen med tuning tok %.0f minutter. \n' % ((utctime_now() - start_time_loop) / 60))

print('---------------------------------------------------------------')
print('                         SLUTT                                 ')
print('---------------------------------------------------------------')
print('\nScriptet brukte totalt %.0f minutter. \n' % ((utctime_now() - start_time) / 60))

