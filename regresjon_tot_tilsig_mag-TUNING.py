from shyft.api import utctime_now  # To time the reading from SMG
from regression_modules import *
start_time = utctime_now()

auto_input = {}
auto_input['tilsig'] = read_and_setup('tilsig')
auto_input['magasin'] = read_and_setup('magasin')

columns = ['ant_kandidater', 'ant_serier', 'r2_modelled', 'r2_tippet', 'r2_samlet', 'reg_period', 'max_p']
# Initializing
max_p = 0.025
reg_period = 208  # Finn hele perioden

#['magasin', 'tilsig']
for variable in ['magasin', 'tilsig']:

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
        # Første loop: Tuner antall kandidater som gir best R2 samlet
        start_time = time.time()
        df_ant_kandidater = pd.DataFrame(columns=columns)
        for antall in range(min_kandidater, max_kandidater+1, 2):
            output = make_estimate_while_looping(variable, region, auto_input[variable], reg_period, max_p, antall)
            df_ant_kandidater = df_ant_kandidater.append(
                {columns[0]: output[0], columns[1]: output[1], columns[2]: output[2], columns[3]: output[3],
                 columns[4]: output[4], columns[5]: output[5], columns[6]: output[6]}, ignore_index=True)
        idx_max = df_ant_kandidater.r2_samlet.idxmax(skipna=True)
        ant_kandidater_beste = int(df_ant_kandidater.ant_kandidater.values[idx_max])
        print('Beste ant_kandidater loop 1: ', ant_kandidater_beste)
        end_time = time.time()
        print('Time to run loop 1: ', end_time - start_time)

        # Andre loop: tuner lengden på den korte regresjonen som gir best R2 samlet
        start_time = time.time()
        df_reg_period = pd.DataFrame(columns=columns)
        for period in range(min_weeks, max_weeks+1, 2):
            period = int(period)
            output = make_estimate_while_looping(variable, region, auto_input[variable], period, max_p, ant_kandidater_beste)
            df_reg_period = df_reg_period.append(
                {columns[0]: output[0], columns[1]: output[1], columns[2]: output[2], columns[3]: output[3],
                 columns[4]: output[4], columns[5]: output[5], columns[6]: output[6]}, ignore_index=True)
        idx_max = df_reg_period.r2_samlet.idxmax(skipna=True)
        reg_period_beste = int(df_reg_period.reg_period.values[idx_max])
        print('Beste reg_period loop 2: ', reg_period_beste)
        end_time = time.time()
        print('Time to run loop 2: ', end_time - start_time)

        # Tredje loop: tuner valget av max p-verdi som gir best R2 samlet
        # df_max_p = pd.DataFrame(columns=columns)
        # for max_p in np.linspace(0.001,0.015,5):
        #    output = make_estimate_while_looping(variable, region, auto_input[variable], reg_period, max_p, ant_kandidater)
        #    df_max_p = df_max_p.append({columns[0]:output[0], columns[1]:output[1], columns[2]:output[2], columns[3]:output[3], columns[4]:output[4], columns[5]:output[5], columns[6]:output[6]},ignore_index=True)
        # idx_max = df_max_p.r2_samlet.idxmax(skipna=True)
        # max_p = df_max_p.max_p.values[idx_max]
        # print('Valgte max_p til å være: ', max_p)

        # FINAL RESULTS AFTER TUNING
        # Update with the input which gives the best R2 samlet
        df_all_methods = pd.concat([df_ant_kandidater, df_reg_period], ignore_index=True, sort=False)
        idx_max = df_all_methods.r2_samlet.idxmax(skipna=True)
        ant_kandidater_beste = int(df_all_methods.ant_kandidater.values[idx_max])
        reg_period_beste = df_all_methods.reg_period.values[idx_max]
        max_p = df_all_methods.max_p.values[idx_max]
        write_input_variables_to_file(region, variable, max_p, ant_kandidater_beste, reg_period_beste)
        # Show results
        show_result_input = make_estimate_and_write(variable, region, auto_input[variable], backup=False)
        show_result(show_result_input)
        print('\nRegresjonen med tuning tok %.0f minutter. \n' % ((utctime_now() - start_time_loop) / 60))

print('---------------------------------------------------------------')
print('                         SLUTT                                 ')
print('---------------------------------------------------------------')
print('\nScriptet brukte totalt %.0f minutter. \n' % ((utctime_now() - start_time) / 60))

