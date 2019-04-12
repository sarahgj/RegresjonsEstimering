def input_Tilsig(region):
    if 'N' in region:
        fasit_key = '/Norg-No' + region[-1] + '.Fasit.....-U9100S0BT0105'
    elif 'S' in region:
        fasit_key = '/Sver-Se' + region[-1] + '.Fasit.....-U9100S0BT0105'
    if region == 'NO1':
        max_p = 0.025
        max_r2 = 100
        regPeriod = 52  # 1 year
    elif region == 'NO2':
        max_p = 0.025
        max_r2 = 100
        regPeriod = 52 * 2.5  # 2.5 years
    elif region == 'NO3':
        max_p = 0.025
        max_r2 = 100
        regPeriod = 52 * 2.5  # 2.5 years
    elif region == 'NO4':
        max_p = 0.025
        max_r2 = 100
        regPeriod = 52  # 1 year
    elif region == 'NO5':
        max_p = 0.025
        max_r2 = 100
        regPeriod = 52  # 1 year
    elif region == 'SE1':
        max_p = 0.025
        max_r2 = 100
        regPeriod = 52  # 1 year
    elif region == 'SE2':
        max_p = 0.0025
        max_r2 = 100
        regPeriod = 52  # 1 year
    elif region == 'SE3':
        max_p = 0.025
        max_r2 = 100
        regPeriod = 52  # 1 year
    elif region == 'SE4':
        max_p = 0.025
        max_r2 = 100
        regPeriod = 78  # 1.5 year
    print('INPUT TILSIG', region, ':')
    print('Høyeste p-verdi (max_p): ', max_p)
    print('Antall utvalgte serier basert på høyeste R2 som brukes (max_r2): ', max_r2)
    print('Antall uker den "korte" regresjonen kjøres (regPeriod): ', regPeriod, '\n')
    return fasit_key, max_p, max_r2, regPeriod


def input_Magasin(region):
    if 'N' in region:
        fasit_key = '/Norg-NO' + region[-1] + '.Fasit.....-U9104A5R-0132'
    elif 'S' in region:
        fasit_key = '/Sver-SE' + region[-1] + '.Fasit.....-U9104A5R-0132'
    if region == 'NO1':
        max_p = 0.0025
        max_r2 = 30
        regPeriod = 52  # 1 year
    elif region == 'NO2':
        max_p = 0.0025
        max_r2 = 30
        regPeriod = 52 * 2.5  # 2.5 years
    elif region == 'NO3':
        max_p = 0.0025
        max_r2 = 30
        regPeriod = 52 * 2.5  # 2.5 years
    elif region == 'NO4':
        max_p = 0.0025
        max_r2 = 30
        regPeriod = 52  # 1 year
    elif region == 'NO5':
        max_p = 0.0025
        max_r2 = 30
        regPeriod = 52  # 1 year
    elif region == 'SE1':
        max_p = 0.0025
        max_r2 = 60
        regPeriod = 52  # 1 year
    elif region == 'SE2':
        max_p = 0.0025
        max_r2 = 60
        regPeriod = 52  # 1 year
    elif region == 'SE3':
        max_p = 0.0025
        max_r2 = 60
        regPeriod = 52  # 1 year
    elif region == 'SE4':
        max_p = 0.0025
        max_r2 = 60
        regPeriod = 78  # 1.5 year
    print('INPUT MAGASIN', region, ':')
    print('Høyeste p-verdi (max_p): ', max_p)
    print('Antall utvalgte serier basert på høyeste R2 som brukes (max_r2): ', max_r2)
    print('Antall uker den "korte" regresjonen kjøres (regPeriod): ', regPeriod, '\n')
    return fasit_key, max_p, max_r2, regPeriod