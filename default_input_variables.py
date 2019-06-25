def get_default(variable, region):
    """
    This function inherits the default functions below, so that they can easily be called.

    Example:
        fasit_key, max_p, ant_kandidater, reg_period = get_default('tilsig','NO1')
    """
    if variable == "magasin":
        func = default_magasin
    else:
        func = default_tilsig
    fasit_key, reg_period, max_p, ant_kandidater = func(region)
    print('DEFAULT INPUT MAGASIN', region, ':')
    print('Antall utvalgte kandidater utifra korrelasjon med fasit (ant_kandidater): ', ant_kandidater)
    print('Høyeste p-verdi som tas med i sluttmodellen (max_p): ', max_p)
    print('Antall uker den "korte" regresjonen kjøres (reg_period): ', reg_period, '\n')
    return fasit_key, reg_period, max_p, ant_kandidater


def default_tilsig(region):
    """
    This function keeps information about default variable settings for each reagion and the variable 'tilsig'.

    Args:
        region: NO1/.../NO5/SE1/../SE4

    Returns:
        fasit_key: Fasit series to be used in the regression.
        ant_kandidater: Number of candidates chosen for the regression based on correlation with the fasit series.
        max_p: highest possible p-value included in the final regression, except if too few series (<6) goes in to the regression.
        reg_period: number of weeks included in the 'short' regression.

    Example:
        fasit_key, max_p, ant_kandidater, reg_period = default_tilsig('NO1')
    """
    if 'N' in region:
        fasit_key = '/Norg-No' + region[-1] + '.Fasit.....-U9100S0BT0105'
    elif 'S' in region:
        fasit_key = '/Sver-Se' + region[-1] + '.Fasit.....-U9100S0BT0105'
    if region == 'NO1':
        max_p = 0.025
        ant_kandidater = 10
        reg_period = 10
    elif region == 'NO2':
        max_p = 0.025
        ant_kandidater = 110
        reg_period = 50
    elif region == 'NO3':
        max_p = 0.025
        ant_kandidater = 30
        reg_period = 100
    elif region == 'NO4':
        max_p = 0.025
        ant_kandidater = 120
        reg_period = 100
    elif region == 'NO5':
        max_p = 0.025
        ant_kandidater = 80
        reg_period = 50
    elif region == 'SE1':
        max_p = 0.025
        ant_kandidater = 100
        reg_period = 70
    elif region == 'SE2':
        max_p = 0.025
        ant_kandidater = 70
        reg_period = 80
    elif region == 'SE3':
        max_p = 0.025
        ant_kandidater = 130
        reg_period = 50
    elif region == 'SE4':
        max_p = 0.025
        ant_kandidater = 50
        reg_period = 50
    return fasit_key, reg_period, max_p, ant_kandidater


def default_magasin(region):
    """
    This function keeps information about default variable settings for each reagion and the variable 'tilsig'.

    Args:
        region: NO1/.../NO5/SE1/../SE4

    Returns:
        fasit_key: Fasit series to be used in the regression.
        ant_kandidater: Number of candidates chosen for the regression based on correlation with the fasit series.
        max_p: highest possible p-value included in the final regression, except if too few series (<6) goes in to the regression.
        reg_period: number of weeks included in the 'short' regression.

    Example:
        fasit_key, max_p, ant_kandidater, reg_period = default_magasin('NO1')
    """
    if 'N' in region:
        fasit_key = '/Norg-NO' + region[-1] + '.Fasit.....-U9104A5R-0132'
    elif 'S' in region:
        fasit_key = '/Sver-SE' + region[-1] + '.Fasit.....-U9104A5R-0132'
    if region == 'NO1':
        max_p = 0.025
        ant_kandidater = 70
        reg_period = 201
    elif region == 'NO2':
        max_p = 0.025
        ant_kandidater = 70
        reg_period = 120
    elif region == 'NO3':
        max_p = 0.025
        ant_kandidater = 45
        reg_period = 126
    elif region == 'NO4':
        max_p = 0.025
        ant_kandidater = 54
        reg_period = 120
    elif region == 'NO5':
        max_p = 0.025
        ant_kandidater = 133
        reg_period = 109
    elif region == 'SE1':
        max_p = 0.025
        ant_kandidater = 120
        reg_period = 206
    elif region == 'SE2':
        max_p = 0.025
        ant_kandidater = 100
        reg_period = 105
    elif region == 'SE3':
        max_p = 0.025
        ant_kandidater = 110
        reg_period = 50
    elif region == 'SE4':
        max_p = 0.025
        ant_kandidater = 110
        reg_period = 50
    return fasit_key, reg_period, max_p, ant_kandidater