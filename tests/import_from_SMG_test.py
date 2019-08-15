def import_magasiner():
    NO1 = {  # NO1 :
        'Randsfjorden_4309': ['/NVE.-Randsfjord....-D1004A0V', 409.2],
        'Raudalsvatn3_531': ['/GLB.-Raudalsvatn...-D1004A5AI0101', 166]}
    NO2 = {  # NO2 :
        'Tysso2_Sum': ['/Tyss-Tysso2........-U1004S0BT0132', 0]}

    NO3 = {  # NO3 :
        'Otta-Krv.omr.EMPS': ['/Otta-Krv.omr.EMPS..-U9104S0BT0132', 0]}
    NO4 = {  # NO4 :
        'Roessvatn_64802': ['/Roes-Roessvatn.....-D1004S0BT0132', 0]}
    NO5 = {  # NO5:
        'Jukla_Sum': ['/Folg-Jukla.........-U1004S0BT0132', 0]}
    Sverige = {  # SE1 :
        'Rebnisjaure_1211': ['/Skel-Rebnisjaure...-D1004V3R-0101SV..SVF', 746.8]}

    KrvOmr = {  # Region Krv.omr.EMPS :
        'KNO1-Krv.omr.EMPS': ['/KNO1-Krv.omr.EMPS..-U9104S0BT0132', 0],
        'KNO2-Krv.omr.EMPS': ['/KNO2-Krv.omr.EMPS..-U9104S0BT0132', 0],
        'KNO3-Krv.omr.EMPS': ['/KNO3-Krv.omr.EMPS..-U9104S0BT0132', 0],
        'KNO4-Krv.omr.EMPS': ['/KNO4-Krv.omr.EMPS..-U9104S0BT0132', 0],
        'KNO5-Krv.omr.EMPS': ['/KNO5-Krv.omr.EMPS..-U9104S0BT0132', 0],
        'KSE2-Krv.omr.EMPS': ['/KSE2-Krv.omr.EMPS..-U9104S0BT0132', 0]}


    return [NO1,NO2,NO3,NO4,NO5,Sverige], ['NO1','NO2','NO3','NO4','NO5','Sverige']


def import_tilsig():
    NO1 = {  # NO1
        'Jondals_478U': ['/HBV/LTM2-Jondals-015.020/LTM/UPDAT/Q_OBSE']}
    NO2 = {  # NO2 :
        'Bulken_598U': ['/HBV/LTM3-Bulken-062.005/LTM/UPDAT/Q_OBSE']}

    NO3 = {  # NO3 :
        'Øyungen_685U': ['/HBV/LTM6-Øyungen-138.001/LTM/UPDAT/Q_OBSE']}

    NO4 = {  # NO4 :
        'Adamsel_5351U': ['/HBV/LTM8-Adamsel-229.006/LTM/UPDAT/Q_OBSE']}

    # Problemer med isoppstuing:
    # 'Polmak._772U' : ['/HBV/LTM8-Polmak.-234.018/LTM/UPDAT/Q_OBSE'],

    NO5 = {  # NO5:
        'Langsim_5190U': ['/HBV/LTM4-Langsim-050.051/LTM/UPDAT/Q_OBSE']}

    SE1 = {  # SE1 :
        'SMHI-Sorsele': ['/SMHI-Sorsele.......-D1054A3KI0108']}

    SE2 = {  # SE2 :
        'SMHI-Rengen': ['/SMHI-Rengen........-D1054A3KI0108']}

    SE3 = {  # SE3 :
        'SMHI-Vattholma': ['/SMHI-Vattholma.....-D1054A3KI0108']}

    SE4 = {  # SE4 :
        'Nissafors_1901U': ['/HBV/LTMS-Nissafors-1901/LTM/UPDAT/Q_OBSE']}

    kjente = {  # Magendring kjent pr prisområde
        'SE4-Kjent': ['/Sver-SE4.kjente....-D9100S0BT0105']}

    ukjente = {  # Tilsig ukjent pr prisområde
        'SE4-Ukjent': ['/Sver-SE4.ukjente...-U9100S0BT0105']}

    tipping = {  # Tilsig BesteEstimat pr prisområde
        'SE4-BesteEstimat': ['/Sver-Se4.BesteEstimat-D9100A5V-0105'],

        # Tilsig Q_obse pr prisområde
        'SE4-QOBSE': ['/Sver-SE4.QOBSE.....-D9100A5V-0105'],

        # Tilsig Q_NFB pr prisområde
        'SE4-QNFB': ['/Sver-SE4.QNFB.......-D9100A5V-0105']}

    MagEndr1 = {  # Tilsig basert på magasinendring pluss Kpp-prod
        'Øvre-Vinstra-SE1Prod': ['/ØvreSE1Prod-Vinstra.EMPS..-D9100A5V-0105']}

    MagEndr2 = {  # Tilsig basert på magasinendring pluss Kpp-prod
        'Ulla-Krv-NO2Prod': ['/Ulla-Krv.omr.EMPS..-D9100A5V-0105']}

    MagEndr3 = {  # Tilsig basert på magasinendring pluss Kpp-prod
        'Skjo-Resten-NO4Prod': ['/Skjo-Krv.res.EMPS..-D9100A5V-0105']}

    MagEndr4 = {  # Tilsig basert på magasinendring pluss Kpp-prod
        'KSE2-Krv-SE2Prod': ['/KSE2-Krv.omr.EMPS..-D9100A5V-0105']}

    MagEndr5 = {  # Tilsig basert på magasinendring pluss Kpp-prod
        'Laga-Krv-SE3Prod': ['/LagaSE3Prod-Krv.omr.EMPS..-D9100A5V-0105']}

    KrfoDel1 = {  # tilsiget pr kraftverksområde
        'Haka-Hakavik': ['/Haka-Hakavik.......-U9100S3BT0104']}

    KrfoDel2 = {  # tilsiget pr kraftverksområde
        'Laga-Åby': ['/Laga-Åby...........-U9100S3BT0104']}

    KrfoDel3 = {  # tilsiget pr kraftverksområde
        'Nore-Pålsbu': ['/Nore-Pålsbu........-U9100S3BT0104']}

    KrfoDel4 = {  # tilsiget pr kraftverksområde
        'RSK.-Vasstøl': ['/RSK.-Vasstøl.......-U9100S3BT0104']}
    # 'Sima-Grasbotntjørni' : ['/Sima-Grasbotntjørni-U9100S3BT0104']}

    KrfoDel5 = {  # tilsiget pr kraftverksområde
        'Tokk-Kjela': ['/Tokk-Kjela.........-U9100S3BT0104']}

    KrfoDel6 = {  # tilsiget pr kraftverksområde
        'Vikf-Refsdal': ['/Vikf-Refsdal.......-U9100S3BT0104']}

    return [NO1, NO2, NO3, NO4, NO5, SE1, SE2, SE3, SE4, kjente, ukjente, tipping, MagEndr1, MagEndr2, MagEndr3,
            MagEndr4, MagEndr5, KrfoDel1, KrfoDel2, KrfoDel3, KrfoDel4, KrfoDel5, KrfoDel6], \
           ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4',
            'Tilsig kjente pr prisområde', 'Tilsig ukjente pr prisområde', 'BesteEstimat, Q_obse, Q_NFB',
            'Tilsig basert på magasinendring pluss Kpp-prod, Del1', 'Del2', 'Del3', 'Del4', 'Del5',
            'Tilsiget pr kraftverksområde: Del1', 'Del2', 'Del3', 'Del4', 'Del5', 'Del6']
