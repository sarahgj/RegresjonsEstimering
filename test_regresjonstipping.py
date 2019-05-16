from regression_modules import get_timeperiods


def test_get_timeperiods():
    period, t = get_timeperiods("Tilsig")
    assert t == "2019.04.01"
