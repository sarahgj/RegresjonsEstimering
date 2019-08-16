@echo on
call activate _aqua
python "D:\users\sarah\Regresjonstipping\scripts\regresjon_tot_tilsig_mag.py"
python "D:\users\sarah\Regresjonstipping\scripts\cleanup_logs.py"
call deactivate
