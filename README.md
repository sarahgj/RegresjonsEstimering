# Regression Estimation of Unknown Magazine Inflow

The scripts in this project is run operationally every week to help the Vannhusholdning to estimate unknown regional 
inflow on Tuesdays. Two auto jobs are run with Task Scheduler from the oslwpsht001p server.

The estimation is made by running a regression, using respectively all known and actual inflow and magazine series 
to mach the inflow and magazine Fasit series.
  
# HowTo:
   1. Run scripts/regresjon_tot_tilsig_mag-TUNING.py to make an updated tuning of the input variables.
   2. Run scripts/regresjon_tot_tilsig_mag.py to update the new tuning results with SMG.

Author: Sarah Gjermo <sarah.gjermo@statkraft.com> \
On behalf of: Svein Farstad <svein.farstad@statkraft.com>