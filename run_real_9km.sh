#!/bin/bash
#
# 2011-11-03
# M.Baltscheffsky 
# WeatherTech Scandinavia AB
#
# 2015-12-02
# B. Claremar

# Usage:
#
# ./run_wrf.sh [startyyyy] [startmm] [startdd] [startHH] [endyyyy] [endmm] [enddd] [endHH]
#
# This script will only work if there is a namelist.input.template in 
# the run dir. OBS the namelist.input file will be overwritten.
# ====================================================================
# Some settings
# ====================================================================
clean=0         # =1 will clean away wrf-stuff in run-folder
move=0          # =1 if you want to move the wrfout_d* files.
wrfout="./"     # Directory to move files to
# ====================================================================
# Edit namelist
# ====================================================================
# Get start and end times
syyyy=$1
smm=$2
sdd=$3
sHH=$4

eyyyy=$5
emm=$6
edd=$7
eHH=$8

doms=$9
Dt=${10}
slph=${11}
blph=${12}
rst=${13}
flex=${14}

echo "$Dt"

# Overwrite namelist.input with namelist.input.30.template
cp -f namelist.input.9.template.real namelist.input

echo "*********************************************************"
echo "Editing namelist"

echo ""
# Edit dates in namelist
sed -i "s/syyyy/$syyyy/g" namelist.input
sed -i "s/smm/$smm/g" namelist.input
sed -i "s/sdd/$sdd/g" namelist.input
sed -i "s/sHH/$sHH/g" namelist.input

sed -i "s/eyyyy/$eyyyy/g" namelist.input
sed -i "s/emm/$emm/g" namelist.input
sed -i "s/edd/$edd/g" namelist.input
sed -i "s/eHH/$eHH/g" namelist.input

sed -i "s/doms/$doms/g" namelist.input
sed -i "s/rst/$rst/g" namelist.input
sed -i "s/Dt/$Dt/g" namelist.input
sed -i "s/flex/$flex/g" namelist.input
sed -i "s/slph/$slph/g" namelist.input
sed -i "s/blph/$blph/g" namelist.input


# ====================================================================
# Start REAL
# ====================================================================
echo "*********************************************************"
echo "Ready to run real";
echo ""
#rm rsl*
#Run real.exe
echo "Running real from $syyyy-$smm-$sdd-$sHH to $eyyyy-$emm-$edd-$eHH"
echo ""
./real.exe 1>real.out 2>real.err
cp rsl.error.0000 rsl.real

# Should be finished with real.exe here
# WRF run finished
echo "Finished running real"
echo "Last lines were:"
tail -n 3 rsl.out.0000
echo ""
# ====================================================================
# Clean up
# ====================================================================
#echo "rm rsl*"
#
#rm rsl

