#!/bin/bash
cd ~/run
del=24
del2=3
m=$(date --date=$del" hours ago" "+%m")
echo $m
d=$(date --date=$del" hours ago" "+%d")
echo $d

echo "Removing"
ls -lrt wrfout*2023-$m-${d}*0
rm -f wrfout*2023-$m-${d}*0

echo "Removing"
ls -lrt wrfout*2023-$m-${d}_{00,12,18}*
rm -f wrfout*2023-$m-${d}_{00,12,18}*

m=$(date --date=$del2" months ago" "+%m")
echo $m
d=$(date --date=$del2" months ago" "+%d")
echo $d

echo "Removing"
ls -lrt wrfout*2022-$m-${d}*
rm -f wrfout*2022-$m-${d}*

