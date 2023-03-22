#!/bin/bash

#Lenght in hours
L=24

opt=1
#opt=1 run both real.exe and wrf.exe
#opt=2 run only real.exe
#opt=3 run only wrf.exe

run=""
lag=6
y=$(date --date="${lag} hours ago" "+%Y")
m=$(date --date="${lag} hours ago" "+%m")
d=$(date --date="${lag} hours ago" "+%d")
echo "start"
echo $d
hh=$(date --date="${lag} hours ago" "+%k")
#aa=$(expr $hh - 2)
a=$(expr $hh / 6)
b=$(expr $a \* 6)
h=$b;
if (( $b < 10 ));then h="0$b"; fi;
echo $h


yys=$y
mms=$m
dds=$d
hhs=$h
yye=$y
mme=$mms
hhe2=$(( $h+$L ))
dde=$(( 10#$d + hhe2 / 24 ))
hhe=$(( $hhe2 % 24 ))
echo "end"
echo $dde
echo $hhe
echo "---"
#exit

#if (( $b > 0 ));then dde=$(expr $dde + 0); fi;
if (( ${dde#0} < 10 )); then dde="0${dde#0}";fi;        
#if 'item "1 3 5 7 8 10 12" "$mm"';then
#	dx=31
#else
#	dx=30
#fi
dx=28
echo $dx
if (( ${dde#0} > ${dx} )); then 
	dde=01
	mme=$(expr $mms + 1 )
        mme=$(($mme % 12 ))
#       mme="0${mme#0}"
        if (( $mme < 12 )); then yye=2023; fi;
fi
if (( $hhe < 10 )); then hhe="0$hhe";fi;
if (( ${dde#0} < 10 )); then dde="0${dde#0}";fi;
if (( ${mme#0} < 10 )); then mme="0${mme#0}";fi;
echo ""
echo $mme
echo $dde
echo $hhe
cd /clusterfs/software/WRF4.3/smdm/WRF-4.3.3/run
#exit

doms=2
dt=54
slph=5
blph=5
rst='.false.'
flex='.false.'

#exit
echo $doms
echo $dt
echo  $slph 
echo $blph
echo $rst
echo $flex

#exit

if [ "$opt" -eq 1 -o "$opt" -eq 2 ];then
	rm -f met_em.d0*
        ln -sf ../../../WPS-4.3.1/met_em.d0*.$yys-$mms-$dds* .
        ln -sf ../../../WPS-4.3.1/met_em.d0*.$yye-$mme-$dde* .
	./run_real_9km.sh $yys $mms $dds $hhs $yye $mme $dde $hhe $doms $dt $slph $blph $rst $flex
fi

#exit

if [ "$opt" -eq 1 -o "$opt" -eq 3 ];then
        ./run_wrf_9km_3km.sh $yys $mms $dds $hhs $yye $mme $dde $hhe $doms $dt $slph $blph $rst $flex
#        ./an.sh
	#echo "moving all wrfout files"
#	mkdir $run
#	echo "wrfout_d* /home/bclaremar/WRF_output/$run/"
#	mv wrfout_d* /home/bclaremar/WRF_output/$run/
fi

