#!/bin/bash


opt=1
#opt=1 run both real.exe and wrf.exe
#opt=2 run only real.exe
#opt=3 run only wrf.exe

run=""

d=$(date --date="7 hours ago" "+%d")
echo $d
hh=$(date --date="7 hours ago" "+%k")
#aa=$(expr $hh - 2)
a=$(expr $hh / 6)
b=$(expr $a \* 6)
h=$b;
if (( $b < 10 ));then h="0$b"; fi;
echo $h


yys=2021
mms=08
dds=$d
hhs=$h
yye=2021
mme=$mms
dde=$(expr $d + 1 ) 
if (( $b > 12 ));then dde=$(expr $dde + 1); fi;
if (( ${dde#0} < 10 )); then dde="0${dde#0}";fi;        
if (( ${dde#0} > 31 )); then 
	dde=01
	mme=$(expr $mms + 1 )
	mme="0${mme#0}"
fi
hhe=$(expr $b + 6 )
if (( $hhe > 18 ));then hhe=$(expr $hhe - 24); fi;
if (( $hhe < 10 )); then hhe="0$hhe";fi;        
#dde=05
#hhe=18
echo $dde
echo $hhe
cd /clusterfs2/WRF/WRF_dmpar/run
#exit

doms=1
dt=54
slph=5
blph=5
rst='.false.'
flex='.false.'

if [ "$opt" -eq 1 -o "$opt" -eq 2 ];then
        ln -sf ../../WPS/met_em.d0*.2021-08-* .
        ln -sf ../../WPS/met_em.d0*.2021-09-* .
	./run_real_9km.sh $yys $mms $dds $hhs $yye $mme $dde $hhe $doms $dt $slph $blph $rst $flex
fi

if [ "$opt" -eq 1 -o "$opt" -eq 3 ];then
        ./run_wrf_9km.sh $yys $mms $dds $hhs $yye $mme $dde $hhe $doms $dt $slph $blph $rst $flex

	#echo "moving all wrfout files"
#	mkdir $run
#	echo "wrfout_d* /home/bclaremar/WRF_output/$run/"
#	mv wrfout_d* /home/bclaremar/WRF_output/$run/
fi

