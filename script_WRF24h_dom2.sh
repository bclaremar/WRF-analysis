#!/bin/bash


opt=1
#opt=1 run both real.exe and wrf.exe
#opt=2 run only real.exe
#opt=3 run only wrf.exe

run=""
lag=5
m=$(date --date="${lag}hours ago" "+%m")
d=$(date --date="${lag} hours ago" "+%d")
echo $d
hh=$(date --date="${lag} hours ago" "+%k")
#aa=$(expr $hh - 2)
a=$(expr $hh / 6)
b=$(expr $a \* 6)
h=$b;
if (( $b < 10 ));then h="0$b"; fi;
echo $h


yys=2021
mms=$m
dds=$d
hhs=$h
yye=2021
mme=$mms
dde=$(expr $d + 1 )


if (( $b > 0 ));then dde=$(expr $dde + 0); fi;
if (( ${dde#0} < 10 )); then dde="0${dde#0}";fi;        
#if 'item "1 3 5 7 8 10 12" "$mm"';then
#	dx=31
#else
#	dx=30
#fi
dx=30
echo $dx
if (( ${dde#0} > ${dx} )); then 
	dde=01
	mme=$(expr $mms + 1 )
        mme="${mme#0}"
fi
hhe=$(expr $b + 00 )
if (( $hhe > 18 ));then hhe=$(expr $hhe - 0); fi;
if (( $hhe < 10 )); then hhe="0$hhe";fi;        
#dde=05
#hhe=18
echo $dde
echo $hhe
cd /clusterfs2/WRF/WRF_dmpar/run
#exit

doms=2
dt=54
slph=4
blph=4
rst='.false.'
flex='.false.'

if [ "$opt" -eq 1 -o "$opt" -eq 2 ];then
	rm -f met_em.d0*
     #   ln -sf ../../WPS/met_em.d0*.2021-11-3* .
        ln -sf ../../WPS/met_em.d0*.2021-12-* .
	./run_real_9km.sh $yys $mms $dds $hhs $yye $mme $dde $hhe $doms $dt $slph $blph $rst $flex
fi

if [ "$opt" -eq 1 -o "$opt" -eq 3 ];then
        ./run_wrf_9km_3km.sh $yys $mms $dds $hhs $yye $mme $dde $hhe $doms $dt $slph $blph $rst $flex
        ./an.sh
	#echo "moving all wrfout files"
#	mkdir $run
#	echo "wrfout_d* /home/bclaremar/WRF_output/$run/"
#	mv wrfout_d* /home/bclaremar/WRF_output/$run/
fi

