#!/bin/bash

#Lenght in hours
L=40

opt=1
#opt=1 run both real.exe and wrf.exe
#opt=2 run only real.exe
#opt=3 run only wrf.exe

run=""
lag=5
y=$(date --date="${lag} hours ago" "+%Y")
m=$(date --date="${lag} hours ago" "+%m")
d=$(date --date="${lag} hours ago" "+%d")
echo $d
hh=$(date --date="${lag} hours ago" "+%k")
#aa=$(expr $hh - 2)
a=$(expr $hh / 6)
b=$(expr $a \* 6)
h=$b;
if (( $b < 10 ));then h="0$b"; fi;
echo $h
echo ""

yys=$y
mms=$m
dds=$d
hhs=$h
yye=$y
mme=$mms
hhe2=$(( $hhs+$L))
dde=$((10#$d + $hhe2 / 24 ))
hhe=$(( $hhe2 % 24))
echo $mme
echo $dde
echo $hhe

#exit
dx=$(cal $(date +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}')
dx=$(cal $(date +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}')
dx1=$(($dx + 1))
echo $dx1
echo $dx
echo $(($dx + 1))
#dx=${dx}
#echo "$(($dx + 1))"
dx=31
if (( ${dde#0} > $dx )); then 
        dde="0$(expr $dde - $dx )"
        mme=$(expr $mms + 1 )
        mme=$(($mme % 12 ))
#       mme="0${mme#0}"
        if (( $mme < 12 )); then yye=2023; fi;
fi
#hhe=$(expr $b + 0 )
#if (( $hhe > 18 ));then hhe=$(expr $hhe - 24); fi;
if (( $hhe < 10 )); then hhe="0$hhe";fi;        
#dde=05
#hhe=18
if (( ${dde#0} < 10 )); then dde="0${dde#0}";fi;
if (( ${mme#0} < 10 )); then mme="0${mme#0}";fi;
echo ""
echo $mme
echo $dde
echo $hhe
cd /clusterfs/software/WRF4.3/smdm/WRF-4.3.3/run
#exit

doms=1
dt=54
slph=4
blph=4
rst='.false.'
flex='.false.'

if [ "$opt" -eq 1 -o "$opt" -eq 2 ];then
        ln -sf ../../../WPS-4.3.1/met_em.d0*.$yys-$mms-$dds* .
        ln -sf ../../../WPS-4.3.1/met_em.d0*.$yye-$mme-$dde* .
      	./run_real_9km.sh $yys $mms $dds $hhs $yye $mme $dde $hhe $doms $dt $slph $blph $rst $flex
fi

if [ "$opt" -eq 1 -o "$opt" -eq 3 ];then
        ./run_wrf_9km.sh $yys $mms $dds $hhs $yye $mme $dde $hhe $doms $dt $slph $blph $rst $flex
	./an.sh
	#echo "moving all wrfout files"
#	mkdir $run
#	echo "wrfout_d* /home/bclaremar/WRF_output/$run/"
#	mv wrfout_d* /home/bclaremar/WRF_output/$run/
fi

