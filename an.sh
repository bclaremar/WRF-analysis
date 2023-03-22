#!/bin/bash
cd ~/run
del=9
m=$(date --date=$del" hours ago" "+%m")
echo $m
d=$(date --date=$del" hours ago" "+%d")
echo $d
hh=$(date --date=$del" hours ago" "+%k")
#aa=$(expr $hh - 2)
a=$(expr $hh / 6)
b=$(expr $a \* 6)
h=$b;
if (( $b < 10 ));then h="0$b"; fi;
echo $h

if [ $h == 06 ];then
	do='1'
else
	do='2'
fi

echo $do
#exit

#echo $m $d $h "d01" 
#./short.sh $m $d $h

if [ $do == '1' ];then
        echo $m $d $h "d01" 
        ./short.sh $m $d $h
elif [ $do == '2' ];then
	echo $m $d $h "d02" 
     #   ./short.sh $m $d $h 
	./short_d02.sh $m $d $h 
fi

file="wrfout_d01_2023-${m}-${d}_${h}:00:00"
echo $file
#file='wrfout_d01_2021-12-17_06:00:00'
if test -f "$file"; then
        echo "$file exists."
fi
#exit

./timeplot.py <<EOF
$m $d $h $do
EOF


