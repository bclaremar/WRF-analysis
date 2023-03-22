#!/bin/bash
cd ~/run
del=10
m=$(date --date=$del" hours ago" "+%m")
echo $m
d=$(date --date=$del" hours ago" "+%d")
echo $d
hh=$(date --date=$del" hours ago" "+%k")
ht=$(date --date="0 hour" "+%k")
ht=$(expr $ht / 2 \* 2 + 1 )

#aa=$(expr $hh - 2)
a=$(expr $hh / 6)
b=$(expr $a \* 6)
h=$b;
if (( $b < 10 ));then h="0$b"; fi;
echo $h
echo $ht

#if [ $h == 06 ];then
#	h='00'
#fi
if [ $h == 06 ];then
        do='1'
        ./mapplot.py  <<EOF
	$m $d $h 1 $ht
EOF
else
        do='1'
	./mapplot.py  <<EOF2
	$m $d $h 1 $ht
EOF2
        do='2'
        ./mapplot.py  <<EOF2
        $m $d $h 2 $ht
EOF2
fi

echo $do
echo $h
#./mapplot.py  <<EOF
#$m $d $h 1 $ht
#EOF

