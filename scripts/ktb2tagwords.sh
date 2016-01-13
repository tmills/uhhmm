#!/bin/bash


for i in $1/*.fid; do
#    echo $i
    cat $i | fromdos | iconv -f EUC-KR -t UTF-8 | grep -v "^;;" | perl -pe 's/^(.*)/<SOL>\1/g;s/\n/ /g' | perl -pe 's/<SOL> <SOL>/\n/g' | perl -pe 's/<SOL>//g' | perl -pe 's/\(\S+//g;s/\)//g;s/  */ /g' | perl -pe 's/\+/ /g' | perl -pe 's/(\S+)\/(\S+)/\2\/\1/g' | perl -pe 's/ \*\S*//g' | perl -pe 's/^ *//g' | grep -v "^$"
done
