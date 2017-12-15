#!/bin/bash

for i in `seq 1 $2`; do
  echo POS$i
  cat $1 |  perl -pe 's/\((\d+) ([^\s\(\)]+)\)/\1:\2/g;s/ /\n/g;s/\)//g' | grep ":" | sort | grep "^$i:" | uniq -c | sort -nr | head -20
done
