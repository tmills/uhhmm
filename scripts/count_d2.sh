#!/bin/bash

for i in $1/last_sample*.txt; do
  echo `echo $i | perl -pe 's/.*sample(\d+)\.txt/\1/g'` `cat $i | perl -lane '@slice=@F[2 .. $#F-1]; print "@slice";' | grep "+\/-" | wc -l`
done | sort -n
