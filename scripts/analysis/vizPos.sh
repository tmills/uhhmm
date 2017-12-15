#!/bin/bash

for i in `seq 0 $2`; do
  echo "********** POS $i *************"
  grep -v 000000 $1 | grep "($i," | sort -nrk8 | head -20
done

