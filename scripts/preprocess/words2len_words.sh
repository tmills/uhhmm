#!/bin/bash

length=`expr $1 - 1`

perl -lane 'if($#F <= '$length'){ print $_; }'
