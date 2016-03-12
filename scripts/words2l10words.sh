#!/bin/bash

perl -lane 'if($#F <= 9){ print $_; }'
