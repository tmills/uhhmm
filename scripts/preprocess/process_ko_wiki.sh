#!/bin/sh

## Remove parens and brackets:
perl -pi -e 's/\([^\)]*\)//g' $*
perl -pi -e 's/\[.*\]//g' $*

## Sentence breaks:
perl -pi -e 's/(\S)\. /\1 .\n/g' $*

## Split numbers and labels
perl -pi -e 's/([\d,\.]+)(\S+)/\1 \2/g' $*

## Add spaces before commas if it's not a number
perl -pi -e 's/([^\s\d]),/\1 ,/g' $*

## Split up acronyms with periods?
perl -pi -e 's/\.(\S)/. \1/g' $*

## space out quotes:
perl -pi -e 's/"(\S)/" \1/g;s/(\S)"/\1 "/g' $*

## Remove leading and trailing whitespace
perl -pi -e 's/^\s*(.*)\s*$/\1\n/' $*

## Remove empty lines:
perl -pi -e 's/^\n$//' $*

