#!/bin/bash

perl -pe 's/ /\n/g' | sort | uniq -c | sort -nr | awk '{print $2}' | head -$*

