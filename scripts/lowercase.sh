#!/bin/bash

perl -pe 'if(m/\//){ s/(\S*)\/(\S*)/\1\/\L\2/g;}else{s/([^\s\/]+)/\L\1/g;}'

