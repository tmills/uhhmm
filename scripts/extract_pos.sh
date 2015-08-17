#!/bin/bash

perl -pe "s/'.\/. \d+\/\d+:(\d+)'/\1/g;s/[\[\]]//g;s/,//g"
