#!/bin/bash

perl -pe "s/.\/.::\d+\/\d+:(\d+);\S*/\1/g"
