#!/bin/bash

perl -pe "s/\S\S?\/\S\S?::((ACT\d+)\/AWA\d+)?:(POS\d+);\S*/\2/g;s/^\s*//"
