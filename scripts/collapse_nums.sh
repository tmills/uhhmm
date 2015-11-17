#!/bin/bash

perl -pe 's/\/\S*\d\S*/\/number_token/g'
