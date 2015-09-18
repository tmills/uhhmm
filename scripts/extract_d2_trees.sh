#!/bin/bash

grep -v "\-d[3-4]" | grep -iv redirect | grep -v ARTICLE | grep -v Equation | grep -v "#)"
