#!/bin/bash

perl -pe 's/(<\/tr>)/\1\n/g'| grep "geonames.de/cou" | perl -pe 's/^.*(http:\/\/geonames.de\/cou\S*\.html)">.*\.html<\/a>"\&gt;<\/span\>([^<>]+)<.*/\2\t\1/' | grep -v "^http"
