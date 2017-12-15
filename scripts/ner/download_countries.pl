#!/usr/bin/perl

use strict;

my $baseUrl = "http://geonames.de/";
for(<STDIN>){
    if(m/^[A-Za-z ]*[A-Za-z]\s*(cou.*\.html)/){
        my $url = $baseUrl . $1;
#        print "Full url is $url\n";
        my $cmd = "wget $url -O data/countries/$1";
        print "Command is $cmd\n";
        `$cmd`; print "Waiting 1 second before next GET\n"; sleep 1;

    }
}