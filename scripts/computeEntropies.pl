#!/usr/bin/perl
use strict;

my %entropies;
my $lastPos = -1;
my $entropy_sum = 0;

while(<STDIN>){
    if(m/P\( (\S*) \| \((\d+),\) \) = (\S+)/){
        my $word = $1;
        my $pos = $2;
        my $prob = $3;
        if($prob ne "0.000000"){
            if(!exists($entropies{$pos})){
                $entropies{$pos} = 0;
            }
            $entropies{$pos} += ($prob * log($prob));
        }
    }
}

for my $key (sort keys %entropies){
    print "$key => ".$entropies{$key}."\n";
}

