#!/usr/bin/perl

#perl -pe 's/ /\n/g' | sort | uniq -c | sort -nr | awk '{print $2}' | head -$*

use strict;

my %counts = {};
my $k = $ARGV[0];

sub hashValueDescendingNum {
   $counts{$b} <=> $counts{$a};
}

while(<STDIN>){
    chomp;
    my @words = split / /;
    for my $word (@words){
        if(! exists($counts{$word})){
            $counts{$word} = 0;
        }
        $counts{$word} = $counts{$word} + 1;
    }
}

my $i = 0;
for my $token (sort hashValueDescendingNum (keys(%counts))){
    #print "$token\t".$counts{$token}."\n";
    print "$token\n";
    $i += 1;
    if($i >= $k){
        last;
    }
}
