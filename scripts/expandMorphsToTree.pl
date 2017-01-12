#!/usr/bin/perl
#
# This script converts the PTB tres with morphological segmentation like:
# (NP token1/NN token2/NN token3/PAN) => (NP (NN token1) (NN token2) (PAN token3))
#
use strict;

while(<STDIN>){
    my $tree = $_;
    my @matches = m/ ([^\)\(]+)\)/g;
    
    for(my $i = 0; $i < $#matches; $i = $i + 1){
      my $lex = $matches[$i];
      #print "lex is $lex\n";
      my @morphs = split /[\+ ]/, $lex;
        my $expanded_lex = "";
        for my $morph (@morphs){
          if($morph =~ m/(.+)\/(.+)/){
            my $token = $1;
            my $pos = $2;
            $expanded_lex .= " ($pos $token)";
          }else{
            $expanded_lex = $lex;
          }
        }
        if($#morphs > 1){
          $expanded_lex = substr $expanded_lex, 1;
        }

        $tree =~ s/\Q$lex\E/$expanded_lex/e;
    }
    print $tree;
}