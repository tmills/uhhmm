#!/usr/bin/perl
#
# This script converts the PTB tres with morphological segmentation like:
# (NP token1/NN token2/NN token3/PAN) => (NP (NN token1) (NN token2) (PAN token3))
#
use strict;

while(<STDIN>){
    my $tree = $_;
    my @matches = m/\((\S+) ([^\)\(]+)\)/g;
    
    for(my $i = 0; $i < $#matches; $i = $i + 2){
      my $cat = $matches[$i];
      my $lex = $matches[$i+1];
      
      my @morphs = split /[\+ ]/, $lex;
      if($#morphs > 0){
        my $expanded_lex = "";
        for my $morph (@morphs){
          my ($token, $pos) = $morph =~ m/(.+)\/(.+)/;
          $expanded_lex .= " ($pos $token)";
        }
        $expanded_lex = substr $expanded_lex, 1;
        $tree =~ s/\Q$lex\E/$expanded_lex/e;
      }
      
    }
    print $tree;
}