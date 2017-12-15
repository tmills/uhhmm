#!/usr/bin/perl

use strict;

while(<STDIN>){
  chomp;
  my @words = split / /;
  
  
  for my $i (0..$#words-1){
    my $word = $words[$i];
    my $word_ind = $i + 1;
    my $next_word = $words[$i+1];
    my $next_ind = $i + 2;
    print("X($next_word-$next_ind, $word-$word_ind)\n");
  }
  print("X(ROOT-0, $words[-1]-".($#words+1).")\n");
  print("\n");
}
