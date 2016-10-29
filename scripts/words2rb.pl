#!/usr/bin/perl

use strict;

while(<STDIN>){
  chomp;
  my @words = split / /;
  for my $i (0..$#words-1){
    my $word = $words[$i];
    print("(X (X $word)");
  }
  my $word = $words[-1];
  print("(X $word)");
  
  for my $i (0..$#words-1){
    print(")");
  }
  print("\n");
}
