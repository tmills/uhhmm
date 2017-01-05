#!/usr/bin/perl

use strict;

while(<STDIN>){
  chomp;
  my @words = split / /;
  
  for my $i (0..$#words-1){
    print("(X ");
  }
  
  for my $i (0..$#words){
    my $word = $words[$i];
    print("(X $word)");
    if($i > 0){
      print(")");
    }
  }
  print("\n");
}
 
