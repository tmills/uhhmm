#!/usr/bin/perl

use strict;

while(<STDIN>){
  chomp;
  my @words = split / /;
  for my $ind (0..$#words){
    my $word = $words[$ind];
    my $headword;
    if ($ind == 0){
      $headword = "ROOT";
    }else{
      $headword = $words[$ind-1];
    }
    print("X($headword-$ind, $word-".($ind+1).")\n");
  }
  print("\n");
}
