#!/usr/bin/perl

use strict;


my @pos = ();
my @word = ();

while(<STDIN>){
  my @words = split / /;
  for my $tagword (@words){
    my @pair = split /\//, $tagword;
    push @pos, $pair[0]+1;
    push @word, $pair[1]+1;
  }
  push @pos, 1;
  push @word, 1;
}

my $numRows = $#word;

print "# name: X\n";
print "# type: matrix\n";
print "# rows: $numRows\n";
print "# columns: 2\n";

($#word == $#pos) or die("Word vector is different length than POS vector!");

for my $i (0..@pos){
    print " $pos[$i] $word[$i]\n";
}
