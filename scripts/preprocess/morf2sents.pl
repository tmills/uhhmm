#!/usr/bin/perl

use strict;

my $sent = "";

while(<STDIN>){
  chomp;
  if(m/(.*\.)\s*$/){
    ## End of sentence;
    $sent .= " $1";
    $sent =~ s/  */ /g;
    $sent =~ s/^ *//;
    print "$sent\n";
    $sent = "";
  }else{
    $sent .= " $_";
  }
}
