#!/usr/bin/perl

use strict;
my @sent;

while(<STDIN>){
  if(m/^#/){
    @sent = ();
  }elsif(m/^\d+\s+(\S+).*/){
    push @sent, $1;
  }elsif(m/^\s*$/){
#    print "End sentence of length".$#sent."\n";
    print join(" ", @sent)."\n";
    @sent = ();
  } 
}

