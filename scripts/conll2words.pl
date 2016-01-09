#!/usr/bin/perl

use strict;
my @sent;

while(<STDIN>){
  if(m/^#/){
    @sent = ();
  }elsif(m/^\s*$/){
#    print "End sentence of length".$#sent."\n";
    print join(" ", @sent)."\n";
    @sent = ();
  }else{
    my @fields = split /\s+/;
    my $tagword = $fields[6]."/".$fields[1];
    push @sent, $tagword;
  }
}

