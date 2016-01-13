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
  }elsif(m/^\d+\s/){
    my @fields = split /\s+/;
    if($fields[2] =~ m/\+/){
        ## Iterate over morpho tokens
        my @words = split /\+/, $fields[2];
        my @tags = split /\+/, $fields[3];
        for my $i (0..$#words){
            my $tagword = $tags[$i]."/".$words[$i];
            push @sent, $tagword;
        }
    }else{    
        ## Just one morpho token -- push it onto sentence
        my $tagword = $fields[3]."/".$fields[1];
        push @sent, $tagword;
    }
  }
}

