#!/usr/bin/perl
use strict;

my @lines;
my %counts;

my $THRESHOLD = $ARGV[0];
my $pos;
my $word;

while(<STDIN>){
    push @lines, $_;
    my @words = split / /;
    for my $element (@words){
        if($element =~ m/(.+)\/(.+)/){
          $pos = $1;
          $word = $2;
        }else{
          $word = $element;
        }
        if(!exists($counts{$word})){
            $counts{$word} = 0;
        }
        $counts{$word} += 1;
    }
}

for my $line (@lines){
    my $out = "";
    my @words = split / /, $line;
    for my $element (@words){
        chomp($element);
        if($element =~ m/(.+)\/(.+)/){
          $pos = $1;
          $word = $2;
        }else{
          $pos = "";
          $word = $element;
        }
        if($counts{$word} < $THRESHOLD){
            $word = "unk";
        }
        if(length($pos) > 0){
            $out .= "$pos/";
        }
        $out .= "$word ";
    }
    print substr($out, 0, length($out)-1)."\n";
}