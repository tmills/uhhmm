#!/usr/bin/perl
use strict;

my @lines;
my %counts;

my $THRESHOLD = $ARGV[0];
my $pos;
my $word;

while(<STDIN>){
    s/^\s+|\s+$//g;
    push @lines, $_;
    my @words = split / /;
    my $num_words = $#words + 1;
    my $num_slashes = scalar(split /\//) - 1;
    my $tagwords = 0;
#    print "Num slashes is $num_slashes and num words is $num_words\n";
    if($num_slashes >= $num_words){
        $tagwords = 1;
    }
    
    for my $element (@words){
        if($tagwords == 1 && $element =~ m/(.+)\/(.+)/){
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
    my $num_words = $#words + 1;
    my $num_slashes = scalar(split /\//) - 1;
    my $tagwords = 0;
#    print "Num slashes is $num_slashes and num words is $num_words\n";
    if($num_slashes >= $num_words){
        $tagwords = 1;
    }
    for my $element (@words){
        chomp($element);
        if($tagwords == 1 && $element =~ m/(.+)\/(.+)/){
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
