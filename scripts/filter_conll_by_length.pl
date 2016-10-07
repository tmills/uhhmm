#!/usr/bin/perl

use strict;

my @sent = ();
my $comments = 0;

my $min = $ARGV[0];
my $max = $ARGV[1];

my $last_index = 0;
print "Printing all sentences longer than $min and shorter than $max words\n";

while(<STDIN>){
    if(m/^#/){
        #print("Adding comment to sentence\n");
        push @sent, $_;
        $comments++;
    }elsif(m/^(\d+)/){
        $last_index = int($1);
        push @sent, $_;
    }else{
        #print "Found a sentence with " . $#sent . " tokens\n";
        ## end of sentence
        my $len = $#sent+1 - $comments;
        
        if($last_index >= $min && $last_index <= $max){
          for my $line (@sent){
            print $line;
          }
          print "\n";
        }
        
        @sent = ();
        $comments = 0;
        $last_index = 0;
    }
}
        
