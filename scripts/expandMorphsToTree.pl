#!/usr/bin/perl

use strict;

while(<STDIN>){
    my $tree = $_;
    my @matches = m/\((\S+) ([^\)\(]+)\)/g;
    
    for(my $i = 0; $i < $#matches; $i = $i + 2){
#      print "### New match ###\n";
#      print $matches[$i] . " " . $matches[$i+1] . "\n";
      my $cat = $matches[$i];
      my $lex = $matches[$i+1];
      
      my @morphs = split /\+/, $lex;
#      print "Morphs are @morphs\n";
      if($#morphs > 0){
        my $expanded_lex = "";
        for my $morph (@morphs){
#          print "Looking at morph $morph\n";
          my ($token, $pos) = $morph =~ m/(.+)\/(.+)/;
#          print "Token is $token and pos is $pos\n";
          $expanded_lex .= " ($pos $token)";
        }
        $expanded_lex = substr $expanded_lex, 1;
#        print "Replacing $lex with $expanded_lex\n";
        $tree =~ s/\Q$lex\E/$expanded_lex/e;
      }
      
    }
    print $tree;
}