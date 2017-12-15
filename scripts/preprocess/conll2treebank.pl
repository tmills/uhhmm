#!/usr/bin/perl

use strict;
my @sent;
my @heads;
my @tags;
my @words;
my @head_counts;

while(<STDIN>){
  if(m/^#/){
    @sent = ();
    @heads = ();
    @tags = ();
    @words = ();
    push @sent, "TOP";
    push @heads, "-1";
    push @tags, "TOP";
    push @words, "TOP";
    push @head_counts, 0;
  }elsif(m/^$/){  ##
  
    ## Turn the collected dependencies into a tree:
    my @tree = ();
    for my $i (0..$#sent) {
        $tree[$i] = "($tags[$i] $words[$i])";
    }
    
    while($#heads > 1){
        my $beforeSize = $#tree;
#        print "@tree \n";
#        print "@heads \n";
#        print "@head_counts \n";
        my $startInd, my $endInd;
        ## Get all the neighbors who $i is the head of
        for my $i (1..$#heads) {
#            print "Looking at index $i\n";
            $startInd = $i;
            $endInd = $i;
            for (my $j = ($i-1); $j >= 1; $j--){
#                print "heads[j] at $j is $heads[$j]\n";
                if ($heads[$j] == $i) {
                    $startInd = $j;
                }else{
                    last;
                }
            }
            for my $j ($i+1..$#heads) {
                if ($heads[$j] == $i) {
                    $endInd = $j;
                }else{
                    last;
                }
            }
        
#            if($startInd < $i || $endInd > $i){
#                print "Found neighbors of $i with the correct index from $startInd to $endInd\n";
#            }
            
            if(($endInd-$startInd) == $head_counts[$i] && ($startInd < $i || $endInd > $i)){
#                print "Combining nodes $startInd to $endInd\n";
                ## build a tree with $i as the head
                my $subtree;
                
                if($startInd < $i && $endInd > $i){
                    $subtree = "(X ";
                    for my $j ($startInd..$endInd) {
                        $subtree .= $tree[$j]." ";
                    }
                    $subtree .= ")";
                }elsif($startInd < $i){
                    
                }elsif($endInd >$i){
                }else{
                    ## shouldn't be possible
                    print STDERR "Error - found in condition that should be impossible.\n";
                    exit;
                }
            
                ## put the tree at the start ind and delete everything after.
                $tree[$startInd] = $subtree;
                splice @tree, $startInd+1, ($endInd-$startInd);

                ## keep the reference to $i's head and move to the front then delete everything after
                $heads[$startInd] = $heads[$i];        
                splice @heads, $startInd+1,  ($endInd-$startInd);
                
                ## decrement all the head indices
                for my $j (1..$#heads){
                    if ($heads[$j] > $endInd ){
                        $heads[$j] -= ($endInd-$startInd);
                    }elsif($heads[$j] > $startInd ){
                        $heads[$j] -= ($i-$startInd);
                    }
                }
                
                ## Recount the head counts based on adjusted indices:
                @head_counts = ();
                $head_counts[0] = 0;
                for my $j (1..$#heads){
                    $head_counts[$j] = 0;
                }
                for my $j (1..$#heads){
                    $head_counts[$heads[$j]]++;
                }
                
#                print "@tree \n";
#                print "@heads \n";
#                print "@head_counts \n";
            }                    
         }
         if($beforeSize == $#tree){
            ## no reductions this time step, just exit
            print "(TOP (S error))\n";
            last;
         }
    }
    
    if($#tree == 1){
      print $tree[1]."\n";
    }
    
    ## Reset the data structures
    @sent = ();
    @heads = ();
    @tags = ();
    @words = ();
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
    $fields[1] =~ s/\(/-LRB-/g;s/\)/-RRB-/g;
    
    push @words, $fields[1];
    push @tags, $fields[3];
    push @heads, $fields[6];
    $head_counts[$fields[6]]++;
  }else{
    #print STDERR "SKipping line $_";
  }
}

