"""
Script for batch execution of random PCFG generation and tree sampling

@author: Cory Shain
"""

import sys, argparse, numpy as np
from PCFG_Generator import PCFG_Generator

argparser = argparse.ArgumentParser('''
Generates batches of random PCFGs and trees sampled from them.
''')
argparser.add_argument('-o', '--output_dir', dest='output_dir', default='.', action='store', help='Output directory path.')
argparser.add_argument('-n', '--dry_run', dest='dry_run', action='store_true', help='Just print a list of the files that would be generated by the command line call, without actually generating them.')
argparser.add_argument('-t', '--num_trees', dest='num_trees', default='20000', action='store', help='Number of trees to sample from each generated PCFG. num_trees can be specified as a single integer, a range of integers "start:end" (in which case a step size of 1 will be used), or a range of integers with an integer step specification "start:step:end". If left unspecified, defaults to 20000.')
argparser.add_argument('-a', '--a_size', dest='a_size', default='5', action='store', help='Number of states in A (active category) variable. a_size can be specified as a single integer, a range of integers "start:end" (in which case a step size of 1 will be used), or a range of integers with an integer step specification "start:step:end". If left unspecified, defaults to 5.')
argparser.add_argument('-b', '--b_size', dest='b_size', default='5', action='store', help='Number of states in B (awaited category) variable. b_size can be specified as a single integer, a range of integers "start:end" (in which case a step size of 1 will be used), or a range of integers with an integer step specification "start:step:end". If left unspecified, defaults to 5.')
argparser.add_argument('-p', '--p_size', dest='p_size', default='5', action='store', help='Number of states in P (part-of-speech category) variable. p_size can be specified as a single integer, a range of integers "start:end" (in which case a step size of 1 will be used), or a range of integers with an integer step specification "start:step:end". If left unspecified, defaults to 5.')
argparser.add_argument('-w', '--w_size', dest='w_size', default='2000', action='store', help='Number of states in W (word category) variable. w_size can be specified as a single integer, a range of integers "start:end" (in which case a step size of 1 will be used), or a range of integers with an integer step specification "start:step:end". If left unspecified, defaults to 1.')
argparser.add_argument('-d', '--depth', dest='depth', default=None, action='store', help='Embedding depth bound. Depth can be specified as a single integer, a range of integers "start:end" (in which case a step size of 1 will be used), or a range of integers with an integer step specification "start:step:end". If left unspecified, defaults to 1.')
argparser.add_argument('-B', '--branching', dest='branching', default=None, action='store', help='Branching direction (i.e. left-branching probability). 0 <= branching <= 1, where 0 is totally right-branching and 1 is totally left-branching. Branching can be specified as a single scalar, a range of scalars "start:end" (in which case a step size of 0.1 will be used), or a range with a step specification "start:step:end". If left unspecified, defaults to <None>, and branching probability will be determined empirically (relative number of left and right branching rules).')
argparser.add_argument('-R', '--recursion', dest='recursion', default=None, action='store', help='Recursion probability. 0 <= recursion < 1. Recursion can be specified as a single scalar, a range of scalars "start:end" (in which case a step size of 0.1 will be used), or a range with a step specification "start:step:end". If left unspecified, defaults to <None>, and recursion probability will be determined empirically (relative number of non-recursive rules).')
argparser.add_argument('-r', '--recursion_bound', dest='recursion_bound', default=None, action='store', help='Recursion bound (maximum recursion allowed during tree sampling). Recursion bound can be specified as a single integer, a range of integers "start:end" (in which case a step size of 1 will be used), or a range of integers with an integer step specification "start:step:end". If left unspecified, defaults to <None>, and no recursion bound will be imposed.')
argparser.add_argument('-T', '--termination', dest='termination', default=None, action='store', help='Branch termination (bottom-out) probability. 0 < recursion <= 1. Termination can be specified as a single scalar, a range of scalars "start:end" (in which case a step size of 0.1 will be used), or a range with a step specification "start:step:end". If left unspecified, defaults to <None>, and termination probability will be determined empirically (relative number of non-terminating rules).')
argparser.add_argument('-O', '--oneword', dest='oneword', default=None, action='store', help='Probability of generating a one-word sentence. 0 < oneword <= 1. One-word probability can be specified as a single scalar, a range of scalars "start:end" (in which case a step size of 0.1 will be used), or a range with a step specification "start:step:end". If left unspecified, defaults to <None>, and one-word probability will be determined empirically (relative number of non-terminating rules).')
argparser.add_argument('-A', '--alpha_coef', dest='alpha_coef', default=None, action='store', help='Coefficient by which to multiply the variable size to obtain a concentration parameter used to generate probability distributions via Dirichlet processes (DP). Alpha coefficient can be specified as a single scalar, a range of scalars "start:end" (in which case a step size of 0.1 will be used), or a range with a step specification "start:step:end". Also supports the string value "rand", in which case concentration parameters will be randomly sampled from the uniform distribution for each distribution in the model. If left unspecified, defaults to <None>, and distributions will be generated by sampling random wieghts from the uniform distribution, rather than by DPs.')
args, unknown = argparser.parse_known_args()

def process_int(s):
  s = s.split(':')
  assert len(s) > 0 and len(s) <=3, 'Improperly formatted CLI argument: %s' %s
  if len(s) == 1:
    return [int(s[0])]
  if len(s) == 2:
    return range(int(s[0]), int(s[1]))
  return range(int(s[0]), int(s[2]), int(s[1]))

def process_real(s):
  if s is None:
    return [s]
  s = s.split(':')
  assert len(s) > 0 and len(s) <=3, 'Improperly formatted CLI argument: %s' %s
  if len(s) == 1:
    return [float(s[0])]
  if len(s) == 2:
    return list(np.arange(float(s[0]), float(s[1]), 0.1))
  return list(np.arange(float(s[0]), float(s[2]), float(s[1])))

def int2str(i):
  if i != None:
    s = str(i)
    return s
  return 'None'

def real2str(r):
  if r != None:
    s = '%.10f'%r
    i, d = s.split('.')
    d_out = ''
    cutoff = False
    for c in d:
      if c == '0' and cutoff:
        break
      if int(c) > 0:
        cutoff=True
      d_out += c
    if int(d_out) == 0:
      return i
    return 'p'.join([i,d_out])
  return 'None'

def main():
  args.num_trees = process_int(args.num_trees)
  args.a_size = process_int(args.a_size)
  args.b_size = process_int(args.b_size)
  args.p_size = process_int(args.p_size)
  args.w_size = process_int(args.w_size)
  args.recursion_bound = process_int(args.recursion_bound)
  if args.depth != None:
    args.depth = process_int(args.depth)
  else:
    args.depth = [None]
  args.branching = process_real(args.branching)
  args.recursion = process_real(args.recursion)
  args.termination = process_real(args.termination)
  args.oneword = process_real(args.oneword)
  if args.alpha_coef == 'rand':
    args.alpha_coef = ['rand']
  else:
    args.alpha_coef = process_real(args.alpha_coef)

  if args.dry_run:
    print('Dry run. Would generate the following files:')
    print('')
  for params in [(t,a,b,p,w,d,B,R,r,T,O,A) for t in args.num_trees \
                                   for a in args.a_size \
                                   for b in args.b_size \
                                   for p in args.p_size \
                                   for w in args.w_size \
                                   for d in args.depth \
                                   for B in args.branching \
                                   for R in args.recursion \
                                   for r in args.recursion_bound \
                                   for T in args.termination \
                                   for O in args.oneword \
                                   for A in args.alpha_coef]:
     t, a, b, p, w, d, B, R, r, T, O, A = params

     sys.stderr.write('Generating model and sampling trees using the following configuration:\n')
     sys.stderr.write('  t=%s\n'%t)
     sys.stderr.write('  a=%s\n'%a)
     sys.stderr.write('  b=%s\n'%b)
     sys.stderr.write('  p=%s\n'%p)
     sys.stderr.write('  w=%s\n'%w)
     sys.stderr.write('  d=%s\n'%d)
     sys.stderr.write('  B=%s\n'%B)
     sys.stderr.write('  R=%s\n'%R)
     sys.stderr.write('  r=%s\n'%R)
     sys.stderr.write('  T=%s\n'%T)
     sys.stderr.write('  O=%s\n'%O)
     sys.stderr.write('  A=%s\n'%A)

     d_str = int2str(d)
     B_str = real2str(B)
     R_str = real2str(R)
     r_str = int2str(r)
     T_str = real2str(T)
     O_str = real2str(O)
     if A == 'rand':
       A_str = 'rand'
     else:
       A_str = real2str(A)


     basename = args.output_dir + '/PCFG_t%s_a%s_b%s_p%s_w%s_d%s_B%s_R%s_r%s_T%s_O%s_A%s' %(t, a, b, p, w, d_str, B_str, R_str, r_str, T_str, O_str, A_str)
     
     if args.dry_run:
       print('  ' + basename + '.linetrees')
       print('  ' + basename + '.model')
       print('  ' + basename + '.info.txt')
       print('')
     else:
       sys.stderr.write('Generating probability model...\n')
       gen = PCFG_Generator(a,b,p,w,1,15,B,R,r,T,O,A)
       sys.stderr.write('Sampling trees...\n')
       if args.depth != None:
         trees = gen.generate_trees(t, args.depth)
       else:
         trees = gen.generate_trees(t)

       sys.stderr.write('Writing output...\n')

       with open(basename + '.linetrees', 'wb') as f:
         for t in trees:
           f.write(str(t) + '\n')
       with open(basename + '.model', 'wb') as f:
         f.write(str(gen))
       with open(basename + '.info.txt', 'wb') as f:
         f.write('Random PCFG Info:\n')
         f.write('\n')
         f.write('PCFG model properties:\n')
         f.write('  Number of active (A) categories: %d\n' %a)
         f.write('  Number of awaited (B) categories: %d\n' %b)
         f.write('  Number of part-of-speech (P) categories: %d\n' %p)
         f.write('  Vocabulary size (W): %d\n' %w)
         if B != None:
           f.write('  Branching preference (left-branching probability): %s\n' %B)
         else:
           f.write('  No branching preference specified.\n')
         if R != None:
           f.write('  Recursion probability: %s\n' %R)
         else:
           f.write('  No recursion probability specified.\n')
         if T != None:
           f.write('  Branch termination (bottom-out) probability: %s\n' %T)
         else:
           f.write('  No branch termination (bottom-out) probability specified.\n')
         if O != None:
           f.write('  One-word sentence probability: %s\n' %O)
         else:
           f.write('  No one-word sentence probability specified.\n')
         if A != None:
           f.write('  Alpha coefficient: %s\n' %A)
         else:
           f.write('  No one-word alpha coefficient specified.\n')
         f.write('\n')
         f.write('Sampled trees properties:\n')
         if d != None:
           f.write('  Maximum allowed embedding depth for tree samples: %d\n' %d)
         else:
           f.write('  No limit on embedding depth.\n')
         if r != None:
           f.write('  Maximum allowed recursion depth for tree samples: %d\n' %r)
         else:
           f.write('  No limit on recursion depth.\n')
    
       sys.stderr.write('PCFG generation and tree sampling complete.\n\n')



main()


