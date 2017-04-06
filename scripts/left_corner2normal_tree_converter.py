from State import Statefrom nltk import treeimport reimport copy"""normalize the sequences if the dateline is weird"""ordering = ['j', 'a', 'b', 'f', 'p']## given a string of parse states, parse them into separate State objectsdef parse_state_seq(state_seq, depth):    single_states = state_seq.split(' ')    states = []    words = []    for state_s in single_states:        state = parse_state(state_s, depth)        states.append(state[0])        words.append(state[1])    return states, words## given a string of a single parse state in raw order, i.e. -::ACT0/AWA0::+::POS0## parse it into a State object following some ordering defined above in orderingdef parse_state(state_string, depth):    state = State(depth)    word = None    state_string = re.split(':+', state_string)    d = 0    for item in state_string:        if item == '-' or item == '+':            item = item == '+' and 1 or 0            for variable in ordering:                if (variable == 'f' or variable == 'j') and getattr(state, variable) == -1:                    setattr(state, variable, int(item))                    break        elif 'ACT' in item:            items = item.split(';')            # print(items)            for single_ab_pair in items:                single_ab = single_ab_pair.split('/')                state.a[d] = int(single_ab[0].replace('ACT', ''))                state.b[d] = int(single_ab[1].replace('AWA', ''))                d += 1        elif 'POS' in item:            state.g = int(item.replace('POS', ''))        else:            word = item    return state, word## normalize the dateline according to the ordering above given a sequence## the normal dateline ordering is F J P A B## the algorithm finds the first A in a sequence and moveall variables accordinglydef normalize_dateline(parse_states, ordering=ordering):    # normal_ordering is 'f', 'j', 'p' , 'a', 'b'    for index, variable in enumerate(ordering):        if variable == 'f':            break    if index == 0: # new ordering is the same as the old ordering, do nothing        return parse_states    else:        normed_states = copy.deepcopy(parse_states)        for index, state in enumerate(normed_states):            if index == len(normed_states) - 1:                next_state = State(len(state.a))                next_state.j = 1            else:                next_state = normed_states[index+1]            for variable in ordering:                if variable != 'f':                    setattr(state, variable, getattr(next_state, variable))                else:                    break        return normed_states## convert a normalized state sequence into a nltk treedef convert_states_into_tree(state_seq, word_seq):    holes = []    cur_acts = []    # string = ' '.join([state.str() for state in state_seq])    # print(string)    if len(state_seq) == 1:        return tree.Tree(state_seq[0].g, [word_seq[0]])    for index, state in enumerate(state_seq):        # print(holes)        new_lex = tree.Tree(state.g, [word_seq[index]])        if state.f == 1 and state.j == 0:            new_lex = tree.Tree(state.g, [word_seq[index]])            new_depth = tree.Tree(state.a[state.max_act_depth()], [new_lex, state.b[state.max_act_depth()]])            holes.append(new_depth)            cur_acts.append(new_depth)        elif state.f == 1 and state.j == 1:            new_subtree = tree.Tree(cur_acts[-1][1], [new_lex, state.b[state.max_act_depth()]])            cur_acts[-1][1] = new_subtree            cur_acts[-1] = new_subtree        elif state.f == 0 and state.j == 0:            cur_acts[-1][1] = new_lex            new_subtree = tree.Tree(state.a[state.max_act_depth()], [holes[-1], state.b[state.max_act_depth()]])            holes[-1] = new_subtree            cur_acts[-1] = new_subtree        elif state.f == 0 and state.j == 1:            if len(holes) > 1:                cur_acts[-1][1] = new_lex                new_subtree = tree.Tree(cur_acts[-2][1], [holes[-1], state.b[state.max_act_depth()]])                del cur_acts[-1]                cur_acts[-1][1] = new_subtree                del holes[-1]                holes[-1] = cur_acts[-1]                cur_acts[-1] = new_subtree            else:                cur_acts[-1][1] = new_lex                assert len(holes) == 1, "Illegal sequence of states!"                return holes[0]    raise Exception("Illegal sequence of states!")# debuggingdef main():    test_string = "-::ACT0/AWA0::+::POS3::how -::ACT5/AWA5::+::POS5::about -::ACT5/AWA5;ACT5/AWA1::-::POS1::a -::ACT5/AWA5;ACT1/AWA4::-::POS4::bit +::ACT5/AWA1::-::POS1::of -::ACT1/AWA4::-::POS4::peanut -::ACT1/AWA4::-::POS4::butter -::ACT2/AWA2::-::POS2::?"    tree_s = """(2  (1    (1      (5 (3 how) (5 (1 (5 (5 about) (1 a)) (4 bit)) (1 of)))      (4 peanut))    (4 butter))  (2 ?))"""    state_seq, word_seq = parse_state_seq(test_string,2)    state_seq = normalize_dateline(state_seq)    for index,state in enumerate(state_seq):        print(state.str(), word_seq[index])    tree = convert_states_into_tree(state_seq, word_seq)    assert str(tree) == tree_sif __name__ == '__main__':    main()