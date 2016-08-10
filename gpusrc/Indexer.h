
#ifndef Indexer_hpp
#define Indexer_hpp

#include <vector>
#include <tuple>
//#include "Model.h"
//#include "State.h"

using namespace std;

vector<int> unravel_index(int index, vector<int> & v);

class Indexer{
public:
    Indexer();
    Indexer(Model * model);
    tuple<int, int, int> getVariableMaxes();
    int get_state_size();
    tuple<int, int, vector<int>, vector<int>, int> extractStacks(int index);
    State extractState(int index);
private:
    int depth;
    int fj_size;
    int a_max;
    int b_max;
    int g_max;
    int state_size;
    int a_size;
    int b_size;
    vector<int> stack_dims;
};

#endif /* Indexer_hpp */