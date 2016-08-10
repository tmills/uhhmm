
#ifndef State_h
#define State_h

#include <vector>

class State{
public:
    State();
    State(int d);
    State(int d, State state);

    int max_awa_depth();
    
    int depth;
    int f;
    int j;    
    std::vector<int> a;
    std::vector<int> b;
    int g;
};
#endif /* State_hpp */
