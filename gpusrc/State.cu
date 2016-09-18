
//#include "State.h"
#include <vector>
#include <iostream>

using namespace std;

State::State(){}
State::State(int d): depth(d), f(-1), j(-1), g(0), a(depth, 0), b(depth, 0) {
    // cout << b.capacity() << endl;
}
State::State(int d, State state) : depth(d), f(state.f), j(state.j), a(state.a), b(state.b), g(state.g){
    cout << "should not call this" << endl;
}

int State::max_awa_depth(){
    // cout << "beginning the function" << endl;
    // cout << "b[0] should be " << b[0] << endl;
    if (b[0] == 0) {
        // cout << "getting in loop" << endl;
        return -1;
    }
    // cout << "getting in second loop" << endl;
    for (int d = 1; d < depth; d++) {
        if (b[d] == 0){
            return d - 1;
        }
    }
    return depth - 1;
}

//int main(){
//    State x = State(3);
//    cout << x.max_awa_depth() << endl;
//    return 0;
//}
