
#ifndef HmmSampler_hpp
#define HmmSampler_hpp

//#include "cusp/array2d.h"
//#include "cusp/array1d.h"
//#include "cusp/csr_matrix.h"
//#include "Model.h"
//#include "Indexer.h"
#include <vector>
#include <tuple>
#include <random>
//#include "State.h"
//using namespace cusp;

//csr_matrix<int, float, device_memory> * tile(int g_len, int state_size);
//int get_sample(array1d<float, device_memory>::view &v);
//void exp_array(array1d<float, device_memory>::value & v);
class SparseView;
class DenseView;
class Dense;
class Array;
class Sparse;
class Indexer;
//state class
class State{
public:
    State();
    State(int d);
    State(int d, State state);
    
    int max_awa_depth();
    int max_act_depth();
    bool depth_check();

    int depth;
    int f;
    int j;
    std::vector<int> a;
    std::vector<int> b;
    int g;
};
// model class
class Model{
public:
    Model(int pi_num_rows,int pi_num_cols,float* pi_vals, int pi_vals_size, int* pi_row_offsets, int pi_row_offsets_size, int* pi_col_indices, int pi_col_indices_size, float* lex_vals, int lex_vals_size, int lex_num_rows, int lex_num_cols, int a_max, int b_max, int g_max, int depth);
    ~Model();
    int get_depth();
    int pi_num_rows;
    int pi_num_cols;
    float* pi_vals;
    int pi_vals_size;
    int* pi_row_offsets;
    int pi_row_offsets_size;
    int* pi_col_indices;
    int pi_col_indices_size;
    // these are precalculated for indexer
    int a_max;
    int b_max;
    int g_max;
    float* lex_vals;
    int lex_vals_size;
    int lex_num_rows;
    int lex_num_cols;
    const unsigned int depth;
    DenseView * lex;
    SparseView * pi;
};


class HmmSampler{
public:
    HmmSampler(int seed);
    HmmSampler();
    ~HmmSampler();
    void set_models(Model * models);
    void initialize_dynprog(int max_len);
    float forward_pass(std::vector<int> sent, int sent_index);
    std::vector<State> reverse_sample(std::vector<int> sent, int sent_index);
    std::tuple<std::vector<State>, float> sample(std::vector<int> sent, int sent_index);
    template <class AView>
    int get_sample(AView &v);
private:
    std::tuple<State, int> _reverse_sample_inner(int& sample_t, int& t);
    Model * p_model = NULL;
    Indexer * p_indexer = NULL;
    DenseView* lexMatrix = NULL;
    Dense* dyn_prog = NULL;
    Sparse* lexMultiplier = NULL;
    SparseView* pi = NULL;
    Array* trans_slice = NULL;
    int seed;
    Array* expanded_lex = NULL;
    Array* sum_dict = NULL;
    std::mt19937 mt;
    std::uniform_real_distribution<float> dist{0.0f,1.0f};
    
};

#endif /* HmmSampler_hpp */
