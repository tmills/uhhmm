
#include "HmmSampler.h"
#include <stdio.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
#include <thrust/extrema.h>
#include <tuple>
#include <ctime>
#include <utility>
#include <cmath>
#include <algorithm>
#include "cusp/array1d.h"
#include "cusp/array2d.h"
#include "cusp/csr_matrix.h"
#include "cusp/blas/blas.h"
#include "cusp/multiply.h"
#include "cusp/elementwise.h"
#include "cusp/functional.h"
#include "cusp/print.h"
#include "myarrays.cu"
#include "Indexer.cu"
#include "State.cu"
#include <chrono> // for testing
#include <random>
#include <limits>
#include <iostream>
#include <iomanip>
using namespace cusp;
using namespace std;
typedef std::chrono::high_resolution_clock Clock;
float nano_to_sec = 1.0e-9f;
__device__ int G_SIZE;
typedef std::numeric_limits<float> float_limit;
typedef cusp::array1d<float,cusp::device_memory> cusparray;
typedef cusparray::view ArrayView;

Model::Model(int pi_num_rows, int pi_num_cols, float* pi_vals, int pi_vals_size, int* pi_row_offsets,
int pi_row_offsets_size, int* pi_col_indices, int pi_col_indices_size, float* lex_vals, int lex_vals_size,
int lex_num_rows, int lex_num_cols, int a_max, int b_max, int g_max, int depth, float* pos_vals, int pos_vals_size,
int EOS_index):
pi_num_rows(pi_num_rows), pi_num_cols(pi_num_cols), pi_vals(pi_vals),
pi_vals_size(pi_vals_size), pi_row_offsets(pi_row_offsets), pi_row_offsets_size(pi_row_offsets_size),
pi_col_indices(pi_col_indices), pi_col_indices_size(pi_col_indices_size), lex_vals(lex_vals),
lex_vals_size(lex_vals_size), lex_num_rows(lex_num_rows), lex_num_cols(lex_num_cols), a_max(a_max),
b_max(b_max), g_max(g_max), depth(depth), pos_vals(pos_vals), pos_vals_size(pos_vals_size), EOS_index(EOS_index){
    pi = new SparseView(pi_num_rows, pi_num_cols, pi_vals_size, pi_row_offsets, pi_col_indices, pi_vals, pi_row_offsets_size, pi_col_indices_size);
    lex = new DenseView(lex_num_rows, lex_num_cols, lex_vals_size, lex_vals);
    pos = new Array(pos_vals, pos_vals_size);
}

Model::~Model(){
    delete lex;
    delete pi;
    delete pos;
}

int Model::get_depth(){
    return depth;
}

// taken away the new tensor
template <class AView>
int HmmSampler::get_sample(AView &v){
    float dart;
//     cout << "get_sample()" << endl;
    thrust::inclusive_scan(thrust::device, v.begin(), v.end(), sum_dict->begin());
    // cout << "v[0] = " << v[0] << "v[-1] = " << v[v.size()-1] << " with length " << v.size() << endl;
//     cout << "sum_dict[0] = " << (*sum_dict)[0] << "sum_dict[-1] = " << (*sum_dict)[sum_dict->size()-1] << " with length: " << sum_dict->size() <<  endl;
    //cusp::print(*sum_dict);
//     cout << "sum done" << endl;
    int dart_target;
    int condition = 1;
    while (condition) {
        dart = dist(mt);
        if (dart != 0.0f and dart != 1.0f) {
//            cout << "dart: "<< scientific << dart << endl;
           dart_target = thrust::upper_bound(thrust::device, sum_dict->begin(), sum_dict->end(), dart) - sum_dict->begin();
           if (dart_target == 0) {
           print(*sum_dict);
           }
//           cout << "dart target (summed): " << dart_target << " " << scientific << (*sum_dict)[dart_target - 1] << endl; //<< " " << scientific << (*sum_dict)[dart_target] << " " << scientific  << (*sum_dict)[dart_target+ 1] << endl;
//           cout << "dart target (summed): " << dart_target << " " << scientific << v[dart_target - 1] << " " << scientific << v[dart_target] << " " << scientific  << v[dart_target+ 1] << endl;
//            cout << "dart target (summed): " << dart_target << " with v size=" << v.size() << endl;
           if (dart_target != v.size() && v[dart_target] != 0.0f ){
                condition = 0;
//                cout << "out of loop" << endl;
           }
        }
     }
//      cout << "dart_target: " << dart_target << endl;
     return dart_target;
}
template <class AView>
int HmmSampler::get_max(AView &v){
    int dart_target = thrust::max_element(thrust::device, v.begin(), v.end()) - v.begin();
    return dart_target;
}

class cuda_exp : public thrust::unary_function<float, float>{
public:
    __device__
   float operator()(float x){
       return pow(10,x);
   }
};
class calc_sparse_column : public thrust::unary_function<int, int>{
public:
   __device__
   int operator()(int x){
       return x % G_SIZE;
   }
};

class g_integer_division : public thrust::unary_function<int, int>{
public:
    __device__
   int operator()(int x){
       return x / G_SIZE;
   }
};

template <class AView>
void exp_array(AView & v){
   cuda_exp x;
   thrust::transform(v.begin(), v.end(), v.begin(), x);
}

// this gives a TRANSPOSED version of a tile matrix for computation convenience
Sparse * tile(int g_len, int state_size){

//     cout << "5.1" << endl;
    int nnz = state_size;
    // cout << g_len << " " << state_size << " " << nnz << endl;
    Sparse * temp_mat = new Sparse(state_size, g_len, nnz); //TRANSPOSED
    blas::fill(temp_mat->values, 1.0f);
//     cout << "5.4" << endl;
    calc_sparse_column temp_func;
    thrust::tabulate(thrust::device, temp_mat->column_indices.begin(), temp_mat->column_indices.end(), temp_func);
    thrust::sequence(thrust::device, temp_mat->row_offsets.begin(), temp_mat->row_offsets.end());
//     cout << "5.5" << endl;

    return temp_mat;
}

// this gives a TRANSPOSED version of matrix repetition for computation convenience
Sparse * expand(int g_len, int state_size){

//     cout << "expand5.1" << endl;
    int col_size = state_size / g_len;
//     cout << "expand5.2" << endl;
    int nnz = state_size;
//     cout << "expand5.3" << endl;
    Sparse * temp_mat = new Sparse(state_size, col_size, nnz); //TRANSPOSED
    blas::fill(temp_mat->values, 1.0f);
//     cout << "expand5.4" << endl;
    g_integer_division temp_func;
    thrust::tabulate(thrust::device, temp_mat->column_indices.begin(), temp_mat->column_indices.end(), temp_func);
    thrust::sequence(thrust::device, temp_mat->row_offsets.begin(), temp_mat->row_offsets.end());
//     cout << "expand5.5" << endl;
    return temp_mat;
}

void get_row(csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView>* s, int i, Array& result,
Array* pos_full_array, int g_max, int b_max){
    blas::fill(result, 0.0f);
    int pi_row_index = i / g_max;
    int b_index = pi_row_index % b_max;
    int g_index = i % g_max;
//    cout << pi_row_index << "pi row index" << endl;
    if (s->row_offsets[pi_row_index] - s->row_offsets[pi_row_index+1] != 0){
        int num_of_entry = s->row_offsets[pi_row_index+1] - s->row_offsets[pi_row_index];
        int start_column_index = s->row_offsets[pi_row_index];
        IndexArrayView column_indices = s -> column_indices.subarray(start_column_index, num_of_entry);
        ValueArrayView values = s -> values.subarray(start_column_index, num_of_entry);
        thrust::scatter(thrust::device, values.begin(), values.end(), column_indices.begin(), result.begin());
//        cusp::print(result);
    }
//    cout << "get row out" << endl;
}

int get_max_len(std::vector<std::vector<int> > sents){
//    cout <<"get_max_len in"<< endl;
    int max_len = 0;
    for(std::vector<int> sent : sents){
        if(sent.size() > max_len){
            max_len = sent.size();
        }
    }
//    cout << "get_max_len out" << endl;

    return max_len;
}

Dense* get_sentence_array(std::vector<std::vector<int> > sents, int max_len){
    Dense* array = new Dense( sents.size(), max_len, 0 );
//    cout <<"get_sentence_array in"<< endl;
    for(int i = 0; i < sents.size(); i++){
      std::vector<int> sent = sents[i];
      for(int token_ind = 0; token_ind < sent.size(); token_ind++){
        array -> operator()(i, token_ind) = sent[token_ind];
      }
    }
//    cout <<"get_sentence_array out"<< endl;
    return array;

}

void HmmSampler::set_models(Model * models){

//     cout << "set_models 1" << endl;
    p_model = models;
//     cout << "set_models 2" << endl;
    if (p_indexer != NULL){
        delete p_indexer;
	p_indexer = NULL;
    }
    p_indexer = new Indexer(models);
//     cout << "set_models 3" << endl;
    int g_len = p_model-> g_max;
    int state_size = p_indexer -> get_state_size();
    cudaMemset(&G_SIZE,0,sizeof(int));
    cudaMemcpyToSymbol(G_SIZE, &g_len, sizeof(int), 0, cudaMemcpyHostToDevice);
//     cout << "set_models 4" << endl;
    lexMatrix = p_model -> lex;
//     cout << "set_models 5" << endl;
    pos_matrix = p_model -> pos;
//    cusp::print(*pos_matrix);
    if (lexMultiplier != NULL){
        delete lexMultiplier;
        lexMultiplier = NULL;
    }
    lexMultiplier = tile(g_len, p_indexer->get_state_size());
//    cout << "set_models 6" << endl;
    expand_mat = expand(g_len, p_indexer->get_state_size());
//    cout << "set_models 7" << endl;
    pi = p_model -> pi;
    // print( *(pi->get_view()) );
    if (trans_slice != NULL){
        delete trans_slice;
        trans_slice = NULL;
    }
    trans_slice = new Array(p_indexer->get_state_size(), 0.0f);
//    cout << "set_models 8" << endl;
    expanded_lex = trans_slice;
    sum_dict = trans_slice;
    int b_len = p_model -> b_max;
//    cout << "pos full array" << endl;
    int depth = p_model -> get_depth();
    pos_full_array = make_pos_full_array(pos_matrix, g_len, b_len, depth, state_size);
//    cout << "set_models 9" << endl;
    // cout.precision(float_limit::max_digits10);
}

Array* HmmSampler::make_pos_full_array(Array* pos_matrix ,int g_max, int b_max, int depth, int state_size){
//    cout << "make_pos_full_array in" << endl;
    int pos_matrix_size = pow(b_max, depth) * g_max * 2;
    int copy_times = state_size / pos_matrix_size;
    Array* temp_array = new Array(state_size, 0.0f);
    for (int i = 0; i < copy_times; i ++){
//        cout << "calc full pos " << i  << " from " << i*pos_matrix_size << " to " << (i+1)*pos_matrix_size <<
//        " total " << copy_times << endl;
        ArrayView one_section_of_array(temp_array->subarray(i*pos_matrix_size, pos_matrix_size));
        copy(*pos_matrix, one_section_of_array);
    }
//    cout << " THE VALUE " << (*temp_array)[26426] << endl;
    return temp_array;
}

void HmmSampler::initialize_dynprog(int batch_size, int max_len){
    try{
//    cout << "initialize_dynprog in" << endl;
    sampler_batch_size = batch_size;
    max_sent_len = max_len;
    dyn_prog = new Dense*[max_len];
    for(int i = 0; i < max_len; i++){
        dyn_prog[i] = new Dense(p_indexer->get_state_size(), batch_size, 0.0f);
    }
//    cout << "initialize_dynprog 2" << endl;
    start_state = new Dense(p_indexer->get_state_size(), batch_size, 0.0f);
    for(int i = 0; i < batch_size; i++){
        start_state->operator()(0, i) = 1;
    }
    int a_max, b_max, g_max;
//    cout << "initialize_dynprog 3" << endl;
    std::tie(a_max, b_max, g_max) = p_indexer -> getVariableMaxes();
    int state_size_no_g = p_indexer->get_state_size() / g_max;
    dyn_prog_part = new Dense(state_size_no_g, batch_size, 0.0f);
//    cout << "initialize_dynprog out" << endl;
    } catch (...) {
        cout << "init dynprog error!" << endl;
        throw;
    }
}

void HmmSampler::g_factored_multiply(Dense* prev_dyn_prog_slice, Dense* this_dyn_prog_slice){
//    cout << "factor multiply 1" << endl;
    csr_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView>* curtailed_transition = pi -> get_view();
    multiply(*curtailed_transition, *prev_dyn_prog_slice, *dyn_prog_part);
//    cout << "dyn prog part" << endl;
    int a_max, b_max, g_max;
    std::tie(a_max, b_max, g_max) = p_indexer -> getVariableMaxes();
    int state_size_no_g = p_indexer->get_state_size() / g_max;
    int state_size = p_indexer->get_state_size();
    multiply(*expand_mat, *dyn_prog_part, *this_dyn_prog_slice);
//    cout << "factor multiply 3" << endl;
    for(int i = 0; i < sampler_batch_size; i++ ){ // this is not efficient. maybe there is a better way
//        cout << "insde g_multipy i " << i << endl;
        array2d<float, device_memory>::column_view this_dyn_prog_slice_column = this_dyn_prog_slice->column(i);
        blas::xmy(this_dyn_prog_slice_column, *pos_full_array ,this_dyn_prog_slice_column);
//        cout << "this dyn prog slice column" << endl;

    }
//    cout << "factor multiply 4" << endl;
}


std::vector<float> HmmSampler::forward_pass(std::vector<std::vector<int> > sents, int sent_index){
//     cout << "Forward in " << endl;
    float normalizer;
    int a_max, b_max, g_max; // index, token, g_len;
    std::tie(a_max, b_max, g_max) = p_indexer -> getVariableMaxes();
    std::vector<int> sent = sents[0];
    std::vector<float> log_probs;
    int batch_size = sents.size();
    int batch_max_len = get_max_len(sents);
    // np_sents is |batches| x max_len
//    Dense* np_sents = get_sentence_array(sents, batch_max_len);
    csr_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView>* pi_view = pi -> get_view();
    
    array2d_view<ValueArrayView, row_major>* lex_view = lexMatrix -> get_view();
//    cout << "Forward in 2 " << endl;
    for(int sent_ind = 0; sent_ind < sents.size(); sent_ind++){
        log_probs.push_back(0);
    }
    
    
    for(int ind = 0; ind < batch_max_len; ind++){
//        cout << "Processing token index " << ind << " for " << batch_size << " sentences." << endl;
        Dense *cur_mat = dyn_prog[ind];
        Dense *prev_mat;
        
        if(ind == 0){
            prev_mat = start_state;
        }else{
            // Grab the ind-1th row of dyn_prog and multiply it by the transition matrix and put it in
            // the ind^th row.
            prev_mat = dyn_prog[ind-1];
        }            

        // prev_mat is |states| x |batches| at time ind-1
        // pi_view is |states| x |states| transition matrix with time t on rows and t-1 on columns (i.e. transposed)
        // so after this multiply cur_mat is |states| x |batches| incorporating transition probabilities
        // but not evidence
//        cout << "Performing transition multiplication" << endl;
        g_factored_multiply(prev_mat, cur_mat);
//        cout << "performing observation multiplications" << endl;

        auto trans_done = Clock::now();
        
        // for now incorporate the evidence sentence-by-sentence:
        for(int sent_ind = 0; sent_ind < sents.size(); sent_ind++){
            // not every sentence in the batch will need the full batch size
            if(sents[sent_ind].size() <= ind){
                continue;
            }
//            cout << "Processing sentence index " << sent_ind << endl;
            int token = sents[sent_ind][ind];
                
            // lex_column is |g| x 1 
            array2d<float, device_memory>::column_view lex_column = lex_view -> column(token);
            // lexMultiplier is state_size x |g|, expanded_lex is state_size x 1
//             cout << "Multiplying lex multiplier by lex column" << endl;
            multiply(* lexMultiplier, lex_column, * expanded_lex);
//             cout << "lex finished" << endl;
            // dyn_prog_row is 1 x state_size
            // dyn_prog_column is state_size x 1
            array2d<float, device_memory>::column_view dyn_prog_col = cur_mat->column(sent_ind);
//             cout << "Multiplying expanded_lex by dyn prog row" << endl;
            blas::xmy(*expanded_lex, dyn_prog_col, dyn_prog_col);
//             cout << "Computing normalizer" << endl;
            normalizer = thrust::reduce(thrust::device, dyn_prog_col.begin(), dyn_prog_col.end());
//             cout << "Normalizing over col with result: " << normalizer << endl;
            blas::scal(dyn_prog_col, 1.0f/normalizer);
//                cout << "Adding logged normalizer to sentence logprobs" << endl;
            log_probs[sent_ind] += log10f(normalizer);
//            cout << " ind "<< ind << " sent_ind " << sent_ind << " token " << token << " log prob " << log10f(normalizer) << " " << normalizer << endl;
            if (sents[sent_ind].size() - 1 == ind){
                int EOS = p_indexer -> get_EOS_full();
//                cout << EOS << endl;
                array2d<float, device_memory>::column_view final_dyn_col = cur_mat->column(sent_ind);
                get_row(pi->get_view(), EOS, *trans_slice, pos_full_array, g_max, b_max);
                float final_normalizer = cusp::blas::dot(*trans_slice, final_dyn_col);
                log_probs[sent_ind] += log10f(final_normalizer);
//                cout << " ind "<< ind << " sent_ind " << sent_ind << "end log prob " << log10f(final_normalizer) << endl;
            }
        }
        
        auto norm_done = Clock::now();
    }
//    cout << "Finished forward pass (cuda) and returning vector with " << log_probs.size() << " elements." << endl;
    return log_probs;
}

std::vector<std::vector<State> > HmmSampler::reverse_sample(std::vector<std::vector<int>> sents, int sent_index, int viterbi=0){
//     cout << "Reverse sampling batch with starting sent index of " << sent_index << endl;
    std::vector<std::vector<State>> sample_seqs;
    std::vector<State> sample_seq;
    std::vector<int> sample_t_seq;
    int last_index, sample_t, sample_depth; // , t, ind; totalK, depth,
    int batch_size = sents.size();
    int batch_max_len = get_max_len(sents);
    State sample_state;
    for(int sent_ind = 0; sent_ind < batch_size; sent_ind++){
        sample_seq = std::vector<State>();
//        cout << "Processing sentence " << sent_ind << " of the batch" << endl;
        std::vector<int> sent = sents[sent_ind];
        // Start with EOS
//        if (sent.size() == 1) {
//            sample_t = p_indexer->get_EOS_1wrd_full();
//        } else {
            sample_t = p_indexer->get_EOS_full(); 
//        }

        for (int t = sent.size() - 1; t > -1; t --){
//            cout << "t" << t << " prev sample t is " << sample_t <<endl;
            std::tie(sample_state, sample_t) = _reverse_sample_inner(sample_t, t, sent_ind, viterbi);
//             cout << "Sample t is " << sample_t << endl;
//             cout << sample_state.f << " " << sample_state.j << " " << sample_state.a[0] << " " << sample_state.a[1] << " " << sample_state.b[0] << " " << sample_state.b[1] << " " << sample_state.g << endl;
            if(!sample_state.depth_check()){
              cout << "Depth error in state assigned at index" << t << endl;
            }
            sample_seq.push_back(sample_state);
            sample_t_seq.push_back(sample_t);
        }
        // auto t4 = Clock::now();
        std::reverse(sample_seq.begin(), sample_seq.end());
//        cout << "finished backward sampling of a sentence" << endl;
        sample_seqs.push_back(sample_seq);
    }
//    for(auto i : sample_t_seq){
//        cout << i << endl;
//    }
//    cout << "Done with reverse()" << endl;
    return sample_seqs;
}


std::tuple<State, int> HmmSampler::_reverse_sample_inner(int& sample_t, int& t, int sent_ind, int viterbi=0){
    float normalizer;
//    cout << "before get row" << endl;
    int g_max =  p_model -> g_max;
    int b_max = p_model -> b_max;
    get_row(pi->get_view(), sample_t, *trans_slice, pos_full_array, g_max, b_max);
    // print(*trans_slice);
    array2d<float, device_memory>::column_view dyn_prog_col = dyn_prog[t]->column(sent_ind);
    float dyn_prog_col_sum = thrust::reduce(thrust::device, dyn_prog_col.begin(), dyn_prog_col.end());
//     cout << "Dyn_prog_row " << dyn_prog_col_sum << endl;

    float trans_slice_sum = thrust::reduce(thrust::device, (*trans_slice).begin(), (*trans_slice).end());
//    cout << "trans_slice " << trans_slice_sum << endl;
    cusp::blas::xmy(*trans_slice, dyn_prog_col, dyn_prog_col);
    normalizer = thrust::reduce(thrust::device, dyn_prog_col.begin(), dyn_prog_col.end());
//     cout << "normalizer " << normalizer <<endl;
    blas::scal(dyn_prog_col, 1.0f/normalizer);
//    cout << "viterbi is " << viterbi << endl;
    if (viterbi == 0){
        sample_t = get_sample(dyn_prog_col);
    } else {
        sample_t = get_max(dyn_prog_col);
    }
//    cout << "sample_t "<< sample_t << endl;
    State sample_state = p_indexer -> extractState(sample_t);
    return std::make_tuple(sample_state, sample_t);
}

std::tuple<std::vector<std::vector<State> >, std::vector<float>> HmmSampler::sample(std::vector<std::vector<int>> sents, int sent_index, int viterbi=0) {
    std::vector<float> log_probs;
    std::vector<std::vector<State> > states;

    try{
        log_probs = forward_pass(sents, sent_index);
    }catch(thrust::system_error &e){
        cerr << "Error in forward pass: " << e.what() << endl;
        throw e;
    }
    try{
        states = reverse_sample(sents, sent_index, viterbi);
    }catch(thrust::system_error &e){
        cerr << "Error in reverse sample: " << e.what() << endl;
        throw e;
    }
    return std::make_tuple(states, log_probs);
}

HmmSampler::HmmSampler() : seed(std::time(0)){
    mt.seed(seed);
}

HmmSampler::HmmSampler(int seed) : seed(seed){
    if (seed == 0){
        mt.seed(std::time(0));
    } else{
        mt.seed(seed);
    }
}
HmmSampler::~HmmSampler(){
    if(dyn_prog != NULL){
        for(int i = 0; i < max_sent_len; i++){
            delete dyn_prog[i];
        }
        delete[] dyn_prog;
    }
    delete start_state;
    delete p_indexer;
    delete lexMultiplier;
    delete trans_slice;
    delete expand_mat;
    delete dyn_prog_part;
    delete pos_full_array;

}
