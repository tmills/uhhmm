
#include "HmmSampler.h"
#include <stdio.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
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
__device__ int STATE_SIZE;
typedef std::numeric_limits<float> float_limit;

Model::Model(int pi_num_rows, int pi_num_cols, float* pi_vals, int pi_vals_size, int* pi_row_offsets, int pi_row_offsets_size, int* pi_col_indices, int pi_col_indices_size, float* lex_vals, int lex_vals_size, int lex_num_rows, int lex_num_cols, int a_max, int b_max, int g_max, int depth): pi_num_rows(pi_num_rows), pi_num_cols(pi_num_cols), pi_vals(pi_vals), pi_vals_size(pi_vals_size), pi_row_offsets(pi_row_offsets), pi_row_offsets_size(pi_row_offsets_size), pi_col_indices(pi_col_indices), pi_col_indices_size(pi_col_indices_size), lex_vals(lex_vals), lex_vals_size(lex_vals_size), lex_num_rows(lex_num_rows), lex_num_cols(lex_num_cols), a_max(a_max), b_max(b_max), g_max(g_max), depth(depth){
    pi = new SparseView(pi_num_rows, pi_num_cols, pi_vals_size, pi_row_offsets, pi_col_indices, pi_vals, pi_row_offsets_size, pi_col_indices_size);
    lex = new DenseView(lex_num_rows, lex_num_cols, lex_vals_size, lex_vals);
}

Model::~Model(){
    delete lex;
    delete pi;
}

int Model::get_depth(){
    return depth;
}

// taken away the new tensor
template <class AView>
int HmmSampler::get_sample(AView &v){
    float dart;
    // cout << "get_sample()" << endl;
    // array1d<float, device_memory> sum_dict(v.size()); // building a new array, maybe not needed
    // this is the equivalent of np.cumsum() or partial_sum in the stl:
    thrust::inclusive_scan(thrust::device, v.begin(), v.end(), sum_dict->begin());
    // cout << "v[0] = " << v[0] << "v[-1] = " << v[v.size()-1] << " with length " << v.size() << endl;
    // cout << "sum_dict[0] = " << (*sum_dict)[0] << "sum_dict[-1] = " << (*sum_dict)[sum_dict->size()-1] << " with length: " << sum_dict->size() <<  endl;
    //cusp::print(*sum_dict);
    // cout << "sum done" << endl; 
    // dart =  static_cast <float> (rand()) / static_cast <float> (RAND_MAX);// / RAND_MAX;
    int dart_target;
    int condition = 1;
    while (condition) {
        dart = dist(mt);
        if (dart != 0.0f and dart != 1.0f) {
           // cout << "dart: "<< scientific << dart << endl;
           dart_target = thrust::upper_bound(thrust::device, sum_dict->begin(), sum_dict->end(), dart) - sum_dict->begin();
           // cout << "dart target (summed): " << dart_target << " " << scientific << (*sum_dict)[dart_target - 1] << " " << scientific << (*sum_dict)[dart_target] << " " << scientific  << (*sum_dict)[dart_target+ 1] << endl; 
           // cout << "dart target (summed): " << dart_target << " with v size=" << v.size() << endl;
           // float minus_one = (*sum_dict)[dart_target - 1];
           // printf( "%A" , minus_one);
           // cout  << " ";
           // float on_target = (*sum_dict)[dart_target];
           // printf("%A", on_target);
           // cout << " ";
           // float plus_one = (*sum_dict)[dart_target+ 1];
           // printf("%A\n", plus_one); 
           if (v[dart_target] != 0.0f && dart_target != v.size()){
                condition = 0;
           }
        }
     }
     // cout << "get_sample() done." << endl;
     return dart_target;
}

class cuda_exp : public thrust::unary_function<float, float>{
public:
   __host__ __device__
   float operator()(float x){
       return pow(10,x);
   }
};
class calc_sparse_column : public thrust::unary_function<int, int>{
public:
   __host__ __device__
   int operator()(int x){
       return x % STATE_SIZE;
   }
};
template <class AView>
void exp_array(AView & v){
   cuda_exp x;
   thrust::transform(v.begin(), v.end(), v.begin(), x);
}

// this gives a TRANSPOSED version of a tile matrix for computation convenience
Sparse * tile(int g_len, int state_size){

    // cout << "5.1" << endl;
    int copy_times = state_size / g_len;
    // cout << "5.2" << endl;
    int nnz = g_len * copy_times;
    // cout << "5.3" << endl;
    // cout << g_len << " " << state_size << " " << nnz << endl;
    Sparse * temp_mat = new Sparse(state_size, g_len, nnz); //TRANSPOSED
    blas::fill(temp_mat->values, 1.0f);
    // cout << "5.4" << endl;
    calc_sparse_column temp_func;
    thrust::tabulate(thrust::device, temp_mat->column_indices.begin(), temp_mat->column_indices.end(), temp_func);
    thrust::sequence(thrust::device, temp_mat->row_offsets.begin(), temp_mat->row_offsets.end());
    // for (int i = 0; i <= state_size; i++){
    //     if (i < state_size){
    //         temp_mat->row_offsets[i] = i;
    //     } else {
    //         temp_mat->row_offsets[i] = nnz;
    //     }
    // }
    // cout << "5.5" << endl;
    return temp_mat;
}

// build a dense array from a row in a sparse matrix
// void get_row(csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView>* s, int i, Array& result){
//     blas::fill(result, 0.0f);
//     if (s->row_offsets[i] - s->row_offsets[i+1] != 0){
//         int num_of_entry = s->row_offsets[i+1] - s->row_offsets[i];
//         int start_column_index = s->row_offsets[i];
//         for (i = 0; i < num_of_entry; i ++){
//             result[s->column_indices[start_column_index + i]] = s->values[start_column_index + i];
//         }
//     }
// }
// using scatter
void get_row(csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView>* s, int i, Array& result){
    blas::fill(result, 0.0f);
    if (s->row_offsets[i] - s->row_offsets[i+1] != 0){
        int num_of_entry = s->row_offsets[i+1] - s->row_offsets[i];
        int start_column_index = s->row_offsets[i];
        IndexArrayView column_indices = s -> column_indices.subarray(start_column_index, num_of_entry);
        ValueArrayView values = s -> values.subarray(start_column_index, num_of_entry);
        thrust::scatter(thrust::device, values.begin(), values.end(), column_indices.begin(), result.begin());
    }
}

int get_max_len(std::vector<std::vector<int> > sents){
    int max_len = 0;
    for(std::vector<int> sent : sents){
        if(sent.size() > max_len){
            max_len = sent.size();
        }
    }
    return max_len;
}

Dense* get_sentence_array(std::vector<std::vector<int> > sents, int max_len){
    Dense* array = new Dense( sents.size(), max_len, 0 );
    
    for(int i = 0; i < sents.size(); i++){
      std::vector<int> sent = sents[i];
      for(int token_ind = 0; token_ind < sent.size(); token_ind++){
        array -> operator()(i, token_ind) = sent[token_ind];
      }
    }
    return array; 
}

void HmmSampler::set_models(Model * models){
    // cout << '1' << endl;
    p_model = models;
    // cout << '2' << endl;
    if (p_indexer != NULL){
        delete p_indexer;
	p_indexer = NULL;
    }
    p_indexer = new Indexer(models);
    // cout << '3' << endl;
    int g_len = p_model-> g_max;
    cudaMemset(&STATE_SIZE,0,sizeof(int));
    cudaMemcpyToSymbol(STATE_SIZE, &g_len, sizeof(int), 0, cudaMemcpyHostToDevice);
    // cout << '4' << endl;
    //if (lexMatrix != NULL){
    //    delete lexMatrix;
    //    lexMatrix = NULL;
    //}
    lexMatrix = p_model -> lex; 
    // print(*(lexMatrix->get_view()));
    // exp_array(lexMatrix->get_view()->values); // exp the lex dist // the gpu models are not logged, should not need this
    // cout << '5' << endl;
    if (lexMultiplier != NULL){
        delete lexMultiplier;
        lexMultiplier = NULL;
    }
    lexMultiplier = tile(g_len, p_indexer->get_state_size());
    // print(*lexMultiplier);
    pi = p_model -> pi;
    // print( *(pi->get_view()) );
    if (trans_slice != NULL){
        delete trans_slice;
        trans_slice = NULL;
    }
    trans_slice = new Array(p_indexer->get_state_size(), 0.0f);
    expanded_lex = trans_slice;
    sum_dict = trans_slice;

    // cout.precision(float_limit::max_digits10);
}

void HmmSampler::initialize_dynprog(int batch_size, int max_len){
    
    max_sent_len = max_len;
    dyn_prog = new Dense*[max_len];
    for(int i = 0; i < max_len; i++){
        dyn_prog[i] = new Dense(p_indexer->get_state_size(), batch_size, 0.0f);
    }

    start_state = new Dense(p_indexer->get_state_size(), batch_size, 0.0f);
    for(int i = 0; i < batch_size; i++){
        start_state->operator()(0, i) = 1;
    }
}

std::vector<float> HmmSampler::forward_pass(std::vector<std::vector<int> > sents, int sent_index){
    // auto t1 = Clock::now();
    // cout << "Forward" << endl;
    float normalizer;
    int a_max, b_max, g_max; // index, token, g_len;
    std::tie(a_max, b_max, g_max) = p_indexer -> getVariableMaxes();
    std::vector<int> sent = sents[0];
    std::vector<float> log_probs;
    int batch_size = sents.size();
    int batch_max_len = get_max_len(sents);
    // np_sents is |batches| x max_len
    Dense* np_sents = get_sentence_array(sents, batch_max_len);
    csr_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView>* pi_view = pi -> get_view();
    
    array2d_view<ValueArrayView, row_major>* lex_view = lexMatrix -> get_view();
    
    // initialize likelihood vector:
    for(int sent_ind = 0; sent_ind < sents.size(); sent_ind++){
        log_probs.push_back(0);
    }
    
    
    for(int ind = 0; ind < batch_max_len; ind++){
        //cout << "Processing token index " << ind << " for " << batch_size << " sentences." << endl;
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
        //cout << "Performing transition multiplication" << endl;
        multiply(*pi_view, *prev_mat, *cur_mat);
        
        //cout << "Done with transition" << endl;
        //cout << "performing observation multiplications" << endl;

        auto trans_done = Clock::now();
        
        // for now incorporate the evidence sentence-by-sentence:
        for(int sent_ind = 0; sent_ind < sents.size(); sent_ind++){
            // not every sentence in the batch will need the full batch size
            if(sents[sent_ind].size() <= ind){
                continue;
            }
            //cout << "Processing sentence index " << sent_ind << endl;
            int token = sents[sent_ind][ind];
                
            // lex_column is |g| x 1 
            array2d<float, device_memory>::column_view lex_column = lex_view -> column(token);
                
            // print(lex_column);
            // cout << '6' << endl;
            // lexMultiplier is state_size x |g|, expanded_lex is state_size x 1
            // cout << "Multiplying lex multiplier by lex column" << endl;
            multiply(* lexMultiplier, lex_column, * expanded_lex);
            // print(expanded_lex);
            // cout << '7' << endl;
            // dyn_prog_row is 1 x state_size
            // dyn_prog_column is state_size x 1
            array2d<float, device_memory>::column_view dyn_prog_col = cur_mat->column(sent_ind);
            // cout << "Multiplying expanded_lex by dyn prog row" << endl;
            blas::xmy(*expanded_lex, dyn_prog_col, dyn_prog_col);
            // cout << "Computing normalizer" << endl;
            normalizer = thrust::reduce(thrust::device, dyn_prog_col.begin(), dyn_prog_col.end());
            // cout << "Normalizing over col with result: " << normalizer << endl;
            //cout << "Scaling by normalizer" << endl;
            blas::scal(dyn_prog_col, 1.0f/normalizer);
            //    cout << "Adding logged normalizer to sentence logprobs" << endl;
            log_probs[sent_ind] += log10f(normalizer);
        }
        
        auto norm_done = Clock::now();
    }
    
    //cout << "Finished forward pass (cuda) and returning vector with " << log_probs.size() << " elements." << endl;
    return log_probs;
}

std::vector<std::vector<State> > HmmSampler::reverse_sample(std::vector<std::vector<int>> sents, int sent_index){
    // cout << "Reverse sampling batch with starting sent index of " << sent_index << endl;
    // auto t2 = Clock::now();
    std::vector<std::vector<State>> sample_seqs;
    std::vector<State> sample_seq;
    std::vector<int> sample_t_seq;
    int last_index, sample_t, sample_depth; // , t, ind; totalK, depth,
    int batch_size = sents.size();
    int batch_max_len = get_max_len(sents);
    // int prev_depth, next_f_depth, next_awa_depth;
    //float sample_log_prob;//trans_prob, 
    // double t0, t1;
//    array1d<float, device_memory> trans_slice;
    State sample_state;
    //sample_log_prob = 0.0f;
    
//    for(std::vector<int> sent : sents){
    for(int sent_ind = 0; sent_ind < batch_size; sent_ind++){
        sample_seq = std::vector<State>();
        //cout << "Processing sentence " << sent_ind << " of the batch" << endl;
        std::vector<int> sent = sents[sent_ind];
        //for(int token_ind = 0; token_ind < sent.size(); token_ind++){
        //  cout << sent[token_ind] << " ";
       // }
        //cout << endl;
        
        last_index = sent.size() - 1;
        // doubly normalized??
        // self.dyn_prog[last_index,:] /= self.dyn_prog[last_index,:].sum()
        sample_t = -1;
        sample_depth = -1;
        //cout << "x1" << endl;
        while (sample_t < 0 || (sample_depth > 0)) {
            //cout << "Sampling last index: " << last_index << " for sentence" << endl;
            array2d<float, device_memory>::column_view dyn_prog_temp_col_view = dyn_prog[last_index]->column(sent_ind);
            //if(sample_t == -1) cusp::print(dyn_prog_temp_col_view);
            //cout << dyn_prog_temp_col_view << endl;
            sample_t = get_sample(dyn_prog_temp_col_view);
            // sample_t = 0;
            //cout << "sample_t=" << sample_t << endl;
            sample_state = p_indexer -> extractState(sample_t);
            //cout << sample_state.f << " " << sample_state.j << " " << sample_state.a[0] << " " << sample_state.a[1] << " " << sample_state.b[0] << " " << sample_state.b[1] << " " << sample_state.g << endl;
            if(!sample_state.depth_check()){
              cout << "Depth error in state assigned to last index" << endl;
            }
            sample_depth = sample_state.max_awa_depth();
            //cout << "Sample depth is "<< sample_depth << endl;
        }
        // auto t3 = Clock::now();
        //cout << "x3" << endl;
        sample_seq.push_back(sample_state);
        sample_t_seq.push_back(sample_t);
        // skip some error handling
    
        for (int t = sent.size() - 2; t > -1; t --){
            //cout << 't' << t << endl;
            // auto t11 = Clock::now();
            std::tie(sample_state, sample_t) = _reverse_sample_inner(sample_t, t, sent_ind);
            // cout << "Sample t is " << sample_t << endl;
            // cout << sample_state.f << " " << sample_state.j << " " << sample_state.a[0] << " " << sample_state.a[1] << " " << sample_state.b[0] << " " << sample_state.b[1] << " " << sample_state.g << endl;
            if(!sample_state.depth_check()){
              cout << "Depth error in state assigned at index" << t << endl;
            }
            sample_seq.push_back(sample_state);
            sample_t_seq.push_back(sample_t);
            // auto t12 = Clock::now();
            // cout << "backpass2inside: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t12 - t11).count() * nano_to_sec << " s" << endl;

        }
        // auto t4 = Clock::now();
        std::reverse(sample_seq.begin(), sample_seq.end());
        //cout << "x5" << endl;
        //cout << sample_seq->size() << endl;
        // auto t5 = Clock::now();
        // cout << "backpass1: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() * nano_to_sec << " s" << endl;
        // cout << "backpass2: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() * nano_to_sec << " s" << endl;
        // cout << "backpass: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t2).count() * nano_to_sec << " s" << endl;
        //for (int k : sample_t_seq){
        //    cout << sent_index << " : " << k  << endl;
        //}
        sample_seqs.push_back(sample_seq);
    }
    
    //cout << "Done with reverse()" << endl;
    return sample_seqs;
}


std::tuple<State, int> HmmSampler::_reverse_sample_inner(int& sample_t, int& t, int sent_ind){
    //int ind;
    float normalizer;
    // auto t11 = Clock::now();
    int prev_sample_t = sample_t;
    get_row(pi->get_view(), sample_t, *trans_slice);
    // cout << "trans_slice" << endl;
    // print(*trans_slice); 
    // auto t12 = Clock::now();
    array2d<float, device_memory>::column_view dyn_prog_col = dyn_prog[t]->column(sent_ind);
    // cout << "Dyn_prog_row" << endl;
    // print(dyn_prog_row);
    float trans_slice_sum = thrust::reduce(thrust::device, (*trans_slice).begin(), (*trans_slice).end());
    if (trans_slice_sum != 0.0f){
        cusp::blas::xmy(*trans_slice, dyn_prog_col, dyn_prog_col);
    }
    // auto t13 = Clock::now();
    cusp::array1d<float, host_memory> un_normalized_sums(dyn_prog_col);
    normalizer = thrust::reduce(thrust::device, dyn_prog_col.begin(), dyn_prog_col.end());
    // cout << "normalizer" << normalizer <<endl;
    blas::scal(dyn_prog_col, 1.0f/normalizer);
    // thrust::transform(dyn_prog_row.begin(), dyn_prog_row.end(), dyn_prog_row.begin(), multiplies_value<float>(1 / normalizer));
    // auto t14 = Clock::now();
    sample_t = get_sample(dyn_prog_col);
    //if (sample_t == 0){
    //    print(dyn_prog_row);
    //}
    // auto t15 = Clock::now();
    State sample_state = p_indexer -> extractState(sample_t);
    // if ((sample_state.a[1] == 0 && sample_state.b[1] != 0)|| (sample_state.a[1] != 0 && sample_state.b[1] == 0) ){
    get_row(pi->get_view(), prev_sample_t, *trans_slice);
        // cout << "Dyn Prog(normalized): " << scientific << dyn_prog_row[sample_t -1] << " " << scientific << dyn_prog_row[sample_t] <<" "<< scientific <<dyn_prog_row[sample_t + 1] <<endl;
        // cout << "Trans Slice (Pi Row): " <<" " << scientific << (*trans_slice)[sample_t-1]<< " "<< scientific << (*trans_slice)[sample_t] <<" " << scientific << (*trans_slice)[sample_t+1]<< endl;
        // cout << "Prefix Sum(unnormalized, before sum): " << scientific << un_normalized_sums[sample_t-1] << " " << scientific << un_normalized_sums[sample_t] << " " << scientific << un_normalized_sums[sample_t + 1] << endl;
        // cout << "Prefix Sum(unnormalozed, before sum): ";
        // float temp_1 = un_normalized_sums[sample_t-1];
        // printf( "%A", temp_1);
        // cout << " ";
        // float temp_2 = un_normalized_sums[sample_t];
        // printf("%A", temp_2);
        // cout << " ";
        // float temp_3 = un_normalized_sums[sample_t+1];
        // printf("%A\n", temp_3);
        // cout << "sample_t: " << sample_t << endl;
        
    // }
    // cout << "backpass1reverseinner: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t12 - t11).count() * nano_to_sec << " s" << endl;
    // cout << "backpass1reverseinner: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t13 - t12).count() * nano_to_sec << " s" << endl;
    // cout << "backpass1reverseinner: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t14 - t13).count() * nano_to_sec << " s" << endl;
    // cout << "backpass1reverseinner: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t15 - t14).count() * nano_to_sec << " s" << endl;

    return std::make_tuple(sample_state, sample_t);
}

std::tuple<std::vector<std::vector<State> >, std::vector<float>> HmmSampler::sample(std::vector<std::vector<int>> sents, int sent_index) {
    std::vector<float> log_probs;
    std::vector<std::vector<State> > states;

    try{
        log_probs = forward_pass(sents, sent_index);
    }catch(thrust::system_error &e){
        cerr << "Error in forward pass: " << e.what() << endl;
        exit(-1);
    }
    try{
        states = reverse_sample(sents, sent_index);
    }catch(thrust::system_error &e){
        cerr << "Error in reverse sample: " << e.what() << endl;
        exit(-1);
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
    //delete[] dyn_prog;
    //delete p_model;
    delete p_indexer;
    //delete lexMatrix;
    //delete dyn_prog;
    delete lexMultiplier;
    //delete pi;
    delete trans_slice;
    //delete expanded_lex;
    //delete sum_dict;
}
