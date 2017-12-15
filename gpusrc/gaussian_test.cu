// include the csr_matrix header file
// nvcc gpusrc/multiply_test.cu -o test_multiply
#include <stdio.h>
#include <iostream>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/gallery/poisson.h>
#include "cuda_helpers.h"

using namespace cusp;
using namespace std;

int main()
{
    int embed_dims = 4;

    cusp::array1d<float, cusp::device_memory> means(embed_dims);
    means[0] = 0.0; means[1] = 1.0; means[2] = 2.0; means[3] = 3.0;

    cusp::array1d<float, cusp::device_memory> stdevs(embed_dims);
    stdevs[0] = 1.0; stdevs[1] = 2.0; stdevs[2] = 3.0; stdevs[3] = 2.0;
    
    thrust::device_vector<double> goldd(embed_dims);
    
    thrust::device_vector<float> goldf(embed_dims);

    std::cout << "Means:" << std::endl;
    cusp::print(means);
    std::cout << "Stdevs:" << std::endl;
    cusp::print(stdevs);

    thrust::device_vector<float> embed_vec(embed_dims);
    thrust::fill(embed_vec.begin(), embed_vec.end(), 0.0);
    thrust::device_vector<double> normalizer(embed_dims);
    thrust::device_vector<float> errors(embed_dims);
    thrust::device_vector<float> stdev_squared(embed_dims);
    thrust::device_vector<double> second_factor(embed_dims);
    thrust::device_vector<double> final_prob(embed_dims);
    thrust::device_vector<float> log_prob(embed_dims);
    thrust::device_vector<double> out_double(embed_dims);
    thrust::device_vector<float> out_float(embed_dims);
    
    // thrust::device_vector<float> temp(numWords);


    thrust::transform(stdevs.begin(), stdevs.end(), normalizer.begin(), normal_logpdf_firstfactor());
    // calculate the numerator of the exponentiated factor (binary function):
    // cout << "  Calculating squared error term..." << endl;
    thrust::transform(embed_vec.begin(), embed_vec.end(), means.begin(), errors.begin(), normal_logpdf_squarederror());
    // calculate the denominator of the exponentiated factor (unary):
    // cout << "  Calculating two stdevs squared..." << endl;
    thrust::transform(stdevs.begin(), stdevs.end(), stdev_squared.begin(), normal_logpdf_twostdevsquared());
    // calculate the exponentiated term (binary)
    // cout << "  Calculating second factor ratio" << endl;
    thrust::transform(errors.begin(), errors.end(), stdev_squared.begin(), second_factor.begin(), normal_logpdf_secondfactor());
    // finalize output with simple multiplication:
    // cout << "  Calculating final probability vector" << endl;
    thrust::transform(normalizer.begin(), normalizer.end(), second_factor.begin(), final_prob.begin(), thrust::multiplies<double>());
    // take the log probabiility:
    thrust::transform(final_prob.begin(), final_prob.end(), log_prob.begin(), normal_logpdf_log());

    // Testing the probabilities:
    cout << "Probability output:" << endl;
    debug_print_double_vector(final_prob);
    // computed in octave:
    goldd[0] = 0.398942; goldd[1] = 0.176033; goldd[2] = 0.106483; goldd[3] = 0.064759;
    thrust::transform(final_prob.begin(), final_prob.end(), goldd.begin(), out_double.begin(), thrust::minus<double>());
    cout << "Difference from expected: " << endl;
    debug_print_double_vector(out_double);

    // testing the natural logarithm:
    cout << "Log prob output:" << endl;
    debug_print_float_vector(log_prob);
    goldf[0] = -0.91894; goldf[1] = -1.73709; goldf[2] = -2.23977; goldf[3] = -2.73709;
    thrust::transform(log_prob.begin(), log_prob.end(), goldf.begin(), out_float.begin(), thrust::minus<float>());
    cout << "Difference from expected: " << endl;
    debug_print_float_vector(out_float);
    
    // test log sum:
    float sum = thrust::reduce(log_prob.begin(), log_prob.end());
    float gold_sum = -7.6329; // from octave
    cout << "Log sum: " << sum << endl;
    cout << "Difference from expected: " << (sum - gold_sum) << endl;

    // test log normalization:
    float max_prob = *(thrust::max_element(log_prob.begin(), log_prob.end()));
    float gold_max = -0.91894; // from octave
    cout << "Max log prob: " << max_prob << endl;
    cout << "Expected max log prob: " << gold_max << endl;
    thrust::fill(out_float.begin(), out_float.end(), max_prob);
    thrust::transform(log_prob.begin(), log_prob.end(), out_float.begin(), log_prob.begin(), thrust::minus<float>());
    cout << "Log probs minus max: " << endl;
    debug_print_float_vector(log_prob);

    // exponentiation:
    thrust::transform(log_prob.begin(), log_prob.end(), out_float.begin(), cuda_exp());
    goldf[0] = 1.0; goldf[1] = 0.44125; goldf[2] = 0.26691; goldf[3] = 0.16233; // from octave
    cout << "Exponentiated probs: " << endl;
    debug_print_float_vector(out_float);
    cout << "Expected exponentiated probs: " << endl;
    debug_print_float_vector(goldf);

    // normalization:
    float norm = thrust::reduce(out_float.begin(), out_float.end());
    float gold_norm = 1.8705;
    thrust::device_vector<float> tempf(embed_dims);
    thrust::fill(tempf.begin(), tempf.end(), norm);
    thrust::transform(out_float.begin(), out_float.end(), tempf.begin(), out_float.begin(), thrust::divides<float>());
    cout << "Normalized probs: " << endl;
    debug_print_float_vector(out_float);
    goldf[0] = 0.534620; goldf[1] = 0.23590; goldf[2] = 0.142697; goldf[3] = 0.086783;
    cout << "Expected normalized probs: " << endl;
    debug_print_float_vector(goldf);
}
