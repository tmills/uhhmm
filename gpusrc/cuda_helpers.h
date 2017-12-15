#ifndef UHHMM_CUDA_FNS
#define UHHMM_CUDA_FNS

__device__ float RECIP_SQRT_2_PI = 0.39894;
__device__ int G_SIZE;

/* works correctly if argument is in +/-[2**-15, 2**17), or zero, infinity, NaN */
// From: https://devtalk.nvidia.com/default/topic/865401/fast-float-to-double-conversion/
__device__ __forceinline__ 
double my_fast_float2double (float a)
{
    unsigned int ia = __float_as_int (a);
    return __hiloint2double ((((ia >> 3) ^ ia) & 0x07ffffff) ^ ia, ia << 29);
}

class normal_logpdf_firstfactor : public thrust::unary_function<float, double> {
public:
    __device__
    double operator()(const float & stdev) const {
        return my_fast_float2double(RECIP_SQRT_2_PI / stdev);
    }
};

class normal_logpdf_squarederror : public thrust::binary_function<float,float,float> {
public:
    __device__
    float operator()(const float & x, const float & mu) const {
        return pow(x-mu, 2.0);
    }
};

class normal_logpdf_twostdevsquared : public thrust::unary_function<float,float> {
public:
    __device__
    float operator()(const float &stdev) const {
        return 2 * pow(stdev, 2.0);
    }
};

class normal_logpdf_secondfactor : public thrust::binary_function<float,float,double> {
public:
    __device__
    double operator()(const float& numerator, const float& denominator){
        return exp(my_fast_float2double(-numerator) / my_fast_float2double(denominator));
    }
};

class normal_logpdf_log : public thrust::unary_function<double,float> {
public:
    __device__
    float operator()(const double& x){
        return __double2float_rd(log(x));
    }
};

class cuda_exp : public thrust::unary_function<float, float>{
public:
    __device__
   float operator()(float x){
       return exp(x);
   }
};

// TODO: Think I can make G_SIZE an argument to a constructor instead of a constant
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

void debug_print_float_vector(thrust::device_vector<float> vec){
    thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
}

void debug_print_double_vector(thrust::device_vector<double> vec){
    thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
}
#endif
