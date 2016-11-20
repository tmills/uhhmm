// include the csr_matrix header file
// nvcc gpusrc/multiply_test.cu -o test_multiply
#include <stdio.h>
#include <iostream>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/functional.h>
#include "cusp/multiply.h"
#include "myarrays.cu"
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/gallery/poisson.h>

using namespace std;

int main()
{
  int state_size = 15;
  //int batch_size = 5;
  
  cusp::array2d<float, cusp::device_memory> pi(state_size, state_size);
  pi(1,0) = 0.19; pi(2,0) = 0.2; pi(3,0) = 0.21; pi(4,0) = 0.22; pi(5,0) = 0.18;
  pi(1,1) = 0.19; pi(2,2) = 0.2; pi(3,3) = 0.21; pi(4,4) = 0.22; pi(5,5) = 0.18;
  pi(1,6) = 0.19; pi(2,7) = 0.2; pi(3,8) = 0.21; pi(4,9) = 0.22; pi(5,10) = 0.18;
  pi(5,1) = 0.29; pi(6,2) = 0.3; pi(7,3) = 0.31; pi(8,4) = 0.32; pi(9,5) = 0.28;
  pi(5,6) = 0.29; pi(6,7) = 0.3; pi(7,8) = 0.31; pi(8,9) = 0.32; pi(9,10) = 0.28;
  
  cout << "Dense version of pi:" << endl;
  cusp::print(pi);
  cusp::csr_matrix<int,float,cusp::device_memory> pi_sparse(pi);
    
  for(int batch_size = 1; batch_size <= 5; batch_size++){
      Dense *prev_mat = new Dense(state_size, batch_size, 0.0f);
      Dense *next_mat = new Dense(state_size, batch_size, 0.0f);
      for(int i = 0; i < batch_size; i++){
          prev_mat->operator()(0, i) = 1;
      }
  
      cusp::multiply(pi_sparse, *prev_mat, *next_mat);
  
      cout << "Previous matrix:" << endl;
    cusp::print(*prev_mat);
      cout << "Next matrix non-zeros:" << endl;
  //cusp::print(*next_mat);
  
      for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < state_size; j++){
             if(next_mat->operator()(j,i) > 0){
                 cout << "Value of next_mat[ " << j << ", " << i << "] = " << next_mat->operator()(j,i) << endl;
             }
          }
      }
  }
}

