// include the csr_matrix header file
// nvcc gpusrc/multiply_test.cu -o test_multiply
#include <stdio.h>
#include <iostream>

#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include "cusp/multiply.h"
#include "myarrays.cu"

using namespace std;

int main()
{
  int state_size = 15;
  int batch_size = 1;

  // allocate storage for (10,10) matrix with 5 nonzeros
  cusp::csr_matrix<int,float,cusp::host_memory> A(state_size, state_size, 5);
  // initialize matrix entries on host
  A.row_offsets[0] = 0;  // first offset is always zero
  A.row_offsets[1] = 0;
  A.row_offsets[2] = 1;
  A.row_offsets[3] = 2;
  A.row_offsets[4] = 3;
  A.row_offsets[5] = 4;
  A.row_offsets[6] = 5; A.row_offsets[7] = 5; A.row_offsets[8] = 5; A.row_offsets[9] = 5; A.row_offsets[10] = 5; A.row_offsets[11] = 5;
  A.row_offsets[12] = 5; A.row_offsets[13] = 5; A.row_offsets[14] = 5; A.row_offsets[15] = 5;
  
  A.column_indices[0] = 0; A.values[0] = 0.19;
  A.column_indices[1] = 0; A.values[1] = 0.2;
  A.column_indices[2] = 0; A.values[2] = 0.21;
  A.column_indices[3] = 0; A.values[3] = 0.22;
  A.column_indices[4] = 0; A.values[4] = 0.18;
  //A.column_indices[5] = 2; A.values[5] = 60;
  // A now represents the following matrix
  //    [10  0 20]
  //    [ 0  0  0]
  //    [ 0  0 30]
  //    [40 50 60]
  // copy to the device
  cout << "Host version of pi (A): " << endl;
  cusp::print(A);
  
  cusp::array2d<float, cusp::host_memory> _A(A);
  
  cout << "Dense version of ip (_A): " << endl;
  cusp::print(_A);
  
  cout << "Device version of pi (B): " << endl;
  cusp::csr_matrix<int,float,cusp::device_memory> B(A);
  cusp::print(B);
  
  Dense *prev_mat = new Dense(state_size, batch_size, 0.0f);
  Dense *next_mat = new Dense(state_size, batch_size, 0.0f);
  for(int i = 0; i < batch_size; i++){
      prev_mat->operator()(0, i) = 1;
  }
  
  multiply(B, *prev_mat, *next_mat);
  
  cout << "Previous matrix:" << endl;
  cusp::print(*prev_mat);
  cout << "Next matrix:" << endl;
  cusp::print(*next_mat);
  
  for(int i = 0; i < batch_size; i++){
      for(int j = 0; j < state_size; j++){
         cout << "Value of next_mat[ " << j << ", " << i << "] = " << next_mat->operator()(j,i) << endl;
      }
  }
  
  // initialize matrix
  cusp::array2d<float, cusp::host_memory> C(2,2);
  C(0,0) = 10;  C(0,1) = 20;
  C(1,0) = 40;  C(1,1) = 50;
  // initialize input vector
  cusp::array1d<float, cusp::host_memory> x(2);
  x[0] = 1;
  x[1] = 2;
  // allocate output vector
  cusp::array1d<float, cusp::host_memory> y(2);
  // compute y = C * x
  cusp::multiply(C, x, y);
  // print y
  cusp::print(y);
  
  
  cusp::array1d<float, cusp::device_memory> prev_array(15, 0.0f);
  cusp::array1d<float, cusp::device_memory> next_array(15, 0.0f);
  prev_array[0] = 1;
  multiply(B, prev_array, next_array);
  cusp::print(next_array);

  cusp::array1d<float, cusp::device_memory> *prev_array_p = &prev_array;
  cusp::array1d<float, cusp::device_memory> *next_array_p = &next_array;

  multiply(B, *prev_array_p, *next_array_p);
  cusp::print(*next_array_p);

}
