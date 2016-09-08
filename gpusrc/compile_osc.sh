#!/bin/bash
# a 'include' folder will be created above uhhmm
module load cuda/7.0.28
module load python/3.4.2
module load gnu/4.9.1
cd ../
mkdir include
git clone git@github.com:cusplibrary/cusplibrary.git
mv cusplibrary/cusp include/
rm -rf cusplibrary
cd - 
nvcc -rdc=true -c -o gpusrc/temp.o gpusrc/HmmSampler.cu -std=c++11 --shared -Xcompiler -fPIC -m64 -I../include
nvcc -dlink -o gpusrc/hmmsampler.o gpusrc/temp.o -lcudart --shared -Xcompiler -fPIC -m64 -L/usr/local/cuda/7.0.28/lib64 -Xlinker -rpath -Xlinker /usr/local/cuda/7.0.28/lib64
ar cru gpusrc/libhmm.a gpusrc/hmmsampler.o gpusrc/temp.o
ranlib gpusrc/libhmm.a
cython --cplus gpusrc/CHmmSampler.pyx
# clang -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -std=c++11 -I/Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/site-packages/numpy/core/include -I/Library/Frameworks/Python.framework/Versions/3.4/include/python3.4m -Isrc -I/usr/local/include/  -c CHmmSampler.cpp -o build/CHmmSampler.o -std=c++11 -L. -lhmm -L/usr/local/cuda/lib -lcudart
#x86_64-linux-gnu-g++ -pthread -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.4/dist-packages/numpy/core/include -I/usr/include/python3.4m -I/usr/local/include/ -c CHmmSampler.cpp -o build/CHmmSampler.o -w -std=c++11 -L. -lhmm -L/usr/local/cuda/lib -lcudart
g++ -pthread -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/anaconda/anaconda3/lib/python3.4/site-packages/numpy/core/include -I/usr/local/anaconda/anaconda3/include/python3.4m -I/usr/local/include/ -c gpusrc/CHmmSampler.cpp -o gpusrc/CHmmSampler.o -w -std=c++11  -L/usr/local/cuda/7.0.28/lib64 -lcudart -L/usr/local/anaconda/anaconda3/lib -lpython3.4m -Lgpusrc/ -lhmm
# clang -bundle -undefined dynamic_lookup -arch x86_64 -g -std=c++11 build/CHmmSampler.o libhmm.a -o build/CHmmSampler.so -L. -lhmm -L/usr/local/cuda/lib -lcudart
g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -std=c++11 gpusrc/CHmmSampler.o gpusrc/libhmm.a -o scripts/CHmmSampler.so -Lgpusrc/ -lhmm -L/usr/local/cuda/7.0.28/lib64 -lcudart
rm gpusrc/temp.o
rm gpusrc/hmmsampler.o
rm gpusrc/libhmm.a
rm gpusrc/CHmmSampler.o
rm gpusrc/CHmmSampler.cpp
# export LD_LIBRARY_PATH=/usr/local/cuda/6.5.14/lib64:$LD_LIBRARY_PATH
# python3 test_suite.py