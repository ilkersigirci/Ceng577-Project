# Ceng577-Project
Implementation of Asynchronous SGD with MPI

## Install MPI
`sudo apt install mpich`

### Verify Installation
`mpiexec --version`

### Verify Installation
`cat /usr/include/boost/version.hpp | grep "BOOST_LIB_VERSION"`

### Compile
`make`

### Run with 3 processors (one process is parameter server, other 2 processes are the workers)
`make run PROC=3`

# Naive SGD
To run synchronous SGD you need to rename `test_naive.cpp.naive` to `main.cpp` and remove other `main.cpp`.

Then `make run PROC=1`.