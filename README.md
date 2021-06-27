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

# Visualizer
To visualize comparison of loss curves you can run following command:

`python visualize_loss_curve.py --input (LOSS_FILE)`

It will visualize comparison of given losses with naive SGD results. Naive SGD results are in `losses.txt` and 
automatically used by visualizer. You don't need to specify it.