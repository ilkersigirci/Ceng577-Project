# Ceng577-Project
Implementation of Asynchronous SGD with MPI

## Install MPI
sudo apt install mpich

### Verify Installation
mpiexec --version

## Install Boost (For Serialization)
sudo apt install libboost-dev
sudo apt install libboost-all-dev

### Verify Installation
cat /usr/include/boost/version.hpp | grep "BOOST_LIB_VERSION"
### Compile
make

### Run with 2 processors
mpirun -n 2 ./a.out