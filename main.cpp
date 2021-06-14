#include <iostream>
#include <string>

#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
    int rank, process_size;
    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
	
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);
	
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the name of the processor
    MPI_Get_processor_name(processor_name, &name_len);
	
	// Print all the configs
    printf("Processor %s, rank %d out of %d processors\n", processor_name, rank, process_size);

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}