#include <iostream>
#include <string>

#include <mpi.h>
#include <stdio.h>
#include <unistd.h> // sleep
#include <stdlib.h> // rand

/*
* MPI_Send -> executes without waiting the receive
            works like MPI_Bsend(?) or MPI_Rsend not MPI_Ssend
* int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
* int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)

* // Wait for the MPI_Isend to complete before progressing further.
* MPI_Wait(&reqs[it], MPI_STATUS_IGNORE);
* request.Wait(status); // C++

* MPI_Test(&reqs[it], &flag, &status);
* flag = request.Test( status ); // C++
*/

/*
* Note the use of the MPI constant MPI_ANY_SOURCE to allow this MPI_Recv call to receive messages from any process. 
* In some cases, a program would need to determine exactly which process sent a message received using MPI_ANY_SOURCE.
* status.MPI_SOURCE will hold that information, immediately following the call to MPI_Recv. 
*/

int main(int argc, char* argv[])
{
    int rank, process_size;
    //int shared_data[4] = {10, 11, 12, 13};
    std::array<int, 4> shared_data{1, 2, 3};
    int model_parameter, gradient_update;
    int iteration_num = 100;
    //int batch_size = 1;
    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    // MPI_Status status;

    const double EPSILON 0.000001;
    const double LEARNING_RATE = 0.001;
    const int PARAMETER_TAG 100;
    const int GRADIENT_TAG  101;
    const int PARAMETER_SERVER 0;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    MPI_Request reqs[iteration_num];
    MPI_Status stats[iteration_num];
	
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);
    //process_size = MPI::COMM_WORLD.Get_size();
	
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //rank = MPI::COMM_WORLD.Get_rank();

    // Get the name of the processor
    MPI_Get_processor_name(processor_name, &name_len);
	
	// Print all the configs
    printf("Processor %s, rank %d out of %d processors\n", processor_name, rank, process_size);

    if (process_size < 2)
    {
        std::cout << "Please run it with at least 2 process." << std::endl;
        std::cout << "Example: mpirun -n 3 ./a_sgd" << std::endl;
        
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) // parameter server
    {
        model_parameter = -1;
        
        int model_parameter_bak = -1;

        for (int process=1; process < process_size; process++) // Broadcast initial model_parameters
            MPI_Send(&model_parameter, 1, MPI_INT, process, PARAMETER_TAG, MPI_COMM_WORLD);
        
        for (int it = 0; it < iteration_num; it++)
        // while (true)
        {
            MPI_Recv(&model_parameter, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stats[it]); // &status

            int recv_source = stats[it].MPI_SOURCE;
            int recv_tag    = stats[it].MPI_TAG;

            if (recv_tag == GRADIENT_TAG)
            {
                w_t1 = w_t - LEARNING_RATE * recv_source;
                t = t + 1;
            }

            else if (recv_tag == PARAMETER_TAG) // pull request
            {
                model_parameter_bak = model_parameter;

                MPI_Isend(&model_parameter, 1, MPI_INT, recv_source, PARAMETER_TAG, MPI_COMM_WORLD, &reqs[it]);
                //printf("Sent model_parameter %d from 0 to 1\n", model_parameter);
            }
        }
    }

    else // workers
    {
        for (int it = 0; it < iteration_num; it++)
        // while (true)
        {
            // sleep(1);
            int random_index = rand() % shared_data.size();
            int minibatch = shared_data[random_index];

            int count;

            MPI_Recv(&model_parameter, 1, MPI_INT, PARAMETER_SERVER, PARAMETER_TAG, MPI_COMM_WORLD, &stats[it]); // &status

            MPI_Get_count(&stats[it] , MPI_INT, &count);
            printf("From source %d, received %d element with tag %d and value %d\n", stats[it].MPI_SOURCE, count, stats[it].MPI_TAG, model_parameter);

            // Gradient Computation
            gradient_update = minibatch * model_parameter;

            MPI_Isend(&gradient_update, 1, MPI_INT, PARAMETER_SERVER, GRADIENT_TAG, MPI_COMM_WORLD, &reqs[it]);
        }
    }

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}