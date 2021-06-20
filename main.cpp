#include <iostream>
#include <string>

#include <mpi.h>
#include <stdio.h>
#include <unistd.h> // sleep
#include <stdlib.h> // rand

#include <fstream>
#include <iterator>

#include "Eigen/Core"
#include "activations/Sigmoid.h"
#include "activations/Identity.h"
#include "layer.h"
#include "Network.h"
#include "losses/RMSE.h"
#include "SGD.h"
#include "utils.h"

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

/*
* MPI_Send -> executes without waiting the receive
*             works like MPI_Bsend(?) or MPI_Rsend not MPI_Ssend
* int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
* int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)

* // Wait for the MPI_Isend to complete before progressing further.
* MPI_Wait(&reqs[t], MPI_STATUS_IGNORE);
* request.Wait(status); // C++

* MPI_Test(&reqs[t], &flag, &status);
* flag = request.Test( status ); // C++
*/

void check_error(int ierr, std::string operation)
{
    if (ierr)
    {
        //printf("%s returned ierr %d\n", operation, ierr);
        std::cout << operation << " returned ierr " << ierr << std::endl;
        exit(1);

        // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

int main(int argc, char* argv[])
{
    int rank, process_size, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    // int shared_data[4] = {10, 11, 12, 13};
    std::array<int, 4> shared_data{1, 2, 3};
    double model_parameter, gradient_update;
    int iteration_num = 100;
    // int batch_size = 1;

    double exchanged_data; // either gradient or model_parameter
    
    double t1, t2; 

    // MPI_Status status;

    const int PARAMETER_TAG = 100;
    const int GRADIENT_TAG  = 101;
    const int PARAMETER_SERVER =  0;

    const double EPSILON = 0.000001;
    const double LEARNING_RATE = 0.001;
    
    int ierr = MPI_Init(&argc, &argv);

    check_error(ierr, "MPI_Init");

    t1 = MPI_Wtime();

    MPI_Request reqs[iteration_num];
    MPI_Status stats[iteration_num];
	
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);
	
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Get_processor_name(processor_name, &name_len);

    if (process_size < 2)
    {
        std::cout << "Please run it with at least 2 process." << std::endl;
        std::cout << "Example: mpirun -n 3 ./a_sgd" << std::endl;
        
        // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        MPI_Finalize();
        exit(1);
    }

    if (rank == 0) // parameter server
    {
        model_parameter = -1;
        
        int model_parameter_bak = model_parameter;

        // for (int process=1; process < process_size; process++) // Broadcast initial model_parameters
        //     MPI_Send(&model_parameter, 1, MPI_DOUBLE, process, PARAMETER_TAG, MPI_COMM_WORLD);
        
        for (int t = 0; t < iteration_num;)
        // while (true)
        {
            std::cout << "Server t " << t << std::endl;

            //TODO: Need Lock for multiple receive?
            MPI_Recv(&exchanged_data, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stats[t]);

            int recv_source = stats[t].MPI_SOURCE;
            int recv_tag    = stats[t].MPI_TAG;

            if (recv_tag == GRADIENT_TAG)
            {
                std::cout << "Gradient Update" << std::endl;
                gradient_update = exchanged_data;
                model_parameter = model_parameter - LEARNING_RATE * gradient_update;
                t++;
            }

            else if (recv_tag == PARAMETER_TAG) // pull request
            {
                std::cout << "Pull Request" << std::endl;
                model_parameter_bak = model_parameter;

                MPI_Isend(&model_parameter, 1, MPI_DOUBLE, recv_source, PARAMETER_TAG, MPI_COMM_WORLD, &reqs[t]);
                //printf("Sent model_parameter %d from 0 to 1\n", model_parameter);
            }
        }
    }

    //TODO: deal with end condition
    else // workers
    {
        for (int t = 0; t < iteration_num; t++)
        // while (true)
        {
            // sleep(1);
            std::cout << "Worker " << rank << ": t " << t << std::endl;
            int random_index = rand() % shared_data.size();
            int minibatch = shared_data[random_index];

            int count;

            MPI_Send(&model_parameter, 1, MPI_DOUBLE, PARAMETER_SERVER, PARAMETER_TAG, MPI_COMM_WORLD);
            MPI_Recv(&model_parameter, 1, MPI_DOUBLE, PARAMETER_SERVER, PARAMETER_TAG, MPI_COMM_WORLD, &stats[t]);

            MPI_Get_count(&stats[t] , MPI_DOUBLE, &count);
            printf("Worker %d received %d element with tag %d and value %f from source %d\n", rank, count, stats[t].MPI_TAG, model_parameter, stats[t].MPI_SOURCE);

            // Gradient Computation
            gradient_update = minibatch * model_parameter;

            MPI_Isend(&gradient_update, 1, MPI_DOUBLE, PARAMETER_SERVER, GRADIENT_TAG, MPI_COMM_WORLD, &reqs[t]);
            // MPI_Send(&gradient_update, 1, MPI_DOUBLE, PARAMETER_SERVER, GRADIENT_TAG, MPI_COMM_WORLD);
        }
    }

    t2 = MPI_Wtime();
    printf("Processor %s, rank %d running Time %f\n", processor_name, rank, t2 - t1);

    MPI_Finalize();

    return 0;
}