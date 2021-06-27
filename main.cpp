#include <iostream>
#include <string>

#include <mpi.h>
#include <stdio.h>
#include <unistd.h> // sleep
#include <stdlib.h> // rand

#include <fstream>
#include <iterator>
#include<ctime>
#include<string>

#include "Eigen/Core"
#include "activations/Sigmoid.h"
#include "activations/Identity.h"
#include "Layer.h"
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
    }
}

int main(int argc, char* argv[])
{
    int rank, process_size, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    // For calculating start and finish time
    double t1, t2;

    const int PARAMETER_TAG = 100;
    const int GRADIENT_TAG  = 101;
    const int EXIT_TAG      = 666;
    const int PARAMETER_SERVER =  0;

    // Create Data
    Vector x1 = Vector::LinSpaced(1000, 0.0, 2.5);
    Vector x2 = Vector::LinSpaced(1000, 3, 6);
    Matrix x = Matrix::Random(2, 1000);
    x.row(0) = x1;
    x.row(1) = x2;
    Matrix y = Matrix::Random(1, 1000);

    // Fill the output for the training
    for (int i = 0; i < y.cols(); i++)
    {
        y(0, i) = std::pow(x(0, i), 2) + std::pow(x(1, i), 2);
    }

    // Init Network
    Network net;
    BaseLayer* layer1 = new FullyConnectedLayer<Identity>(2, 200);
    BaseLayer* layer2 = new FullyConnectedLayer<Sigmoid>(200, 200);
    BaseLayer* layer3 = new FullyConnectedLayer<Identity>(200, 1);

    // Add layers to the network object
    net.layers.push_back(layer1);
    net.layers.push_back(layer2);
    net.layers.push_back(layer3);

    // Init loss function
    net.loss = new RMSE();

    const int NETWORK_PARAM_SIZE = net.get_parameters().size();

    // Training Hyper-parameters
    const int num_iters = 2000;
    const int batch_size = 16;

    // network warm up
    Matrix x_batch = Matrix::Random(2, batch_size);
    Matrix y_batch = Matrix::Random(1, batch_size);
    fetch_batches(x, y, batch_size, x_batch, y_batch);
    net.batch_fit(x_batch, y_batch);

    // Optimizer
    SGD sgd;

    std::vector<double> all_losses;

    int ierr = MPI_Init(&argc, &argv);

    check_error(ierr, "MPI_Init");

    t1 = MPI_Wtime();

    MPI_Request req;
    MPI_Status stat;

    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Get_processor_name(processor_name, &name_len);

    if (process_size < 2)
    {
        std::cout << "Please run it with at least 2 process." << std::endl;
        std::cout << "Example: mpirun -n 3 ./a_sgd" << std::endl;

        MPI_Finalize();
        exit(1);
    }

    if (rank == 0) // parameter server
    {
        int LOG_PERIOD = 50;
        int live_process_cnt = process_size - 1;
        std::vector<double> received_grads;
        received_grads.resize(NETWORK_PARAM_SIZE);

        for (int t = 0; t < num_iters + 1 && live_process_cnt;) {
            MPI_Recv(&received_grads[0], NETWORK_PARAM_SIZE, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                     &stat);

            int recv_source = stat.MPI_SOURCE;
            int recv_tag = stat.MPI_TAG;

            if (recv_tag == GRADIENT_TAG) {
                net.set_grads(received_grads);
                net.step(sgd);
                Matrix x_batch = Matrix::Random(2, batch_size);
                Matrix y_batch = Matrix::Random(1, batch_size);
                fetch_batches(x, y, batch_size, x_batch, y_batch);
                net.batch_fit(x_batch, y_batch);

                double loss = net.loss->loss();
                all_losses.push_back(loss);
                if (t % LOG_PERIOD == 0) {
                    printf("Iteration %d, Server get Gradient Update from Worker %d and the loss is %f\n", t,
                           recv_source,
                           loss);
                }

                t++;
            } else if (recv_tag == PARAMETER_TAG) // pull request
            {
                if (t >= num_iters - live_process_cnt + 1) {
                    int temp_data = 1;
                    MPI_Send(&temp_data, 1, MPI_INT, recv_source, EXIT_TAG, MPI_COMM_WORLD);
                    live_process_cnt -= 1;
                }
                else {
                    std::vector<double> network_params = net.get_parameters();
                    MPI_Ssend(&network_params[0], NETWORK_PARAM_SIZE, MPI_DOUBLE, recv_source, PARAMETER_TAG,
                              MPI_COMM_WORLD);
                }
            }
        }
        std::time_t tt = std::time(0);
        std::ofstream output_file("./losses_" + std::to_string(tt) + ".txt");
        std::ostream_iterator<double> output_iterator(output_file, "\n");
        std::copy(all_losses.begin(), all_losses.end(), output_iterator);
    }

    else // workers
    {
        int t = 0;
        std::vector<double> network_parameters;
        network_parameters.resize(NETWORK_PARAM_SIZE);

        while (true)
        {
            int count;

            MPI_Isend(&t, 1, MPI_DOUBLE, PARAMETER_SERVER, PARAMETER_TAG, MPI_COMM_WORLD, &req);
            MPI_Recv(&network_parameters[0], NETWORK_PARAM_SIZE, MPI_DOUBLE, PARAMETER_SERVER, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

            int recv_tag = stat.MPI_TAG;

            if (recv_tag == EXIT_TAG)
            {
                printf("Worker %d received EXIT CODE\n", rank);
                break;
            }
            net.set_parameters(network_parameters);

            Matrix x_batch = Matrix::Random(2, batch_size);
            Matrix y_batch = Matrix::Random(1, batch_size);
            fetch_batches(x, y, batch_size, x_batch, y_batch);

            MPI_Get_count(&stat , MPI_DOUBLE, &count);

            net.batch_fit(x_batch, y_batch);

            std::vector<double> net_grads = net.get_grads();
            MPI_Send(&net_grads[0], NETWORK_PARAM_SIZE, MPI_DOUBLE, PARAMETER_SERVER, GRADIENT_TAG, MPI_COMM_WORLD);
            t++;
        }
    }

    t2 = MPI_Wtime();
    printf("Processor %s, rank %d running Time %f\n", processor_name, rank, t2 - t1);

    MPI_Finalize();

    return 0;
}