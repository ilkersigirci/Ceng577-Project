//#include "Eigen/Core"
//#include "activations/Sigmoid.h"
//#include "activations/Identity.h"
//#include "layer.h"
//#include "Network.h"
//#include "losses/RMSE.h"
//#include "SGD.h"
//#include "utils.h"
//
//typedef Eigen::MatrixXd Matrix;
//typedef Eigen::VectorXd Vector;
//
//
//int main(){
//    Vector x1 = Vector::LinSpaced(1000, 0.0, 3.15);
//    Vector x2 = Vector::LinSpaced(1000, 0.0, 3.15);
//    Matrix x = Matrix::Random(2, 1000);
//    x.row(0) = x1;
//    x.row(1) = x2;
//    Matrix y = Matrix::Random(1, 1000);
//
//    // Fill the output for the training
//    for (int i = 0; i < y.cols(); i++)
//    {
//        y(0, i) = std::pow(x(0, i), 2) + std::pow(x(1, i), 2);
//    }
//
//    // Fill the output for the test
//    Matrix xt = (Matrix::Random(2, 1000).array() + 1.0) / 2 * 3.15;
//    Matrix yt = Matrix::Random(1, 1000);
//
//    for (int i = 0; i < yt.cols(); i++)
//    {
//        yt(0, i) = std::pow(xt(0, i), 2) + std::pow(xt(1, i), 2);
//    }
//
//    Network net;
//    BaseLayer* layer1 = new FullyConnectedLayer<Identity>(2, 200);
//    BaseLayer* layer2 = new FullyConnectedLayer<Sigmoid>(200, 200);
//    BaseLayer* layer3 = new FullyConnectedLayer<Identity>(200, 1);
//
//
//    // Add layers to the network object
//    net.layers.push_back(layer1);
//    net.layers.push_back(layer2);
//    net.layers.push_back(layer3);
//
//    // specify loss function
//    net.loss = new RMSE();
//
//    SGD sgd;
//
//    int num_iters = 10000;
//    const int batch_size = 8;
//
//    int j;
//    for(int i=0; i<num_iters; i++){
//        Matrix x_batch = Matrix::Random(2, batch_size);
//        Matrix y_batch = Matrix::Random(1, batch_size);
//        fetch_batches(x, y, batch_size, x_batch, y_batch);
//        net.batch_fit(x_batch, y_batch, sgd);
//    }
//
//    Matrix t = Matrix::Random(2, 1);
//    std::cout<<t<<std::endl;
//    std::cout << net.predict(t) << std::endl;
//}