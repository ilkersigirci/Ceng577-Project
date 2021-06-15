//
// Created by bedirhan on 15.06.2021.
//

#ifndef CENG577_PROJECT_LAYER_H
#define CENG577_PROJECT_LAYER_H


#include "Eigen/Core"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

template<typename Activation>
class FullyConnectedLayer {
private:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

    Matrix weights;
    Vector biases;

    Matrix d_weights;
    Matrix d_biases;

    Matrix y; // y = Ax + b
    Matrix forward_res;

    int in_size;
    int out_size;

public:
    FullyConnectedLayer(const int in_size, const int out_size) {
        this->in_size = in_size;
        this->out_size = out_size;

        this->weights.resize(in_size, out_size);
        this->biases.resize(out_size);

        // initialize weights and biases
        this->weights = -sqrt(in_size) + (Eigen::ArrayXXd::Random(in_size, out_size) * 0.5 + 0.5) * (2 * sqrt(in_size));
        this->biases = -sqrt(in_size) + (Eigen::ArrayXXd::Random(out_size, 1) * 0.5 + 0.5) * (2 * sqrt(in_size));

    }

    void forward(const Matrix& x) {
        const int batch_size = x.cols();

        this->y.resize(this->out_size, batch_size);
        this->y.noalias() = this->weights.transpose() * x;
        this->y.colwise() += this->biases;

        this->forward_res.resize(this->out_size, batch_size);
        this->forward_res.noalias() = Activation::forward(y, res);
    }

    void backward(const Matrix& x, const)

    void printer() {
        std::cout << this->in_size << std::endl;
        std::cout << this->out_size << std::endl;

        std::cout << this->weights << std::endl;
        std::cout << std::endl;
        std::cout << this->biases << std::endl;

    }
};


#endif //CENG577_PROJECT_LAYER_H
