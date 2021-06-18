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

#include "SGD.h"


class BaseLayer{
public:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    Matrix weights;
    Vector biases;

    Matrix d_weights;
    Matrix d_biases;

    Matrix y; // y = Ax + b
    Matrix forward_res; // activation(y)

    Matrix d_input; // derivative w.r.t input

    int in_size;
    int out_size;

    BaseLayer(const int in_size, const int out_size){
        this->in_size=in_size;
        this->out_size=out_size;
    }
    virtual ~BaseLayer(){}

    virtual void forward(const Matrix &x)=0;
    virtual void backward(const Matrix &prev_layer_data, const Matrix &next_layer_data)=0;
    virtual void update(SGD& sgd)=0;
    virtual std::vector<double> get_parameters()=0;
    virtual void set_parameters(const std::vector<double>& param)=0;
    virtual void printer()=0;
};


template<typename Activation>
class FullyConnectedLayer: public BaseLayer {
public:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

    Matrix weights;
    Vector biases;

    Matrix d_weights;
    Matrix d_biases;

    Matrix y; // y = Ax + b
    Matrix forward_res; // activation(y)

    Matrix d_input; // derivative w.r.t input

    int in_size;
    int out_size;

    FullyConnectedLayer(const int in_size, const int out_size): BaseLayer(in_size, out_size) {
        this->weights.resize(in_size, out_size);
        this->biases.resize(out_size);

        // initialize weights and biases
        // Xavier initialization
        this->weights = -sqrt(in_size) + (Eigen::ArrayXXd::Random(in_size, out_size) * 0.5 + 0.5) * (2 * sqrt(in_size));
        this->biases = -sqrt(in_size) + (Eigen::ArrayXXd::Random(out_size, 1) * 0.5 + 0.5) * (2 * sqrt(in_size));
    }

    ~FullyConnectedLayer(){}

    void forward(const Matrix &x) {
        const int batch_size = x.cols();

        this->y.resize(this->out_size, batch_size);
        this->y.noalias() = this->weights.transpose() * x;
        this->y.colwise() += this->biases;

        this->forward_res.resize(this->out_size, batch_size);
        Activation::activation(y, this->forward_res);
    }

    void backward(const Matrix &prev_layer_data, const Matrix &next_layer_data) {
        const int batch_size = prev_layer_data.cols();

        Matrix &dLy = this->y;
        Activation::apply_jacobian(this->forward_res, next_layer_data, dLy);

        this->d_weights.noalias() = prev_layer_data * dLy.transpose() / batch_size;
        this->d_biases = dLy.rowwise().mean();

        d_input.resize(this->in_size, batch_size);
        d_input.noalias() = this->weights * dLy;
    }

    void update(SGD& sgd){
        ConstAlignedMapVec dw(this->d_weights.data(), this->d_weights.size());
        ConstAlignedMapVec db(this->d_biases.data(), this->d_biases.size());
        AlignedMapVec      w(this->weights.data(), this->weights.size());
        AlignedMapVec      b(this->biases.data(), this->biases.size());
        sgd.update(dw, w);
        sgd.update(db, b);
    }

    std::vector<double> get_parameters(){
        std::vector<double> res(this->weights.size() + this->biases.size());
        std::copy(this->weights.data(), this->weights.data() + this->weights.size(), res.begin());
        std::copy(this->biases.data(), this->biases.data() + this->biases.size(),
                  res.begin() + this->weights.size());
        return res;
    }

    void set_parameters(const std::vector<double>& param){
        if (static_cast<int>(param.size()) != this->weights.size() + this->biases.size())
        {
            throw std::invalid_argument("[class FullyConnectedLayer]: Parameter size does not match");
        }

        std::copy(param.begin(), param.begin() + this->weights.size(), this->weights.data());
        std::copy(param.begin() + this->weights.size(), param.end(), this->biases.data());
    }

    void printer() {
        std::cout << this->in_size << std::endl;
        std::cout << this->out_size << std::endl;

        std::cout << this->weights << std::endl;
        std::cout << std::endl;
        std::cout << this->biases << std::endl;

    }
};


#endif //CENG577_PROJECT_LAYER_H
