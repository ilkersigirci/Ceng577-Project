//
// Created by ozhan on 17.06.2021.
//

#ifndef CENG577_PROJECT_NETWORK_H
#define CENG577_PROJECT_NETWORK_H

#include <vector>

#include "Eigen/Core"
#include "SGD.h"
#include "losses/RMSE.h"
#include "layer.h"

class Network {
public:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::RowVectorXi IntegerVector;

    std::vector<BaseLayer*> layers;
    RMSE* loss;

    Network(){
        this->loss = NULL;
    }

    ~Network(){
        const int layer_count = this->layers.size();

        for(int i=0; i<layer_count; i++){
            delete this->layers[i];
        }

        if(this->loss){
            delete this->loss;
        }
    }

    std::vector<std::vector<double>> get_parameters(){
        const int layer_count = this->layers.size();
        std::vector<std::vector<double>> res;
        res.reserve(layer_count);

        for(int i=0; i<layer_count; i++){
            res.push_back(this->layers[i]->get_parameters());
        }

        return res;
    }
    template <typename X, typename Y>
    void batch_fit(const Eigen::MatrixBase<X>& x,
                   const Eigen::MatrixBase<Y>& y, SGD& sgd){
        this->forward(x);
        this->backward(x, y);
        this->step(sgd);
    }

    void forward(const Matrix& input){
        const int num_layer = this->layers.size();

        if (num_layer <= 0){
            return;
        }
        if (input.rows() != this->layers[0]->in_size){
            //TODO: remove
            throw std::invalid_argument("[class Network]: Input data have incorrect dimension");
        }

        // input layer
        this->layers[0]->forward(input);

        // hidden layers
        for(int i = 1; i < num_layer; i++){
            this->layers[i]->forward(this->layers[i - 1]->forward_res);
        }
    }
    template <typename T>
    void backward(const Matrix& input, const T& target){
        const int num_layer = this->layers.size();

        if (num_layer <= 0){
            return;
        }

        BaseLayer* first_layer = this->layers[0];
        BaseLayer* last_layer = this->layers[num_layer - 1];

        this->loss->evaluate(last_layer->forward_res, target);

        if (num_layer == 1){
            first_layer->backward(input, this->loss->backward());
            return;
        }

        last_layer->backward(this->layers[num_layer-2]->forward_res, this->loss->output);

        for (int i = num_layer - 2; i > 0; i--){
            this->layers[i]->backward(this->layers[i - 1]->forward_res, this->layers[i + 1]->d_input);
        }

        first_layer->backward(input, this->layers[1]->d_input);
    }

    void step(SGD& sgd){
        const int num_layer = this->layers.size();
        if (num_layer <= 0){
            return;
        }

        for (int i = 0; i < num_layer; i++){
            this->layers[i]->update(sgd);
        }
    }

    Matrix predict(const Matrix& x){
        const int num_layer = this->layers.size();

        if (num_layer <= 0){
            return Matrix();
        }

        this->forward(x);
        return this->layers[num_layer-1]->forward_res;
    }


};


#endif //CENG577_PROJECT_NETWORK_H