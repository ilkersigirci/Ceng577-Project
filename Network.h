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

    std::vector<double> get_grads(){
        const int layer_count = this->layers.size();
        std::vector<double> res;

        for(int i=0; i<layer_count; i++){
            std::vector<double> layer_grads = this->layers[i]->get_grads();
            res.insert(res.end(), layer_grads.begin(), layer_grads.end());
        }

        return res;
    }

    std::vector<double> get_parameters(){
        const int layer_count = this->layers.size();
        std::vector<double> res;

        for(int i=0; i<layer_count; i++){
            std::vector<double> layer_params = this->layers[i]->get_parameters();
            res.insert(res.end(), layer_params.begin(), layer_params.end());
        }

        return res;
    }

    template <typename X, typename Y>
    void batch_fit(const Eigen::MatrixBase<X>& x,
                   const Eigen::MatrixBase<Y>& y){
        this->forward(x);
        this->backward(x, y);
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

    void set_grads(std::vector<double>& grads){
        int network_param_size = 0;

        for(int i=0; i<this->layers.size(); i++){
            network_param_size += this->layers[i]->get_param_size();
        }

        if(network_param_size != grads.size()){
            throw std::invalid_argument("[class Network]: grads size is not match with network parameter size");
        }

        int offset = 0;
        for(int i=0; i<this->layers.size(); i++){
            int layer_size = this->layers[i]->get_param_size();
            std::vector<double> layer_grads = slicing(grads, offset, offset + layer_size);
            this->layers[i]->set_grads(layer_grads);
            offset += layer_size;
        }
    }


    void set_parameters(std::vector<double>& params){
        int network_param_size = 0;

        for(int i=0; i<this->layers.size(); i++){
            network_param_size += this->layers[i]->get_param_size();
        }

        if(network_param_size != params.size()){
            throw std::invalid_argument("[class Network]: params size is not match with network parameter size");
        }

        int offset = 0;
        for(int i=0; i<this->layers.size(); i++){
            int layer_size = this->layers[i]->get_param_size();
            std::vector<double> layer_params = slicing(params, offset, offset + layer_size);
            this->layers[i]->set_parameters(layer_params);
            offset += layer_size;
        }
    }


    // https://www.geeksforgeeks.org/slicing-a-vector-in-c/
    std::vector<double> slicing(std::vector<double>& arr,
                        int X, int Y)
    {
        // Starting and Ending iterators
        auto start = arr.begin() + X;
        auto end = arr.begin() + Y;

        // To store the sliced vector
        std::vector<double> result(Y - X);

        // Copy vector using copy function()
        std::copy(start, end, result.begin());

        // Return the final sliced vector
        return result;
    }

};


#endif //CENG577_PROJECT_NETWORK_H
