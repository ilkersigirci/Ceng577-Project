#pragma once

#include <vector>
#include <Eigen>

class Data
{
    public:
    
        Data(Eigen::SparseVector<double> data_, double label, int n_features);

        double dot_product(Eigen::VectorXd weight);

        Eigen::SparseVector<double>& getData();

    private:

        Eigen::SparseVector<double> data;

        double label;
        int n_features;
        bool is_sparse;
};