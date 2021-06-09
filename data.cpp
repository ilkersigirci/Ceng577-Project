#include "data.hpp"

Data::Data(Eigen::SparseVector<double> data, double label, int n_features)
{
	this->data       = data;
	this->label      = label;
	this->n_features = n_features;
}

Eigen::SparseVector<double>& Data::getData()
{
    return this->data;
}