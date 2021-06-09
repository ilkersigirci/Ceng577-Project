#include "a_sgd.hpp"

double dot_product(Data &data, Eigen::VectorXd &weight)
{
    return data.getData().dot(weight);
}