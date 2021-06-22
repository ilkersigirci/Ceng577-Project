//
// Created by ozhan on 17.06.2021.
//

#ifndef CENG577_PROJECT_SGD_H
#define CENG577_PROJECT_SGD_H

#include "Eigen/Core"


class SGD {
private:
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Vector::ConstAlignedMapType ConstAlignedMapType;
    typedef Vector::AlignedMapType AlignedMapType;
public:
    double lrate;

    SGD(const double& lrate = (double)0.001): lrate(lrate){}

    void update(ConstAlignedMapType& grad, AlignedMapType& vec){
        vec.noalias() -= lrate * grad;
    }
};


#endif //CENG577_PROJECT_SGD_H
