//
// Created by bedirhan on 15.06.2021.
//

#ifndef CENG577_PROJECT_LOSS_H
#define CENG577_PROJECT_LOSS_H

#include "Eigen/Core"

class Loss {
private:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

    Matrix output;
public:
    // calculate layer output
    void evaluate(const Matrix& input, const Matrix& target){
            const int c = input.cols();
            const int r = input.rows();

            output.resize(r, c);

            output.noalias() = input - target;
    }

    double loss() const{
        return output.squaredNorm() / output.cols();
    }

    Matrix& backward(){
        return output;
    }


};


#endif //CENG577_PROJECT_LOSS_H
