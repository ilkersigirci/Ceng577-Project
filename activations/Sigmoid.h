#ifndef CENG577_PROJECT_SIGMOID_H
#define CENG577_PROJECT_SIGMOID_H

#include "../Eigen/Core"

class Sigmoid{
private:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
public:
    static inline void activation(const Matrix& X, Matrix& Result){
        Result.array() = (double) 1 / (1 + (-X.array()).exp());
    }

    static inline void apply_jacobian(const Matrix& A, const Matrix& F, Matrix& G){
        G.array() = A.array() * ((double) 1 - A.array()) * F.array();
    }
};


#endif //CENG577_PROJECT_SIGMOID_H
