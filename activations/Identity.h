#ifndef CENG577_PROJECT_IDENTITY_H
#define CENG577_PROJECT_IDENTITY_H

#include "../Eigen/Core"

class Identity{
private:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
public:
    static inline void activation(const Matrix& X, Matrix& Result){
        Result.noalias() = X;
    }

    static inline void apply_jacobian(const Matrix& A, const Matrix& F, Matrix& G){
        G.noalias() = F;
    }
};


#endif //CENG577_PROJECT_IDENTITY_H
