#ifndef CENG577_PROJECT_UTILS_H
#define CENG577_PROJECT_UTILS_H

void shuffle_inplace(int* arr, int size){
    std::random_shuffle(arr, arr + size);
}

template <typename DerivedX, typename DerivedY>
void fetch_batches(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y,
                   int batch_size, Eigen::MatrixBase<DerivedX>& x_batch, Eigen::MatrixBase<DerivedY>& y_batch){

    const int total_size = x.cols();

    Eigen::VectorXi id = Eigen::VectorXi::LinSpaced(total_size, 0, total_size - 1);
    shuffle_inplace(id.data(), total_size);

    for(int i=0; i<batch_size; i++) {
        x_batch.col(i).noalias() = x.col(id[i]);
        y_batch.col(i).noalias() = y.col(id[i]);
    }
}


#endif //CENG577_PROJECT_UTILS_H
