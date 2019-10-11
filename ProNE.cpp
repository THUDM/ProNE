#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <cstdio>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <gflags/gflags.h>
#include <boost/math/special_functions/bessel.hpp>

#include "frpca/frpca.h"
#include "frpca/matrix_vector_functions_intel_mkl.h"
#include "frpca/matrix_vector_functions_intel_mkl_ext.h"

using namespace std;
using namespace Eigen;
using namespace boost;

const float EPS = 0.00000000001f;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SMatrixXf;


DEFINE_string(filename, "data/PPI.ungraph", "Filename for edgelist file.");
DEFINE_string(emb1, "sparse.emb", "Filename for svd results.");
DEFINE_string(emb2, "spectral.emb", "Filename for svd results.");
DEFINE_int32(num_node, 3890, "Number of node in the graph.");
DEFINE_int32(num_rank, 128, "Embedding dimension.");
DEFINE_int32(num_step, 10, "Number of order for recursion.");
DEFINE_int32(num_iter, 5, "Number of iter in randomized svd.");
DEFINE_int32(num_thread, 10, "Number of threads.");
DEFINE_double(theta, 0.5, "Parameter of ProNE");
DEFINE_double(mu, 0.1, "Parameter of ProNE");


SMatrixXf readGraph(string filename, int num_node){
    SMatrixXf A(num_node, num_node);
    typedef Eigen::Triplet<float> T;
    vector<T> tripletList;
    ifstream fin(filename.c_str());
    while (1)
    {
        string x, y;
        if (!(fin >> x >> y))
            break;
        int a = atoi(x.c_str()), b = atoi(y.c_str());
        if (a==b) continue;
        tripletList.push_back(T(a, b, 1));
        tripletList.push_back(T(b, a, 1));
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    return A;
}

SMatrixXf l1Normalize(SMatrixXf & mat){
    SMatrixXf mat2(mat.rows(), mat.cols());
    for (int k=0; k<mat.outerSize(); ++k){
        int num_neighbor = mat.row(k).sum();
        for (SMatrixXf::InnerIterator it(mat,k); it; ++it)
            mat2.insert(k, it.col()) = it.value()/num_neighbor;
    }
    return mat2;
}

MatrixXf & l2Normalize(MatrixXf & mat){
    for (int i = 0; i < mat.rows(); ++i){
        float ssn = sqrt(mat.row(i).squaredNorm());
        if (ssn < EPS) ssn = EPS;
        mat.row(i) = mat.row(i) / ssn;
      }
    return mat;
}

SMatrixXf & validate(SMatrixXf & mat){
    for (int k=0; k<mat.outerSize(); ++k)
          for (SMatrixXf::InnerIterator it(mat,k); it; ++it)
              if (it.value() <=0)
                mat.coeffRef(k, it.col()) = 1;
    return mat;
}

SMatrixXf & smfLog(SMatrixXf & mat){
    for (int k=0; k<mat.outerSize(); ++k)
          for (SMatrixXf::InnerIterator it(mat,k); it; ++it)
              mat.coeffRef(it.row(), it.col()) = log(it.value());
    return mat;
}

float bessel(int a, float b){
    return boost::math::cyl_bessel_i(a, b);
}


MatrixXf getEmbbeddingViaDenseSvd(MatrixXf &data, int rank){
    Eigen::BDCSVD<Eigen::MatrixXf> svdOfC(data, Eigen::ComputeThinU);
    MatrixXf emb = svdOfC.matrixU() * svdOfC.singularValues().cwiseSqrt().asDiagonal();
    emb = l2Normalize(emb);
    return emb;
}


MatrixXf runFrPCA(SMatrixXf & input, int rank, int iter)
{
    int m = input.rows(), nnz = input.nonZeros();
    mat_coo *A = coo_matrix_new(m, m, nnz);
    A->nnz = nnz;
    int i=0;
    for (int k=0; k<input.outerSize(); ++k)
        for (SMatrixXf::InnerIterator it(input,k); it; ++it)
          {
            A->rows[i] = k+1;
            A->cols[i] = it.col()+1;
            A->values[i] = it.value();
            i += 1;
          }
    cout << "read matrix done..." <<endl;
    // coo_matrix_print(A);

    //transform it to CSR format
    mat_csr* D = csr_matrix_new();
    csr_init_from_coo(D, A);
    coo_matrix_delete(A);

    //the test for frPCA
    mat *U = matrix_new(m, rank);
    mat *S = matrix_new(rank, 1);
    mat *V = matrix_new(m, rank);    
    frPCA(D, &U, &S, &V, rank, iter);

    // matrix_print(U);
    // matrix_print(S);

    MatrixXf emb = MatrixXf::Random(m, rank);
    for (int i=0; i<m; i++)
        for (int j=0; j<rank; j++)
            emb(i, j) = matrix_get_element(U,i,j) *  sqrt(matrix_get_element(S,j,0));
    cout << "matrix decomposition done" <<endl;
    return emb;
}


MatrixXf getSparseEmbedding(SMatrixXf & A, int rank, int num_iter){
    time_t t1 = time(NULL);
    int row = A.rows(), col = A.cols();
    SMatrixXf B = l1Normalize(A);
    SMatrixXf C = B.transpose();
    SMatrixXf D(col, col), E(row, col), F(row, col);
    for (int i = 0; i < row; ++i){
        D.insert(i, i) = pow(C.row(i).sum(), 0.75);
    }

    D = D / D.sum();
    E = A * D;

    B = validate(B);
    E = validate(E);

    B = smfLog(B);
    E = smfLog(E);
    F = B - E;
    cout << "preprocess time: "<< (time(NULL) - t1 + 0.0) << endl;
    cout << "number of nnz: "<< F.nonZeros() <<endl;

    MatrixXf emb = runFrPCA(F, rank, num_iter);

    emb = l2Normalize(emb); 
    return emb;
}


MatrixXf getSpectralEmbedding(SMatrixXf & A, MatrixXf & a, int step, float theta, float mu){
    time_t t1 = time(NULL);
    cout << "Chebyshev series --------------- " << endl;
    if (step==1) return a;
    int num_node = a.rows(), rank = a.cols();
    SMatrixXf I(num_node, num_node);
    for (int i = 0; i < num_node; ++i)
        I.insert(i, i) = 1;
    A = A + I;
    SMatrixXf B = l1Normalize(A);
    SMatrixXf L = I - B;
    SMatrixXf M = L - mu * I;


    MatrixXf Lx0 = a;
    MatrixXf Lx1 = M * a, Lx2;
    Lx1 = 0.5 * M * Lx1 - a;

    MatrixXf conv = bessel(0, theta)* Lx0;
    conv -= 2 * bessel(1, theta)* Lx1;
    for(int i=2; i<step; i++){
        Lx2 = M * Lx1;
        Lx2 = (M * Lx2 - 2 * Lx1) - Lx0;

        if (i % 2 == 0)
            conv += 2 * bessel(i, theta) * Lx2;
        else
            conv -= 2 * bessel(i, theta) * Lx2;
        Lx0 = Lx1;
        Lx1 = Lx2;
        cout << "Bessell time: " << i <<"\t"<< (time(NULL) - t1 + 0.0) << endl;
    }
    MatrixXf emb = A * (a - conv);
    cout << "Chebyshev time: "<< (time(NULL) - t1 + 0.0) << endl;
    
    // time_t t2 = time(NULL);
    // MatrixXf emb = getEmbbeddingViaDenseSvd(emb, rank);
    // cout << "dense svd time: "<< (time(NULL) - t2 + 0.0) << endl;
    emb = l2Normalize(emb); 
    return emb;
}


void saveEmbedding(MatrixXf &data, string output){
    int m = data.rows(), d = data.cols();
    FILE *emb = fopen(output.c_str(), "wb");
    fprintf(emb, "%d %d\n", m, d);
    for (int i = 0; i < m; i++)
    {
        fprintf(emb, "%d", i);
        for (int j = 0; j < d; j++)
            fprintf(emb, " %f", data(i, j));
        fprintf(emb, "\n");
    }
    fclose(emb);
}


int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    Eigen::setNbThreads(FLAGS_num_thread);

    time_t t1 = time(NULL);
    SMatrixXf A = readGraph(FLAGS_filename, FLAGS_num_node);

    MatrixXf feature = getSparseEmbedding(A, FLAGS_num_rank, FLAGS_num_iter);
    time_t t2 = time(NULL);
    cout << "Running time of get sparse embedding: " << (t2 - t1 + 0.0) << endl;

    MatrixXf embedding = getSpectralEmbedding(A, feature, FLAGS_num_step, FLAGS_theta, FLAGS_mu);
    time_t t3 = time(NULL);
    cout << "Running time of get spectral embedding: " << (t3 - t2 + 0.0)  << endl;
    cout << "Running time of ProNE: " << (t3 - t1 + 0.0) << endl;

    saveEmbedding(feature, FLAGS_emb1);
    saveEmbedding(embedding, FLAGS_emb2);
    cout << "Embedding save done " << endl;

}
