#include <iostream>
#include <fstream>
#include <cstdio>
#include <complex>
#include <set>
#include <cmath>
#include <map>
#include <ctime>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <gflags/gflags.h>
#include <redsvd/redsvd.hpp>
#include <boost/math/special_functions/bessel.hpp>

using namespace Eigen;
using namespace REDSVD;
using namespace boost;
using namespace std;

const float EPS = 0.00000000001f;

DEFINE_string(filename, "test.ungraph", "Filename for edgelist file.");
DEFINE_string(emb1, "sparse.emb", "Filename for svd results.");
DEFINE_string(emb2, "spectral.emb", "Filename for svd results.");
DEFINE_int32(num_node, 4, "Number of node in the graph.");
DEFINE_int32(num_rank, 2, "Embedding dimension.");
DEFINE_int32(num_step, 5, "Number of order for recursion.");
DEFINE_int32(num_iter, 2, "Number of iter in randomized svd.");
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

void printSmf(SMatrixXf & mat){
    for (int k=0; k<mat.outerSize(); ++k)
          for (SMatrixXf::InnerIterator it(mat,k); it; ++it)
            cout <<"(" <<k << ", "<<it.col()<<", "<<it.value()<<")"<<endl;
}


MatrixXf & svdFlip(MatrixXf & mat){
    VectorXf max_abs_num = mat.cwiseAbs().colwise().maxCoeff(); 
    for (int i = 0; i < mat.cols(); ++i){
        float sign = max_abs_num(i) >= 0? 1.0:-1.0;
        mat.col(i) = mat.col(i) * sign;
      }
    return mat;
}

MatrixXf randomizedRangeFinder(SMatrixXf &A, int size, int num_iter){
    int n_samples = A.rows(), n_features= A.cols();
    MatrixXf Q = MatrixXf::Random(n_features, size), L(n_samples, size);
    Eigen::FullPivLU<MatrixXf> lu1(n_samples, size);
    Eigen::FullPivLU<MatrixXf> lu2(n_features, size);
    for(int i=0; i<num_iter;i++)
    {
        lu1.compute(A * Q);
        L.setIdentity();
        L.block(0, 0, n_samples, size).triangularView<Eigen::StrictlyLower>() = lu1.matrixLU();
        L = lu1.permutationP().inverse() * L; 

        lu2.compute(A.transpose() * L);
        Q.setIdentity();
        Q.block(0, 0, n_features, size).triangularView<Eigen::StrictlyLower>() = lu2.matrixLU();
        Q = lu2.permutationP().inverse() * Q;
    }
    Eigen::ColPivHouseholderQR<MatrixXf> qr(A * Q);
    // return qr.colsPermutation().inverse() * qr.householderQ();
    return qr.householderQ() * MatrixXf::Identity(n_samples, size);
}

MatrixXf randomizedSvd(SMatrixXf &data, int rank, int num_iter){
    int n_oversamples = 10;
    int n_random = rank + n_oversamples;
    int n_samples = data.rows(), n_features= data.cols();
    if(n_random > min(n_samples, n_features))
        n_random = min(n_samples, n_features);
    
    MatrixXf Q = randomizedRangeFinder(data, n_random, num_iter);
    cout <<"Q computed done"<<endl;
    MatrixXf B = Q.transpose() * data;
    
    // Eigen::JacobiSVD<MatrixXf> svdOfB(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::BDCSVD<Eigen::MatrixXf> svdOfB(B, Eigen::ComputeThinU);
    
    VectorXf s = svdOfB.singularValues();
    // MatrixXf V = svdOfB.matrixV();
    MatrixXf U = Q * svdOfB.matrixU();

    U = svdFlip(U);
    
    MatrixXf newU = U.block(0, 0, n_samples, rank);
    // MatrixXf V = svdOfB.matrixV().block(0, 0, n_samples, rank);
    VectorXf newS = s.head(rank);

    MatrixXf emb = newU * newS.cwiseSqrt().asDiagonal();

    emb = l2Normalize(emb);
    return emb;
}


MatrixXf getEmbbeddingViaSvd(SMatrixXf &data, int rank){
    RedSVD redsvd;
    redsvd.run(data, rank);
    MatrixXf emb = redsvd.matrixU() * redsvd.singularValues().cwiseSqrt().asDiagonal();
    emb = l2Normalize(emb);
    return emb;
}

MatrixXf getEmbbeddingViaDenseSvd(MatrixXf &data, int rank){
    // Eigen::JacobiSVD<Eigen::MatrixXf> svdOfC(data, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Eigen::BDCSVD<Eigen::MatrixXf> svdOfC(data, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::BDCSVD<Eigen::MatrixXf> svdOfC(data, Eigen::ComputeThinU);
    MatrixXf emb = svdOfC.matrixU() * svdOfC.singularValues().cwiseSqrt().asDiagonal();
    emb = l2Normalize(emb);
    return emb;
}

MatrixXf getSparseEmbedding(SMatrixXf & A, int rank, int num_iter){
    time_t t1 = time(NULL);
    // cout << "number of nnz: "<< A.nonZeros() <<endl;
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
    // printSmf(F);
    //cout << "number of nnz: "<< F.nonZeros() <<endl;

    MatrixXf emb = getEmbbeddingViaSvd(F, rank);
    //MatrixXf emb = randomizedSvd(F, rank, num_iter);
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
    // cout << "number of nnz: "<< M.nonZeros() <<endl;
    // printSmf(M);

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
    MatrixXf F = A * (a - conv);
    cout << "Chebyshev time: "<< (time(NULL) - t1 + 0.0) << endl;
    time_t t2 = time(NULL);

    MatrixXf emb = getEmbbeddingViaDenseSvd(F, rank);
    cout << "dense svd time: "<< (time(NULL) - t2 + 0.0) << endl;
    return emb;
}


int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    time_t start_time = time(NULL);
    Eigen::setNbThreads(FLAGS_num_thread);

    SMatrixXf A = readGraph(FLAGS_filename, FLAGS_num_node);
    time_t t1 = time(NULL);
    cout << "Running time of read graph: " << (t1 - start_time + 0.0)<< endl;

    MatrixXf feature = getSparseEmbedding(A, FLAGS_num_rank, FLAGS_num_iter);
    time_t t2 = time(NULL);
    cout << "Running time of get sparse embedding: " << (t2 - t1 + 0.0) << endl;

    MatrixXf embedding = getSpectralEmbedding(A, feature, FLAGS_num_step, FLAGS_theta, FLAGS_mu);
    time_t t3 = time(NULL);
    cout << "Running time of get spectral embedding: " << (t3 - t2 + 0.0)  << endl;
    cout << "Running time of ProNE: " << (t3 - start_time + 0.0) << endl;
    saveEmbedding(feature, FLAGS_emb1);
    saveEmbedding(feature, FLAGS_emb2);

}
