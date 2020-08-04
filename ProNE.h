#include <iostream>
#include <fstream>
#include <cstdio>
#include <complex>
#include <set>
#include <cmath>
#include <map>
#include <ctime>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <gflags/gflags.h>
#include <redsvd/redsvd.hpp>
#include <boost/math/special_functions/bessel.hpp>

extern "C" {
int main_f(int argc, char** argv);
int main_f2(char* filename, char* emb1, char* emb2, int num_node, int num_step, int num_thread, int num_rank, float theta, float mu);
}
