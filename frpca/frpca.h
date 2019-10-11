
#include "matrix_vector_functions_intel_mkl_ext.h"
#include <math.h>

void LUfraction(mat *A, mat *L);

void eigSVD(mat* A, mat **U, mat **S, mat **V);

void frPCAt(mat_csr *A, mat **U, mat **S, mat **V, int k, int q);


void frPCA(mat_csr *A, mat **U, mat **S, mat **V, int k, int q);


void randQB_basic_csr(mat_csr *M, int k, int p, mat **U, mat **S, mat **V);