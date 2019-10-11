#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"
#include "string.h"

int m;
int n;

void frPCAtest()
{
    FILE* fid;
    fid = fopen("SNAP.dat","r");
    m = 82168;
    n = m;
    int nnz = 948464;
    
    //Fisrtly read the matrix with COO format (index of m and n begins with 1 [not 0])
    mat_coo *A = coo_matrix_new(m, n, nnz);
    printf("start read file...\n");
    A->nnz = nnz;
    int i;
    for(i=0;i<A->nnz;i++)
    {
        double ii, jj;
        double kk;
        fscanf(fid, "%lf %lf %lf", &ii, &jj, &kk);
        A->rows[i] = (int)ii;
        A->cols[i] = (int)jj;
        A->values[i] = kk;
    }
    printf("read file done...\n");

    mat_csr* D = csr_matrix_new();
    
    //Secondly transform it to CSR format
    csr_init_from_coo(D, A);
    coo_matrix_delete(A);
    printf("matrix transform done...\n");
    int k = 128;
    int pass = 12;
    struct timeval start_timeval, end_timeval;
    
    
    //the test for frPCA
    mat *U = matrix_new(m, k);
    mat *S = matrix_new(k, 1);
    mat *V = matrix_new(n, k);
    printf("result for frPCA prepare done...\n");
    
    for(i=0;i<1;i++)
    {
        printf("before frpca");
        gettimeofday(&start_timeval, NULL);
        printf("before frpca");
        frPCA(D, &U, &S, &V, k, pass);
        gettimeofday(&end_timeval, NULL);
        printf("time =  %f\n", get_seconds_frac(start_timeval,end_timeval));
    }
    
    //the test for frPCAt
    mat *U0, *S0, *V0;
    U0 = matrix_new(m, k);
    S0 = matrix_new(k, 1);
    V0 = matrix_new(n, k);
    for(i=0;i<1;i++)
    {
        gettimeofday(&start_timeval, NULL);
        frPCAt(D, &U0, &S0, &V0, k, pass);
        gettimeofday(&end_timeval, NULL);
        printf("time =  %f\n", get_seconds_frac(start_timeval,end_timeval));
    }
    mat *UU, *SS, *VV;
    
    //the test for basic rPCA
    for(i=0;i<1;i++)
    {
        gettimeofday(&start_timeval, NULL);
        randQB_basic_csr(D, k, (pass-1)/2, &UU, &SS, &VV);
        gettimeofday(&end_timeval, NULL);
        printf("time =  %f\n", get_seconds_frac(start_timeval,end_timeval));
    }
    
    //show the singular values of three algorithms
    for(i=0;i<k;i++)
        printf("%f %f %f\n", S->d[k-i-1], S0->d[k-i-1], SS->d[k*i+i]);
}

int main()
{
    frPCAtest();
    return 0;
}
