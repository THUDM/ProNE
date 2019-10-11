#include "frpca.h"

/*[L, ~] = lu(A) as in MATLAB*/
void LUfraction(mat *A, mat *L)
{
    matrix_copy(L, A);
    // printf("after matrix_copy\n");
    int *ipiv = (int *)malloc(sizeof(int)*L->nrows);
    // printf("before LAPACKE_dgetrf\n");    
    LAPACKE_dgetrf (LAPACK_COL_MAJOR, L->nrows, L->ncols, L->d, L->nrows, ipiv);   
    // printf("after LAPACKE_dgetrf\n");    
    int i,j;
    #pragma omp parallel private(i,j) 
    {
    #pragma omp for     
        for(i=0;i<L->ncols;i++)
        {
            for(j=0;j<i;j++)
            {
                L->d[i*L->nrows+j] = 0;
            }
            L->d[i*L->nrows+i] = 1;
        }
    }
    
    {    
        for(i=L->ncols-1;i>=0;i--)
        {
            int ipi = ipiv[i]-1;
            for(j=0;j<L->ncols;j++)
            {
                double temp = L->d[j*L->nrows+ipi];
                L->d[j*L->nrows+ipi] = L->d[j*L->nrows+i];
                L->d[j*L->nrows+i] = temp;
            }
        }
    }
}

/*[U, S, V] = eigSVD(A)*/
void eigSVD(mat* A, mat **U, mat **S, mat **V)
{
    matrix_transpose_matrix_mult(A, A, *V);
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (*V)->ncols, (*V)->d, (*V)->ncols, (*S)->d);
    mat *V1 = matrix_new((*V)->ncols, (*V)->ncols);
    matrix_copy(V1, (*V));
    int i, j;
    // printf("before parallel shared\n");
    #pragma omp parallel shared(V1,S) private(i,j) 
    {
    #pragma omp for 
        for(i=0; i<V1->ncols; i++)
        {
            (*S)->d[i] = sqrt((*S)->d[i]);
            for(j=0; j<V1->nrows;j++)            
                V1->d[i*V1->nrows+j] /= (*S)->d[i];
        }
    }
    // printf("before matrix_matrix_mult\n");
    matrix_matrix_mult(A, V1, (*U));
}

/*[U, S, V] = frSVD(A, k, p)*/
void frPCAt(mat_csr *A, mat **U, mat **S, mat **V, int k, int q)
{
    int s = 5;    
    mat *Q = matrix_new(A->nrows, k+s);
    mat *Qt = matrix_new(A->ncols, k+s);
    //mat *UU = matrix_new(A->nrows, k+s);
    mat *SS = matrix_new(k+s, 1);
    mat *VV = matrix_new(k+s, k+s);
    if(q%2 == 0)
    {
        initialize_random_matrix(Q);
        csr_matrix_transpose_matrix_mult(A, Q, Qt);
        if(q==2)
        {
            eigSVD(Qt, &Qt, &SS, &VV);
        }
        else
        {
            LUfraction(Qt, Qt);
        }
    }
    else
    {
        initialize_random_matrix(Qt);
    }
    int niter = (q-1)/2, i;
    for(i=1;i<=niter;i++)
    {
        csr_matrix_matrix_mult(A, Qt, Q);
        csr_matrix_transpose_matrix_mult(A, Q, Qt);
        if(i==niter)
        {
            eigSVD(Qt, &Qt, &SS, &VV);
        }
        else
        {
            LUfraction(Qt, Qt);
        }
    }
    csr_matrix_matrix_mult(A, Qt, Q);
    eigSVD(Q, &Q, &SS, &VV);
    int inds[k]; 
    for(i=s;i<k+s;i++)
    {
        inds[i-s] = i;
    }
    mat *VV2 = matrix_new(k+s, k);
    matrix_get_selected_columns(Q, inds, *U);
    matrix_get_selected_rows(SS, inds, *S);
    matrix_get_selected_columns(VV, inds, VV2);
    matrix_matrix_mult(Qt, VV2, (*V));
}

/*[U, S, V] = frSVD(A, k, p)*/
void frPCA(mat_csr *A, mat **U, mat **S, mat **V, int k, int q)
{
    // printf("before conduct frpca\n");
    int s = 5;    
    mat *Q = matrix_new(A->nrows, k+s);
    mat *Qt = matrix_new(A->ncols, k+s);
    mat *UU = matrix_new(A->ncols, k+s);
    mat *SS = matrix_new(k+s, 1);
    mat *VV = matrix_new(k+s, k+s);
    // printf("before recurence\n");
    if(q%2 == 0)
    {
        initialize_random_matrix(Qt);
        // printf("after initialize_random_matrix\n");
        csr_matrix_matrix_mult(A, Qt, Q);
        // printf("after csr_matrix_matrix_mult\n");

        if(q==2)
        {
            // printf("before eigSVD\n");
            eigSVD(Q, &Q, &SS, &VV);
            // printf("after eigSVD\n");
        }
        else
        {
            // printf("before LUfraction\n");
            LUfraction(Q, Q);
            // printf("after LUfraction\n");
        }
    }
    else
    {
        initialize_random_matrix(Q);
        // printf("after initialize_random_matrix\n");
    }
    // printf("before iteration\n");
    int niter = (q-1)/2, i;
    for(i=1;i<=niter;i++)
    {
        // printf("iteration in frpca\n");
        csr_matrix_transpose_matrix_mult(A, Q, Qt);
        // printf("after csr_matrix_transpose_matrix_mult\n");
        csr_matrix_matrix_mult(A, Qt, Q);
        if(i==niter)
        {
            eigSVD(Q, &Q, &SS, &VV);
        }
        else
        {
            LUfraction(Q, Q);
        }
    }
    csr_matrix_transpose_matrix_mult(A, Q, Qt);
    eigSVD(Qt, &UU, &SS, &VV);
    int inds[k]; 
    for(i=s;i<k+s;i++)
    {
        inds[i-s] = i;
    }
    mat *VV2 = matrix_new(k+s, k);
    matrix_get_selected_columns(UU, inds, *V);
    matrix_get_selected_rows(SS, inds, *S);
    matrix_get_selected_columns(VV, inds, VV2);
    matrix_matrix_mult(Q, VV2, (*U));
}

void randQB_basic_csr(mat_csr *M, int k, int p, mat **U, mat **S, mat **V) {
    int m, n, i, l=k+5;
    m = M->nrows; n = M->ncols;
    
    mat *Q = matrix_new(m, l);
    mat *B = matrix_new(l, n);
    mat *Vt = matrix_new(l, n);
    mat *VV = matrix_new(n, l);
    mat *UU = matrix_new(l, l);
    mat *UUk = matrix_new(l, k);
    mat *SS = matrix_new(l, l);
    *U = matrix_new(m ,k); 
    *V = matrix_new(n, k);
    *S = matrix_new(k, k);
    
    // samples
    mat *R, *G, *Bt;
    R = matrix_new(n, l);
    G = Q;
    Bt = R;
    initialize_random_matrix(R);
    csr_matrix_matrix_mult(M, R, G);
    QR_factorization_getQ_inplace(G);
    
    // power iteration
    if (p > 0) {
        for (i = 0; i < p; i++) {
            csr_matrix_transpose_matrix_mult(M, G, R);
            QR_factorization_getQ_inplace(R);
            csr_matrix_matrix_mult(M, R, G);
            QR_factorization_getQ_inplace(G);
        }
    }
    
    //QR_factorization_getQ_inplace(G);
    csr_matrix_transpose_matrix_mult(M, Q, Bt);
    matrix_build_transpose(B, Bt);
    
    singular_value_decomposition(B, UU, SS, Vt);
    matrix_build_transpose(VV, Vt);
    int inds[k]; 
    for(i=0;i<k;i++)
    {
        inds[i] = i;
    }
    matrix_get_selected_columns(UU, inds, UUk);
    matrix_matrix_mult(Q, UUk, (*U));
    matrix_get_selected_columns(VV, inds, (*V));
    matrix_copy_first_k_rows_and_columns(*S, SS);
    
    matrix_delete(R);
    
}
