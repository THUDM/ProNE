#include <stdio.h>
#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"
#include <math.h>
#include "mkl.h"

/* C = beta*C + alpha*A(1:Anrows, 1:Ancols)[T]*B(1:Bnrows, 1:Bncols)[T] */
void submatrix_submatrix_mult_with_ab(mat *A, mat *B, mat *C, 
        int Anrows, int Ancols, int Bnrows, int Bncols, int transa, int transb, double alpha, double beta) {

    int opAnrows, opAncols, opBnrows, opBncols;
    if (transa == CblasTrans) {
        opAnrows = Ancols;
        opAncols = Anrows;
    } else {
        opAnrows = Anrows;
        opAncols = Ancols;
    }
    
    if (transb == CblasTrans) {
        opBnrows = Bncols;
        opBncols = Bnrows;
    } else {
        opBnrows = Bnrows;
        opBncols = Bncols;
    }
    
    if (opAncols != opBnrows) {
        printf("error in submatrix_submatrix_mult()");
        exit(0);
    }
    
    cblas_dgemm(CblasColMajor, (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb, 
        opAnrows, opBncols, // m, n, 
        opAncols, // k
        alpha, A->d, A->nrows, // 1, A, rows of A as declared in memory 
        B->d, B->nrows, // B, rows of B as declared in memory 
        beta, C->d, C->nrows // 0, C, rows of C as declared.
    );
}

void submatrix_submatrix_mult(mat *A, mat *B, mat *C, int Anrows, int Ancols, int Bnrows, int Bncols, int transa, int transb) {
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    submatrix_submatrix_mult_with_ab(A, B, C, Anrows, Ancols, Bnrows, Bncols, transa, transb, alpha, beta);
}


/* D = M(:,inds)' */
void matrix_get_selected_columns_and_transpose(mat *M, int *inds, mat *Mc) {
    int i;
    vec *col_vec; 
    #pragma omp parallel shared(M,Mc,inds) private(i,col_vec) 
    {
    #pragma omp parallel for
    for(i=0; i<(Mc->nrows); i++){
        col_vec = vector_new(M->nrows);
        matrix_get_col(M,inds[i],col_vec);
        matrix_set_row(Mc,i,col_vec);
        vector_delete(col_vec);
    }
    }
}

void matrix_set_selected_rows_with_transposed(mat *M, int *inds, mat *Mc) {
    int i;
    vec *col_vec; 
    #pragma omp parallel shared(M,Mc,inds) private(i,col_vec) 
    {
    #pragma omp parallel for
    for(i=0; i<(Mc->ncols); i++){
        col_vec = vector_new(Mc->nrows); 
        matrix_get_col(Mc,i,col_vec);
        matrix_set_row(M,inds[i],col_vec);
        vector_delete(col_vec);
    }
    }
}

void linear_solve_UTxb(mat *A, mat *b) {
    LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',   //
        A->nrows,
        b->ncols, 
        A->d,
        A->nrows,
        b->d,
        b->nrows
    );
}


mat_coo* coo_matrix_new(int nrows, int ncols, int capacity) {
    mat_coo *M = (mat_coo*)malloc(sizeof(mat_coo));
    M->values = (double*)calloc(capacity, sizeof(double));
    M->rows = (int*)calloc(capacity, sizeof(int));
    M->cols = (int*)calloc(capacity, sizeof(int));
    M->nnz = 0;
    M->nrows = nrows; M->ncols = ncols;
    M->capacity = capacity;
    return M;
}

void coo_matrix_delete(mat_coo *M) {
    free(M->values);
    free(M->cols);
    free(M->rows);
    free(M);
}

void coo_matrix_print(mat_coo *M) {
    int i;
    for (i = 0; i < M->nnz; i++) {
        printf("(%d, %d: %f), ", *(M->rows+i), *(M->cols+i), *(M->values+i));
    }
    printf("\n");
}

// 0-based interface
void set_coo_matrix_element(mat_coo *M, int row, int col, double val, int force_new) {
    if (!(row >= 0 && row < M->nrows && col >=0 && col < M->ncols)) {
        printf("error: wrong index\n");
        exit(0);
    }
    if (!force_new) {
        int i;
        for (i = 0; i < M->nnz; i++) {
            if (*(M->rows + i) == row+1 && *(M->cols + i) == col+1) {
                *(M->values + i) = val;
                return;
            }
        }
    }
    if (M->nnz < M->capacity) {
        *(M->rows+M->nnz) = row+1;
        *(M->cols+M->nnz) = col+1;
        *(M->values+M->nnz) = val;
        M->nnz = M->nnz+1;
        return;
    }
    printf("error: capacity exceeded. capacity=%d, nnz=%d\n", M->capacity, M->nnz);
    exit(0);
}

void coo_matrix_matrix_mult(mat_coo *A, mat *B, mat *C) {
    /* 
    void mkl_dcoomm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *rowind , const MKL_INT *colind , 
        const MKL_INT *nnz , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    double alpha = 1.0, beta = 0.0;
    const char *trans = "N";
    const char *metadescra = "GXXF";
    mkl_dcoomm(
        trans, &(A->nrows), &(C->ncols), 
        &(A->ncols), &(alpha), metadescra, 
        A->values, A->rows, A->cols,
        &(A->nnz), B->d, &(B->nrows),
        &(beta), C->d, &(C->nrows));
}

void coo_matrix_transpose_matrix_mult(mat_coo *A, mat *B, mat *C) {
    /* 
    void mkl_dcoomm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *rowind , const MKL_INT *colind , 
        const MKL_INT *nnz , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    double alpha = 1.0, beta = 0.0;
    const char *trans = "T";
    const char *metadescra = "GXXF";
    mkl_dcoomm(
        trans, &(A->nrows), &(C->ncols), 
        &(A->ncols), &(alpha), metadescra, 
        A->values, A->rows, A->cols,
        &(A->nnz), B->d, &(B->nrows),
        &(beta), C->d, &(C->nrows));
}

void coo_matrix_copy_to_dense(mat_coo *A, mat *B) {
    int i, j;
    // printf("z1\n");
    for (i = 0; i < B->nrows; i++) {
        for (j = 0; j < B->ncols; j++) {
            matrix_set_element(B, i, j, 0.0);
        }
    }
    // printf("z2\n");
    for (i = 0; i < A->nnz; i++) {
        matrix_set_element(B, *(A->rows+i)-1, *(A->cols+i)-1, *(A->values+i) );
    }
    // printf("z3\n");
}


double get_rand_uniform(VSLStreamStatePtr stream) {
    double ans;
    vdRngUniform( VSL_RNG_METHOD_UNIFORM_STD , stream, 1, &ans, 0.0, 1.0);
    return ans;
}

double get_rand_normal(VSLStreamStatePtr stream) {
    double ans;
    vdRngUniform( VSL_RNG_METHOD_UNIFORM_STD , stream, 1, &ans, 0.0, 1.0);
    return ans;
}

void gen_rand_coo_matrix(mat_coo *M, double density) {
    VSLStreamStatePtr stream_u;
    VSLStreamStatePtr stream_n;
    // vslNewStream( &stream_u, BRNG, time(NULL));
    // vslNewStream( &stream_n, BRNG, time(NULL));
    vslNewStream( &stream_u, BRNG, 123);
    vslNewStream( &stream_n, BRNG, 456);
    int i, j;
    for (i = 0; i < M->nrows; i++) {
        for (j = 0; j < M->ncols; j++) {
            if (get_rand_uniform(stream_u) < density) {
                set_coo_matrix_element(M, i, j, get_rand_normal(stream_n), 1);
            }
        }
    }
}

void coo_matrix_sort_element(mat_coo *A) {
    int i, j;
    // seletion sort
    for (i = 0; i < A->nnz; i++) {
        for (j = i+1; j < A->nnz; j++) {
            if ( (A->rows[i] > A->rows[j]) || 
                (A->rows[i] == A->rows[j] && A->cols[i] > A->cols[j]) ) {
                double dtemp; int itemp;
                itemp = A->rows[i];
                A->rows[i] = A->rows[j];
                A->rows[j] = itemp;
                itemp = A->cols[i];
                A->cols[i] = A->cols[j];
                A->cols[j] = itemp;
                dtemp = A->values[i];
                A->values[i] = A->values[j];
                A->values[j] = dtemp;
            }
        }
    }
}

void csr_matrix_delete(mat_csr *M) {
    free(M->values);
    free(M->cols);
    free(M->pointerB);
    free(M->pointerE);
    free(M);
}

void csr_matrix_print(mat_csr *M) {
    int i;
    printf("values: ");
    for (i = 0; i < M->nnz; i++) {
        printf("%f ", M->values[i]);
    }
    printf("\ncolumns: ");
    for (i = 0; i < M->nnz; i++) {
        printf("%d ", M->cols[i]);
    }
    printf("\npointerB: ");
    for (i = 0; i < M->nrows; i++) {
        printf("%d\t", M->pointerB[i]);
    }
    printf("\npointerE: ");
    for (i = 0; i < M->nrows; i++) {
        printf("%d\t", M->pointerE[i]);
    }
    printf("\n");
}

mat_csr* csr_matrix_new() {
    mat_csr *M = (mat_csr*)malloc(sizeof(mat_csr));
    return M;
}

void csr_init_from_coo(mat_csr *D, mat_coo *M) {
    D->nrows = M->nrows; 
    D->ncols = M->ncols;
    D->pointerB = (int*)malloc(D->nrows*sizeof(int));
    D->pointerE = (int*)malloc(D->nrows*sizeof(int));
    D->cols = (int*)calloc(M->nnz, sizeof(int));
    D->nnz = M->nnz;
    
    // coo_matrix_sort_element(M);
    D->values = (double*)malloc(M->nnz * sizeof(double));
    memcpy(D->values, M->values, M->nnz * sizeof(double));
    
    int current_row, cursor=0;
    for (current_row = 0; current_row < D->nrows; current_row++) {
        D->pointerB[current_row] = cursor+1;
        while (M->rows[cursor]-1 == current_row) {
            D->cols[cursor] = M->cols[cursor];
            cursor++;
        }
        D->pointerE[current_row] = cursor+1;
    }
}

void csr_matrix_matrix_mult(mat_csr *A, mat *B, mat *C) {
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb , 
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char * transa = "N";
    double alpha = 1.0, beta = 0.0;
    const char *matdescra = "GXXF";
    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
        &(A->ncols), &alpha, matdescra,
        A->values, A->cols, A->pointerB,
        A->pointerE, B->d, &(B->nrows),
        &beta, C->d, &(C->nrows));
}

void csr_matrix_transpose_matrix_mult(mat_csr *A, mat *B, mat *C) {
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb , 
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char * transa = "T";
    double alpha = 1.0, beta = 0.0;
    const char *matdescra = "GXXF";
    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
        &(A->ncols), &alpha, matdescra,
        A->values, A->cols, A->pointerB,
        A->pointerE, B->d, &(B->nrows),
        &beta, C->d, &(C->nrows));
}

