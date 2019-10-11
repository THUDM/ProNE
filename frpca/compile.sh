#!/bin/bash

icc -O3 -mkl -qopenmp frpca.c frpca_test.c matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -o frpca_test
