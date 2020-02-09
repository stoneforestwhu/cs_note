### cublasSgemm

```c++
cublasStatus_t cublasSgemm (cublasHandle_t handle, 
                            cublasOperation_t transa,
                            cublasOperation_t transb, 
                            int m,
                            int n,
                            int k,
                            const float *alpha, /* host or device pointer */  
                            const float *A, 
                            int lda,
                            const float *B,
                            int ldb, 
                            const float *beta, /* host or device pointer */  
                            float *C,
                            int ldc);

```

cublasOperation_t的值为 CUBLAS_OP_N(表示不转置)和CUBLAS_OP_T(表示转置)

矩阵A的维度时 mxn 存储在指针A指向的存储空间

矩阵B的维度是 kxn 存储在指针B指向的存储空间

矩阵C的维度是mxn 存储在指针C指向的存储空间

C = alpha * A * B + beta * C

cublasHandle_t使用函数cublasCreate((cublasHandle_t *handle));进行申请，之后使用cublasDestroy_v2 (cublasHandle_t handle)进行释放。

在cublas里面所有矩阵都是使用列优先进行存储的，因此lda，ldb，ldc表示的是矩阵的行数。