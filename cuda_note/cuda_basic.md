## threadIdx、blockIdx、blockDim和gridDim

threadIdx是一个uint3类型，表示一个线程的索引

blockIdx是一个uint3类型，表示一个线程块的索引，一个线程块中通常有多个线程

blockDim是一个dim3类型，表示线程块的大小，其中包含3个uint成员变量

gridDim是一个dim3类型，表示网格的大小，一个网格中通常有多个线程块

```c++
struct __device_builtin__ dim3
{
    unsigned int x, y, z;
#if defined(__cplusplus)
    __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __host__ __device__ operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif /* __cplusplus */
};
```

​        对于一个9x9x9的3维张量来说，可以设定gridDim = {3, 3, 3}, blockDim = {3, 3, 3} threadIdx的范围是{[0-2], [0-2], [0-2]}, blockIdx的范围是{[0-2], [0-2], [0-2]}



## kernel_func_call

kernel_func<<<block_num, thread_num, shared_memory_size>>>(parameter_1, paramter_2, ...)

block_num和thread_num作为参数构造了dim3对象，构造对象时，没有给参数赋值的都默认为1

参考：https://www.cnblogs.com/1024incn/p/4605502.html

