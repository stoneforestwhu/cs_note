## Appendix A. CUDA-ENABLED GPUS  



​         http://developer.nvidia.com/cuda-gpus lists all CUDA-enabled devices with their compute capability.  

​        The compute capability, number of multiprocessors, clock frequency, total amount of device memory, and other properties can be queried using the runtime (see reference manual).

## Appendix B. C++ LANGUAGE EXTENSIONS  

### B.1. Function Execution Space Specifiers  

​        Function execution space specifiers denote whether a function executes on the host or on the device and whether it is callable from the host or from the device.  

####  B.1.1. \_\_global\_\_  

​       The \_\_global\_\_ exection space specifier declares a function as being a kernel. Such a function is:  

‣  Executed on the device, 

‣  Callable from the host, 

‣  Callable from the device for devices of compute capability 3.2 or higher (see CUDA Dynamic Parallelism for more details).  

​        A \_\_global\_\_ function must have void return type, and cannot be a member of a class.  Any call to a \_\_global\_\_ function must specify its execution configuration as described in Execution Configuration.  A call to a \_\_global\_\_ function is asynchronous, meaning it returns before the device has completed its execution.  

####  B.1.2. \_\_device\_\_ 

​       The \_\_device\_\_ execution space specifier declares a function that is:  ‣  Executed on the device, ‣  Callable from the device only.  The \_\_global\_\_ and \_\_device\_\_ execution space specifiers cannot be used together.   

#### B.1.3. \_\_host\_\_  

​        The \_\_host\_\_ execution space specifier declares a function that is:  

‣  Executed on the host,

 ‣  Callable from the host only.  

​        It is equivalent to declare a function with only the \_\_host\_\_ execution space specifier or to declare it without any of the \_\_host\_\_, \_\_device\_\_, or \_\_global\_\_ execution space specifier; in either case the function is compiled for the host only.  

​        The \_\_global\_\_ and \_\_host\_\_ execution space specifiers cannot be used together. 

​        The \_\_device\_\_ and \_\_host\_\_ execution space specifiers can be used together however, in which case the function is compiled for both the host and the device.

​        The __CUDA_ARCH__ macro introduced in Application Compatibility can be used to  differentiate code paths between host and device:

```c++
 __host__ __device__ func() 
 { 
     #if __CUDA_ARCH__ >= 700 
     // Device code path for compute capability 7.x 
     #elif __CUDA_ARCH__ >= 600 
     // Device code path for compute capability 6.x 
     #elif __CUDA_ARCH__ >= 500 
     // Device code path for compute capability 5.x 
     #elif __CUDA_ARCH__ >= 300 
     // Device code path for compute capability 3.x 
     #elif !defined(__CUDA_ARCH__) 
     // Host code path #endif 
 } 
```

####  B.1.4. \_\_noinline\_\_and \_\_forceinline\_\_  

​        The compiler inlines any \_\_device\_\_ function when deemed appropriate. 

​        The \_\_noinline\_\_function qualifier can be used as a hint for the compiler not to inline the function if possible. 

​        The \_\_forceinline\_\_ function qualifier can be used to force the compiler to inline the function. 

​        The \_\_noinline\_\_and \_\_forceinline\_\_ function qualifiers cannot be used together, and neither function qualifier can be applied to an inline function.  

### B.2. Variable Memory Space Specifiers 

​        Variable memory space specifiers denote the memory location on the device of a variable.  

​        An automatic variable declared in device code without any of the \_\_device\_\_, \_\_shared\_\_ and \_\_constant\_\_ memory space specifiers described in this section generally resides in a register. However in some cases the compiler might choose to place it in local memory, which can have adverse performance consequences as detailed in Device Memory Accesses.  

#### B.2.1. \_\_device\_\_  

​         The \_\_device\_\_ memory space specifier declares a variable that resides on the device.  At most one of the other memory space specifiers defined in the next three sections may be used together with \_\_device\_\_ to further denote which memory space the variable belongs to. If none of them is present, the variable:

‣  Resides in global memory space,

‣  Has the lifetime of the CUDA context in which it is created, 

‣  Has a distinct object per device, 

‣  Is accessible from all the threads within the grid and from the host through the runtime library (cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()).  

#### B.2.2. \_\_constant\_\_  

The \_\_constant\_\_ memory space specifier, optionally used together with \_\_device\_\_, declares a variable that:  ‣  Resides in constant memory space, 

‣  Has the lifetime of the CUDA context in which it is created,

 ‣  Has a distinct object per device, 

‣  Is accessible from all the threads within the grid and from the host through the runtime library (cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()).  

#### B.2.3. \_\_shared\_\_  

​        The \_\_shared\_\_ memory space specifier, optionally used together with \_\_device\_\_, declares a variable that:  

 ‣  Resides in the shared memory space of a thread block, 

‣  Has the lifetime of the block, 

‣  Has a distinct object per block, 

‣  Is only accessible from all the threads within the block, 

‣  Does not have a constant address. 

​       When declaring a variable in shared memory as an external array such as  

```c++
extern __shared__ float shared[]; 
```

​       the size of the array is determined at launch time (see Execution Configuration). All variables declared in this fashion, start at the same address in memory, so that the layout  of the variables in the array must be explicitly managed through offsets. For example, if one wants the equivalent of 

```
short array0[128]; 
float array1[64];
int  array2[256]; 
```

in dynamically allocated shared memory, one could declare and initialize the arrays the following way:

```c++
extern __shared__ float array[];
__device__ void func()   // __device__ or __global__ function
{ 
     short* array0 = (short*)array;
     float* array1 = (float*)&array0[128]; 
     int*  array2 =  (int*)&array1[64]; 
} 
```

​        Note that pointers need to be aligned to the type they point to, so the following code, for example, does not work since array1 is not aligned to 4 bytes. 

```c++
extern __shared__ float array[];
__device__ void func()   // __device__ or __global__ function 
{ 
    short* array0 = (short*)array;
    float* array1 = (float*)&array0[127];
} 
```

 Alignment requirements for the built-in vector types are listed in Table 4.  

####  B.2.4. \_\_managed\_\_  

The \_\_managed\_\_ memory space specifier, optionally used together with \_\_device\_\_, declares a variable that: 

 ‣  Can be referenced from both device and host code, e.g., its address can be taken or it can be read or written directly from a device or host function.

 ‣  Has the lifetime of an application. 

 See **\_\_managed\_\_ Memory Space Specifier** for more details.  

####  B.2.5. \_\_restrict\_\_  

​        nvcc supports restricted pointers via the \_\_restrict\_\_ keyword.  

​        Restricted pointers were introduced in C99 to alleviate the aliasing problem that exists in C-type languages, and which inhibits all kind of optimization from code re-ordering to common sub-expression elimination.  

​         Here is an example subject to the aliasing issue, where use of restricted pointer can help the compiler to reduce the number of instructions: 

```c++
void foo(const float* a, const float* b, float* c) 
{ 
    c[0] = a[0] * b[0]; 
    c[1] = a[0] * b[0]; 
    c[2] = a[0] * b[0] * a[1]; 
    c[3] = a[0] * a[1]; 
    c[4] = a[0] * b[0]; 
    c[5] = b[0]; 
    ... 
} 
```

​         In C-type languages, the pointers a, b, and c may be aliased, so any write through c could modify elements of a or b. This means that to guarantee functional correctness, the compiler cannot load a[0] and b[0] into registers, multiply them, and store the result to both c[0] and c[1], because the results would differ from the abstract execution model if, say, a[0] is really the same location as c[0]. So the compiler cannot take advantage of the common sub-expression. Likewise, the compiler cannot just reorder the computation of c[4] into the proximity of the computation of c[0] and c[1] because the preceding write to c[3] could change the inputs to the computation of c[4].  

​         By making a, b, and c restricted pointers, the programmer asserts to the compiler that the pointers are in fact not aliased, which in this case means writes through c would never overwrite elements of a or b. This changes the function prototype as follows:  

```c++
void foo(const float* __restrict__ a, 
         const float* __restrict__ b, 
         float* __restrict__ c);  
```

​        Note that all pointer arguments need to be made restricted for the compiler optimizer to derive any benefit. With the \_\_restrict\_\_ keywords added, the compiler can now reorder and do common sub-expression elimination at will, while retaining functionality identical with the abstract execution model: 

```c++
void foo(const float* __restrict__ a,
const float* __restrict__ b, 
float* __restrict__ c) 
{
     float t0 = a[0]; 
     float t1 = b[0];
     float t2 = t0 * t2;
     float t3 = a[1];
     c[0] = t2;
     c[1] = t2; 
     c[4] = t2;
     c[2] = t2 * t3; 
     c[3] = t0 * t3;
     c[5] = t1;
     ... 
}  
```

​        The effects here are a reduced number of memory accesses and reduced number of computations. This is balanced by an increase in register pressure due to "cached" loads and common sub-expressions.  

​        Since register pressure is a critical issue in many CUDA codes, use of restricted pointers can have negative performance impact on CUDA code, due to reduced occupancy.  

### B.3. Built-in Vector Types 

#### B.3.1. char, short, int, long, longlong, float, double 

​        These are vector types derived from the basic integer and floating-point types. They are structures and the 1st, 2nd, 3rd, and 4th components are accessible through the fields x, y, z, and w, respectively. They all come with a constructor function of the form **make_\<type name>**; for example, 

```c++
 int2 make_int2(int x, int y);  
```

which creates a vector of type int2 with value (x, y).  

The alignment requirements of the vector types are detailed in Table 4. 

 Table 4  Alignment Requirements  

Type            Alignment 

 char1, uchar1        1 

 char2, uchar2        2  

char3, uchar3        1  

char4, uchar4        4  

short1, ushort1       2  

short2, ushort2       4 

 short3, ushort3       2  

short4, ushort4       8 

 int1, uint1         4 

 int2, uint2         8  

int3, uint3         4 

 int4, uint4         16 

 long1, ulong1        4 if sizeof(long) is equal to sizeof(int) 8, otherwise 

 long2, ulong2        8 if sizeof(long) is equal to sizeof(int), 16, otherwise 

 long3, ulong3        4 if sizeof(long) is equal to sizeof(int), 8, otherwise 

 long4, ulong4        16  

  Type           Alignment  

longlong1, ulonglong1   8  

longlong2, ulonglong2   16  

longlong3, ulonglong3   8 

 longlong4, ulonglong4   16  

float1           4  

float2           8  

float3           4  

float4          16  

double1          8 

 double2          16  

double3          8  

double4          16  

####  B.3.2. dim3 

​        This type is an integer vector type based on uint3 that is used to specify dimensions. When defining a variable of type dim3, any component left unspecified is initialized to 1. 

### B.4. Built-in Variables 

​        Built-in variables specify the grid and block dimensions and the block and thread indices. They are only valid within functions that are executed on the device. 

#### B.4.1. gridDim 

​        This variable is of type dim3 (see dim3) and contains the dimensions of the grid.  

#### B.4.2. blockIdx

​        This variable is of type uint3 (see char, short, int, long, longlong, float, double) and contains the block index within the grid.  

#### B.4.3. blockDim

​      This variable is of type dim3 (see dim3) and contains the dimensions of the block.  

#### B.4.4. threadIdx  

​     This variable is of type uint3 (see char, short, int, long, longlong, float, double ) and contains the thread index within the block. 

#### B.4.5. warpSize 

​        This variable is of type int and contains the warp size in threads (see SIMT Architecture for the definition of a warp).  

### B.5. Memory Fence Functions 

​        The CUDA programming model assumes a device with a weakly-ordered memory model, that is the order in which a CUDA thread writes data to shared memory, global memory, page-locked host memory, or the memory of a peer device is not necessarily the order in which the data is observed being written by another CUDA or host thread.  For example, if thread 1 executes writeXY() and thread 2 executes readXY() as defined in the following code sample 

```c++
 __device__ volatile int X = 1, Y = 2;
 __device__ void writeXY() 
 { 
     X = 10; 
     Y = 20;
 }  
 __device__ void readXY() 
 { 
     int A = X;
     int B = Y; 
 }  
```

​        it is possible that B ends up equal to 20 and A equal to 1 for thread 2. In a strongly- ordered memory model, the only possibilities would be: 

 ‣  A equal to 1 and B equal to 2,

 ‣  A equal to 10 and B equal to 2, 

‣  A equal to 10 and B equal to 20,  

​       Memory fence functions can be used to enforce some ordering on memory accesses. The memory fence functions differ in the scope in which the orderings are enforced but they are independent of the accessed memory space (shared memory, global memory, page- locked host memory, and the memory of a peer device). 

```c++
 void __threadfence_block();  
```

ensures that:  

‣  All writes to all memory made by the calling thread before the call to **\_\_threadfence_block()** are observed by all threads in the block of the calling thread as occurring before all writes to all memory made by the calling thread after the call to **\_\_threadfence_block**(); 

‣  All reads from all memory made by the calling thread before the call to **\_\_threadfence_block**() are ordered before all reads from all memory made by the calling thread after the call to \_\_threadfence_block().  

```c++
void __threadfence();  
```

acts as \_\_threadfence_block() for all threads in the block of the calling thread and also ensures that no writes to all memory made by the calling thread after the call to \_\_threadfence() are observed by any thread in the device as occurring before any write to all memory made by the calling thread before the call to \_\_threadfence(). Note that for this ordering guarantee to be true, the observing threads must truly observe the memory and not cached versions of it; this is ensured by using the volatile keyword as detailed in Volatile Qualifier.  

```c++
void __threadfence_system();  
```

acts as \_\_threadfence_block() for all threads in the block of the calling thread and also ensures that all writes to all memory made by the calling thread before the call to \_\_threadfence_system() are observed by all threads in the device, host threads, and all threads in peer devices as occurring before all writes to all memory made by the calling thread after the call to \_\_threadfence_system(). 

​     \_\_threadfence_system() is only supported by devices of compute capability 2.x and higher.  

​     In the previous code sample, inserting a fence function call between X = 10; and Y = 20; and between int A = X; and int B = Y; would ensure that for thread 2, A will always be equal to 10 if B is equal to 20. If thread 1 and 2 belong to the same block, it is enough to use \_\_threadfence_block(). If thread 1 and 2 do not belong to the same block, \_\_threadfence() must be used if they are CUDA threads from the same device and __threadfence_system() must be used if they are CUDA threads from two different devices. 

​     A common use case is when threads consume some data produced by other threads as illustrated by the following code sample of a kernel that computes the sum of an array of N numbers in one call. Each block first sums a subset of the array and stores the result in global memory. When all blocks are done, the last block done reads each of these partial sums from global memory and sums them to obtain the final result. In order to determine which block is finished last, each block atomically increments a counter to signal that it is done with computing and storing its partial sum (see Atomic Functions about atomic functions). The last block is the one that receives the counter value equal to gridDim.x-1. If no fence is placed between storing the partial sum and incrementing the counter, the counter might increment before the partial sum is stored and therefore,  might reach gridDim.x-1 and let the last block start reading partial sums before they have been actually updated in memory.

  Memory fence functions only affect the ordering of memory operations by a thread; they do not ensure that these memory operations are visible to other threads (like \_\_syncthreads() does for threads within a block (see Synchronization Functions)). In the code sample below, the visibility of memory operations on the result variable is ensured by declaring it as volatile (see Volatile Qualifier).  

```c++
__device__ unsigned int count = 0; 
__shared__ bool isLastBlockDone; 
__global__ void sum(const float* array, unsigned int N, volatile float* result) 
{ 
    // Each block sums a subset of the input array. 
    float partialSum = calculatePartialSum(array, N);  
    if (threadIdx.x == 0) {  
        // Thread 0 of each block stores the partial sum 
        // to global memory. The compiler will use 
        // a store operation that bypasses the L1 cache 
        // since the "result" variable is declared as 
        // volatile. This ensures that the threads of 
        // the last block will read the correct partial 
        // sums computed by all other blocks. 
        result[blockIdx .x] = partialSum;  
        // Thread 0 makes sure that the incrementation 
        // of the "count" variable is only performed after 
        // the partial sum has been written to global memory. 
        __threadfence();  
        // Thread 0 signals that it is done. 
        unsigned int value = atomicInc(&count, gridDim.x);   
        // Thread 0 determines if its block is the last 
        // block to be done. 
        isLastBlockDone = (value == (gridDim.x - 1)); 
    }  
    // Synchronize to make sure that each thread reads 
    // the correct value of isLastBlockDone. 
    \_\_syncthreads();  
    if (isLastBlockDone) {  
        // The last block sums the partial sums 
        // stored in result[0 .. gridDim.x-1] 
        float totalSum = calculateTotalSum(result);  
        if (threadIdx.x == 0) {  
            // Thread 0 of last block stores the total sum 
            // to global memory and resets the count
            // varialble, so that the next kernel call 
            // works properly. 
            result[0] = totalSum; 
            count = 0; 
        } 
    } 
} 
```



### B.6. Synchronization Functions  

```c++
void __syncthreads(); 
```

waits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to \_\_syncthreads() are visible to all threads in the block.  

​       **\_\_syncthreads**() is used to coordinate communication between the threads of the same block. When some threads within a block access the same addresses in shared or global memory, there are potential read-after-write, write-after-read, or write-after- write hazards for some of these memory accesses. These data hazards can be avoided by synchronizing threads in-between these accesses. 

​       **\_\_syncthreads**() is allowed in conditional code but only if the conditional evaluates identically across the entire thread block, otherwise the code execution is likely to hang or produce unintended side effects.  

​       Devices of compute capability 2.x and higher support three variations of \_\_syncthreads() described below. 

```c++
 int __syncthreads_count(int predicate); 
```

 is identical to **\_\_syncthreads**() with the additional feature that it evaluates predicate for all threads of the block and returns the number of threads for which predicate evaluates to non-zero.  

```c++
int __syncthreads_and(int predicate); 
```

is identical to **\_\_syncthreads**() with the additional feature that it evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non- zero for all of them. 

```c++
 int __syncthreads_or(int predicate);
```

__ is identical to \_\_syncthreads() with the additional feature that it evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non- zero for any of them. 

```c++
void __syncwarp(unsigned mask=0xffffffff);
```

will cause the executing thread to wait until all warp lanes named in mask have executed a \_\_syncwarp() (with the same mask) before resuming execution. All non- exited threads named in mask must execute a corresponding \_\_syncwarp() with the same mask, or the result is undefined. 

 Executing \_\_syncwarp() guarantees memory ordering among threads participating in the barrier. Thus, threads within a warp that wish to communicate via memory can store to memory, execute \_\_syncwarp(), and then safely read values stored by other threads in the warp. 

```
 For .target sm_6x or below, all threads in mask must execute the same __syncwarp() in convergence, and the union of all values in mask must be equal to the active mask. Otherwise, the behavior is undefined.  
```

### B.7. Mathematical Functions 

​        The reference manual lists all C/C++ standard library mathematical functions that are supported in device code and all intrinsic functions that are only supported in device code.  

​        Mathematical Functions provides accuracy information for some of these functions when relevant.  

### B.8. Texture Functions 

 Texture objects are described in Texture Object API  

Texture references are described in Texture Reference API  

Texture fetching is described in Texture Fetching.  

### B.8.1. Texture Object API 

####  B.8.1.1. tex1Dfetch()  

```c++
template<class T> 
T tex1Dfetch(cudaTextureObject_t texObj, int x); 
```

 fetches from the region of linear memory specified by the one-dimensional texture object texObj using integer texture coordinate x. tex1Dfetch() only works with non- normalized coordinates, so only the border and clamp addressing modes are supported. It does not perform any texture filtering. For integer types, it may optionally promote the integer to single-precision floating point.  

B.8.1.2. tex1D() 

 template<class T> T tex1D(cudaTextureObject_t texObj, float x); 

 fetches from the CUDA array specified by the one-dimensional texture object texObj using texture coordinate x.  

B.8.1.3. tex1DLod() 

 template<class T> T tex1DLod(cudaTextureObject_t texObj, float x, float level); 

 fetches from the CUDA array specified by the one-dimensional texture object texObj using texture coordinate x at the level-of-detail level. 

 B.8.1.4. tex1DGrad() 

 template<class T> T tex1DGrad(cudaTextureObject_t texObj, float x, float dx, float dy); 

 fetches from the CUDA array specified by the one-dimensional texture object texObj using texture coordinate x. The level-of-detail is derived from the X-gradient dx and Y- gradient dy.

  B.8.1.5. tex2D()

  template<class T> T tex2D(cudaTextureObject_t texObj, float x, float y);

  fetches from the CUDA array or the region of linear memory specified by the two- dimensional texture object texObj using texture coordinate (x,y). 

 B.8.1.6. tex2DLod()  

template<class T> tex2DLod(cudaTextureObject_t texObj, float x, float y, float level);  

fetches from the CUDA array or the region of linear memory specified by the two- dimensional texture object texObj using texture coordinate (x,y) at level-of-detail level.  

B.8.1.7. tex2DGrad() 

 template<class T> T tex2DGrad(cudaTextureObject_t texObj, float x, float y, float2 dx, float2 dy);  

fetches from the CUDA array specified by the two-dimensional texture object texObj using texture coordinate (x,y). The level-of-detail is derived from the dx and dy gradients.  

B.8.1.8. tex3D()  

template<class T> T tex3D(cudaTextureObject_t texObj, float x, float y, float z); 

 fetches from the CUDA array specified by the three-dimensional texture object texObj using texture coordinate (x,y,z). 

 B.8.1.9. tex3DLod()  

template<class T> T tex3DLod(cudaTextureObject_t texObj, float x, float y, float z, float level); 

 fetches from the CUDA array or the region of linear memory specified by the three- dimensional texture object texObj using texture coordinate (x,y,z) at level-of-detail level.  

 B.8.1.10. tex3DGrad()  

template<class T> T tex3DGrad(cudaTextureObject_t texObj, float x, float y, float z, float4 dx, float4 dy);  

fetches from the CUDA array specified by the three-dimensional texture object texObj using texture coordinate (x,y,z) at a level-of-detail derived from the X and Y gradients dx and dy. 

 B.8.1.11. tex1DLayered() 

 template<class T> T tex1DLayered(cudaTextureObject_t texObj, float x, int layer);  

fetches from the CUDA array specified by the one-dimensional texture object texObj  using texture coordinate x and index layer, as described in Layered Textures  

B.8.1.12. tex1DLayeredLod() 

 template<class T> T tex1DLayeredLod(cudaTextureObject_t texObj, float x, int layer, float level); 

 fetches from the CUDA array specified by the one-dimensional layered texture at layer layer using texture coordinate x and level-of-detail level.  

B.8.1.13. tex1DLayeredGrad()  

template<class T> T tex1DLayeredGrad(cudaTextureObject_t texObj, float x, int layer, float dx, float dy);  

fetches from the CUDA array specified by the one-dimensional layered texture at layer layer using texture coordinate x and a level-of-detail derived from the dx and dy gradients.  

B.8.1.14. tex2DLayered()  template<class T> T tex2DLayered(cudaTextureObject_t texObj, float x, float y, int layer);  fetches from the CUDA array specified by the two-dimensional texture object texObj using texture coordinate (x,y) and index layer, as described in Layered Textures. 

 B.8.1.15. tex2DLayeredLod()  template<class T> T tex2DLayeredLod(cudaTextureObject_t texObj, float x, float y, int layer, float level); 

 fetches from the CUDA array specified by the two-dimensional layered texture at layer layer using texture coordinate (x,y). 

B.8.1.16. tex2DLayeredGrad() 

 template<class T> T tex2DLayeredGrad(cudaTextureObject_t texObj, float x, float y, int layer, float2 dx, float2 dy); 

 fetches from the CUDA array specified by the two-dimensional layered texture at layer layer using texture coordinate (x,y) and a level-of-detail derived from the dx and dy X and Y gradients.  

B.8.1.17. texCubemap() 

 template<class T> T texCubemap(cudaTextureObject_t texObj, float x, float y, float z);  fetches the CUDA array specified by the three-dimensional texture object texObj using texture coordinate (x,y,z), as described in Cubemap Textures. 

 B.8.1.18. texCubemapLod()

  template<class T> T texCubemapLod(cudaTextureObject_t texObj, float x, float, y, float z, float level);  fetches from the CUDA array specified by the three-dimensional texture object texObj  using texture coordinate (x,y,z) as described in Cubemap Textures. The level-of-detail used is given by level.  

B.8.1.19. texCubemapLayered()

  template<class T> T texCubemapLayered(cudaTextureObject_t texObj, float x, float y, float z, int layer);  fetches from the CUDA array specified by the cubemap layered texture object texObj using texture coordinates (x,y,z), and index layer, as described in Cubemap Layered Textures.  

B.8.1.20. texCubemapLayeredLod() 

 template<class T> T texCubemapLayeredLod(cudaTextureObject_t texObj, float x, float y, float z, int layer, float level);  fetches from the CUDA array specified by the cubemap layered texture object texObj using texture coordinate (x,y,z) and index layer, as described in Cubemap Layered Textures, at level-of-detail level level.  

B.8.1.21. tex2Dgather() 

 template<class T> T tex2Dgather(cudaTextureObject_t texObj, float x, float y, int comp = 0);  

  fetches from the CUDA array specified by the 2D texture object texObj using texture coordinates x and y and the comp parameter as described in Texture Gather.

  B.8.2. Texture Reference API 

 B.8.2.1. tex1Dfetch() 

 template<class DataType> Type tex1Dfetch( texture<DataType, cudaTextureType1D, cudaReadModeElementType> texRef, int x); 

 float tex1Dfetch( texture<unsigned char, cudaTextureType1D, cudaReadModeNormalizedFloat> texRef, int x);  float tex1Dfetch( texture<signed char, cudaTextureType1D, cudaReadModeNormalizedFloat> texRef, int x);  float tex1Dfetch( texture<unsigned short, cudaTextureType1D, cudaReadModeNormalizedFloat> texRef, int x);  float tex1Dfetch( texture<signed short, cudaTextureType1D, cudaReadModeNormalizedFloat> texRef, int x);  fetches from the region of linear memory bound to the one-dimensional texture reference texRef using integer texture coordinate x. tex1Dfetch() only works with non-normalized coordinates, so only the border and clamp addressing modes are supported. It does not perform any texture filtering. For integer types, it may optionally promote the integer to single-precision floating point.  Besides the functions shown above, 2-, and 4-tuples are supported; for example:  float4 tex1Dfetch( texture<uchar4, cudaTextureType1D, cudaReadModeNormalizedFloat> texRef, int x);  fetches from the region of linear memory bound to texture reference texRef using texture coordinate x.  B.8.2.2. tex1D()  template<class DataType, enum cudaTextureReadMode readMode> Type tex1D(texture<DataType, cudaTextureType1D, readMode> texRef, float x);  fetches from the CUDA array bound to the one-dimensional texture reference texRef using texture coordinate x. Type is equal to DataType except when readMode is equal to cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is equal to the matching floating-point type.  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 141  ----------------------- Page 158----------------------- C++ Language Extensions  B.8.2.3. tex1DLod()  template<class DataType, enum cudaTextureReadMode readMode> Type tex1DLod(texture<DataType, cudaTextureType1D, readMode> texRef, float x, float level);  fetches from the CUDA array bound to the one-dimensional texture reference texRef using texture coordinate x. The level-of-detail is given by level. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.4. tex1DGrad()  template<class DataType, enum cudaTextureReadMode readMode> Type tex1DGrad(texture<DataType, cudaTextureType1D, readMode> texRef, float x, float dx, float dy);  fetches from the CUDA array bound to the one-dimensional texture reference texRef using texture coordinate x. The level-of-detail is derived from the dx and dy X- and Y-gradients. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.5. tex2D()  template<class DataType, enum cudaTextureReadMode readMode> Type tex2D(texture<DataType, cudaTextureType2D, readMode> texRef, float x, float y);  fetches from the CUDA array or the region of linear memory bound to the two- dimensional texture reference texRef using texture coordinates x and y. Type is equal to DataType except when readMode is equal to cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is equal to the matching floating-point type.  B.8.2.6. tex2DLod()  template<class DataType, enum cudaTextureReadMode readMode> Type tex2DLod(texture<DataType, cudaTextureType2D, readMode> texRef, float x, float y, float level);  fetches from the CUDA array bound to the two-dimensional texture reference texRef using texture coordinate (x,y). The level-of-detail is given by level. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.7. tex2DGrad()  template<class DataType, enum cudaTextureReadMode readMode> Type tex2DGrad(texture<DataType, cudaTextureType2D, readMode> texRef, float x, float y, float2 dx, float2 dy);  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 142  ----------------------- Page 159----------------------- C++ Language Extensions  fetches from the CUDA array bound to the two-dimensional texture reference texRef using texture coordinate (x,y). The level-of-detail is derived from the dx and dy X- and Y-gradients. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.8. tex3D()  template<class DataType, enum cudaTextureReadMode readMode> Type tex3D(texture<DataType, cudaTextureType3D, readMode> texRef, float x, float y, float z);  fetches from the CUDA array bound to the three-dimensional texture reference texRef using texture coordinates x, y, and z. Type is equal to DataType except when readMode is equal to cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is equal to the matching floating-point type.  B.8.2.9. tex3DLod()  template<class DataType, enum cudaTextureReadMode readMode> Type tex3DLod(texture<DataType, cudaTextureType3D, readMode> texRef, float x, float y, float z, float level);  fetches from the CUDA array bound to the two-dimensional texture reference texRef using texture coordinate (x,y,z). The level-of-detail is given by level. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.10. tex3DGrad()  template<class DataType, enum cudaTextureReadMode readMode> Type tex3DGrad(texture<DataType, cudaTextureType3D, readMode> texRef, float x, float y, float z, float4 dx, float4 dy);  fetches from the CUDA array bound to the two-dimensional texture reference texRef using texture coordinate (x,y,z). The level-of-detail is derived from the dx and dy X- and Y-gradients. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.11. tex1DLayered()  template<class DataType, enum cudaTextureReadMode readMode> Type tex1DLayered( texture<DataType, cudaTextureType1DLayered, readMode> texRef, float x, int layer);  fetches from the CUDA array bound to the one-dimensional layered texture reference texRef using texture coordinate x and index layer, as described in Layered Textures. Type is equal to DataType except when readMode is equal to  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 143  ----------------------- Page 160----------------------- C++ Language Extensions  cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is equal to the matching floating-point type.  B.8.2.12. tex1DLayeredLod()  template<class DataType, enum cudaTextureReadMode readMode> Type tex1DLayeredLod(texture<DataType, cudaTextureType1D, readMode> texRef, float x, int layer, float level);  fetches from the CUDA array bound to the one-dimensional texture reference texRef using texture coordinate x and index layer as described in Layered Textures. The level- of-detail is given by level. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.13. tex1DLayeredGrad()  template<class DataType, enum cudaTextureReadMode readMode> Type tex1DLayeredGrad(texture<DataType, cudaTextureType1D, readMode> texRef, float x, int layer, float dx, float dy);  fetches from the CUDA array bound to the one-dimensional texture reference texRef using texture coordinate x and index layer as described in Layered Textures. The level-of-detail is derived from the dx and dy X- and Y-gradients. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.14. tex2DLayered()  template<class DataType, enum cudaTextureReadMode readMode> Type tex2DLayered( texture<DataType, cudaTextureType2DLayered, readMode> texRef, float x, float y, int layer);  fetches from the CUDA array bound to the two-dimensional layered texture reference texRef using texture coordinates x and y, and index layer, as described in Texture Memory. Type is equal to DataType except when readMode is equal to cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is equal to the matching floating-point type.  B.8.2.15. tex2DLayeredLod()  template<class DataType, enum cudaTextureReadMode readMode> Type tex2DLayeredLod(texture<DataType, cudaTextureType2D, readMode> texRef, float x, float y, int layer, float level);  fetches from the CUDA array bound to the two-dimensional texture reference texRef using texture coordinate (x,y) and index layer as described in Layered Textures. The level-of-detail is given by level. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 144  ----------------------- Page 161----------------------- C++ Language Extensions  B.8.2.16. tex2DLayeredGrad()  template<class DataType, enum cudaTextureReadMode readMode> Type tex2DLayeredGrad(texture<DataType, cudaTextureType2D, readMode> texRef, float x, float y, int layer, float2 dx, float2 dy);  fetches from the CUDA array bound to the two-dimensional texture reference texRef using texture coordinate (x,y) and index layer as described in Layered Textures. The level-of-detail is derived from the dx and dy X- and Y-gradients. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.17. texCubemap()  template<class DataType, enum cudaTextureReadMode readMode> Type texCubemap( texture<DataType, cudaTextureTypeCubemap, readMode> texRef, float x, float y, float z);  fetches from the CUDA array bound to the cubemap texture reference texRef using texture coordinates x, y, and z, as described in Cubemap Textures. Type is equal to DataType except when readMode is equal to cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is equal to the matching floating-point type.  B.8.2.18. texCubemapLod()  template<class DataType, enum cudaTextureReadMode readMode> Type texCubemapLod(texture<DataType, cudaTextureType3D, readMode> texRef, float x, float y, float z, float level);  fetches from the CUDA array bound to the two-dimensional texture reference texRef using texture coordinate (x,y,z). The level-of-detail is given by level. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.19. texCubemapLayered()  template<class DataType, enum cudaTextureReadMode readMode> Type texCubemapLayered( texture<DataType, cudaTextureTypeCubemapLayered, readMode> texRef, float x, float y, float z, int layer);  fetches from the CUDA array bound to the cubemap layered texture reference texRef using texture coordinates x, y, and z, and index layer, as described in Cubemap Layered Textures. Type is equal to DataType except when readMode is equal to cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is equal to the matching floating-point type.  B.8.2.20. texCubemapLayeredLod()  template<class DataType, enum cudaTextureReadMode readMode> Type texCubemapLayeredLod(texture<DataType, cudaTextureType3D, readMode> texRef, float x, float y, float z, int layer, float level);  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 145  ----------------------- Page 162----------------------- C++ Language Extensions  fetches from the CUDA array bound to the two-dimensional texture reference texRef using texture coordinate (x,y,z) and index layer as described in Layered Textures. The level-of-detail is given by level. Type is the same as DataType except when readMode is cudaReadModeNormalizedFloat (see Texture Reference API), in which case Type is the corresponding floating-point type.  B.8.2.21. tex2Dgather()  template<class DataType, enum cudaTextureReadMode readMode> Type tex2Dgather( texture<DataType, cudaTextureType2D, readMode> texRef, float x, float y, int comp = 0);  fetches from the CUDA array bound to the 2D texture reference texRef using texture coordinates x and y and the comp parameter as described in Texture Gather. Type is a 4- component vector type. It is based on the base type of DataType except when readMode is equal to cudaReadModeNormalizedFloat (see Texture Reference API), in which case it is always float4.  B.9. Surface Functions  Surface functions are only supported by devices of compute capability 2.0 and higher.  Surface objects are described in described in Surface Object API  Surface references are described in Surface Reference API.  In the sections below, boundaryMode specifies the boundary mode, that is how out-of- range surface coordinates are handled; it is equal to either cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range, or cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are ignored, or cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to fail.  B.9.1. Surface Object API  B.9.1.1. surf1Dread()  template<class T> T surf1Dread(cudaSurfaceObject_t surfObj, int x, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array specified by the one-dimensional surface object surfObj using coordinate x.  B.9.1.2. surf1Dwrite  template<class T> void surf1Dwrite(T data, cudaSurfaceObject_t surfObj, int x, boundaryMode = cudaBoundaryModeTrap);  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 146  ----------------------- Page 163----------------------- C++ Language Extensions  writes value data to the CUDA array specified by the one-dimensional surface object surfObj at coordinate x.  B.9.1.3. surf2Dread()  template<class T> T surf2Dread(cudaSurfaceObject_t surfObj, int x, int y, boundaryMode = cudaBoundaryModeTrap); template<class T> void surf2Dread(T* data, cudaSurfaceObject_t surfObj, int x, int y, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array specified by the two-dimensional surface object surfObj using coordinates x and y.  B.9.1.4. surf2Dwrite()  template<class T> void surf2Dwrite(T data, cudaSurfaceObject_t surfObj, int x, int y, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array specified by the two-dimensional surface object surfObj at coordinate x and y.  B.9.1.5. surf3Dread()  template<class T> T surf3Dread(cudaSurfaceObject_t surfObj, int x, int y, int z, boundaryMode = cudaBoundaryModeTrap); template<class T> void surf3Dread(T* data, cudaSurfaceObject_t surfObj, int x, int y, int z, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array specified by the three-dimensional surface object surfObj using coordinates x, y, and z.  B.9.1.6. surf3Dwrite()  template<class T> void surf3Dwrite(T data, cudaSurfaceObject_t surfObj, int x, int y, int z, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array specified by the three-dimensional object surfObj at coordinate x, y, and z.  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 147  ----------------------- Page 164----------------------- C++ Language Extensions  B.9.1.7. surf1DLayeredread()  template<class T> T surf1DLayeredread( cudaSurfaceObject_t surfObj, int x, int layer, boundaryMode = cudaBoundaryModeTrap); template<class T> void surf1DLayeredread(T data, cudaSurfaceObject_t surfObj, int x, int layer, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array specified by the one-dimensional layered surface object surfObj using coordinate x and index layer.  B.9.1.8. surf1DLayeredwrite()  template<class Type> void surf1DLayeredwrite(T data, cudaSurfaceObject_t surfObj, int x, int layer, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array specified by the two-dimensional layered surface object surfObj at coordinate x and index layer.  B.9.1.9. surf2DLayeredread()  template<class T> T surf2DLayeredread( cudaSurfaceObject_t surfObj, int x, int y, int layer, boundaryMode = cudaBoundaryModeTrap); template<class T> void surf2DLayeredread(T data, cudaSurfaceObject_t surfObj, int x, int y, int layer, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array specified by the two-dimensional layered surface object surfObj using coordinate x and y, and index layer.  B.9.1.10. surf2DLayeredwrite()  template<class T> void surf2DLayeredwrite(T data, cudaSurfaceObject_t surfObj, int x, int y, int layer, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array specified by the one-dimensional layered surface object surfObj at coordinate x and y, and index layer.  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 148  ----------------------- Page 165----------------------- C++ Language Extensions  B.9.1.11. surfCubemapread()  template<class T> T surfCubemapread( cudaSurfaceObject_t surfObj, int x, int y, int face, boundaryMode = cudaBoundaryModeTrap); template<class T> void surfCubemapread(T data, cudaSurfaceObject_t surfObj, int x, int y, int face,  boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array specified by the cubemap surface object surfObj using coordinate x and y, and face index face.  B.9.1.12. surfCubemapwrite()  template<class T> void surfCubemapwrite(T data, cudaSurfaceObject_t surfObj, int x, int y, int face, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array specified by the cubemap object surfObj at coordinate x and y, and face index face.  B.9.1.13. surfCubemapLayeredread()  template<class T> T surfCubemapLayeredread( cudaSurfaceObject_t surfObj, int x, int y, int layerFace, boundaryMode = cudaBoundaryModeTrap); template<class T> void surfCubemapLayeredread(T data, cudaSurfaceObject_t surfObj, int x, int y, int layerFace, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array specified by the cubemap layered surface object surfObj using coordinate x and y, and index layerFace.  B.9.1.14. surfCubemapLayeredwrite()  template<class T> void surfCubemapLayeredwrite(T data, cudaSurfaceObject_t surfObj, int x, int y, int layerFace, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array specified by the cubemap layered object surfObj at coordinate x and y, and index layerFace.  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 149  ----------------------- Page 166----------------------- C++ Language Extensions  B.9.2. Surface Reference API  B.9.2.1. surf1Dread()  template<class Type> Type surf1Dread(surface<void, cudaSurfaceType1D> surfRef, int x, boundaryMode = cudaBoundaryModeTrap); template<class Type> void surf1Dread(Type data, surface<void, cudaSurfaceType1D> surfRef, int x, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array bound to the one-dimensional surface reference surfRef using coordinate x.  B.9.2.2. surf1Dwrite  template<class Type> void surf1Dwrite(Type data, surface<void, cudaSurfaceType1D> surfRef, int x, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array bound to the one-dimensional surface reference surfRef at coordinate x.  B.9.2.3. surf2Dread()  template<class Type> Type surf2Dread(surface<void, cudaSurfaceType2D> surfRef, int x, int y, boundaryMode = cudaBoundaryModeTrap); template<class Type> void surf2Dread(Type* data, surface<void, cudaSurfaceType2D> surfRef, int x, int y, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array bound to the two-dimensional surface reference surfRef using coordinates x and y.  B.9.2.4. surf2Dwrite()  template<class Type> void surf3Dwrite(Type data, surface<void, cudaSurfaceType3D> surfRef, int x, int y, int z, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array bound to the two-dimensional surface reference surfRef at coordinate x and y.  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 150  ----------------------- Page 167----------------------- C++ Language Extensions 



 B.9.2.5. surf3Dread()  

template<class Type> Type surf3Dread(surface<void, cudaSurfaceType3D> surfRef, int x, int y, int z, boundaryMode = cudaBoundaryModeTrap); template<class Type> void surf3Dread(Type* data, surface<void, cudaSurfaceType3D> surfRef, int x, int y, int z, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array bound to the three-dimensional surface reference surfRef using coordinates x, y, and z.

  B.9.2.6. surf3Dwrite() 

 template<class Type> void surf3Dwrite(Type data, surface<void, cudaSurfaceType3D> surfRef, int x, int y, int z, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array bound to the three-dimensional surface reference surfRef at coordinate x, y, and z.  B.9.2.7. surf1DLayeredread()  template<class Type> Type surf1DLayeredread( surface<void, cudaSurfaceType1DLayered> surfRef, int x, int layer, boundaryMode = cudaBoundaryModeTrap); template<class Type> void surf1DLayeredread(Type data, surface<void, cudaSurfaceType1DLayered> surfRef, int x, int layer, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array bound to the one-dimensional layered surface reference surfRef using coordinate x and index layer. 

 B.9.2.8. surf1DLayeredwrite() 

 template<class Type> void surf1DLayeredwrite(Type data, surface<void, cudaSurfaceType1DLayered> surfRef, int x, int layer, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array bound to the two-dimensional layered surface reference surfRef at coordinate x and index layer.



B.9.2.9. surf2DLayeredread()

  template<class Type> Type surf2DLayeredread( surface<void, cudaSurfaceType2DLayered> surfRef, int x, int y, int layer, boundaryMode = cudaBoundaryModeTrap); template<class Type> void surf2DLayeredread(Type data, surface<void, cudaSurfaceType2DLayered> surfRef, int x, int y, int layer, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array bound to the two-dimensional layered surface reference surfRef using coordinate x and y, and index layer.  B.9.2.10. surf2DLayeredwrite()  template<class Type> void surf2DLayeredwrite(Type data, surface<void, cudaSurfaceType2DLayered> surfRef, int x, int y, int layer, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array bound to the one-dimensional layered surface reference surfRef at coordinate x and y, and index layer.  B.9.2.11. surfCubemapread()  template<class Type> Type surfCubemapread( surface<void, cudaSurfaceTypeCubemap> surfRef, int x, int y, int face, boundaryMode = cudaBoundaryModeTrap); template<class Type> void surfCubemapread(Type data, surface<void, cudaSurfaceTypeCubemap> surfRef, int x, int y, int face, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array bound to the cubemap surface reference surfRef using coordinate x and y, and face index face.  

 B.9.2.12. surfCubemapwrite() 

 template<class Type> void surfCubemapwrite(Type data, surface<void, cudaSurfaceTypeCubemap> surfRef, int x, int y, int face, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array bound to the cubemap reference surfRef at coordinate x and y, and face index face. 

 B.9.2.13. surfCubemapLayeredread()  

template<class Type> Type surfCubemapLayeredread( surface<void, cudaSurfaceTypeCubemapLayered> surfRef, int x, int y, int layerFace, boundaryMode = cudaBoundaryModeTrap); template<class Type> void surfCubemapLayeredread(Type data, surface<void, cudaSurfaceTypeCubemapLayered> surfRef, int x, int y, int layerFace, boundaryMode = cudaBoundaryModeTrap);  reads the CUDA array bound to the cubemap layered surface reference surfRef using coordinate x and y, and index layerFace. 

 B.9.2.14. surfCubemapLayeredwrite() 

 template<class Type> void surfCubemapLayeredwrite(Type data, surface<void, cudaSurfaceTypeCubemapLayered> surfRef, int x, int y, int layerFace, boundaryMode = cudaBoundaryModeTrap);  writes value data to the CUDA array bound to the cubemap layered reference surfRef at coordinate x and y, and index layerFace.  B.10. Read-Only Data Cache Load Function  The read-only data cache load function is only supported by devices of compute capability 3.5 and higher.  T __ldg(const T* address);  returns the data of type T located at address address, where T is char, signed char, short, int, long, long long unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long, char2, char4, short2, short4, int2, int4, longlong2 uchar2, uchar4, ushort2, ushort4, uint2, uint4, ulonglong2 float, float2, float4, double, or double2. The operation is cached in the read-only data cache (see Global Memory).  B.11. Time Function  clock_t clock(); long long int clock64();  when executed in device code, returns the value of a per-multiprocessor counter that is incremented every clock cycle. Sampling this counter at the beginning and at the end of a kernel, taking the difference of the two samples, and recording the result per thread provides a measure for each thread of the number of clock cycles taken by the device to completely execute the thread, but not of the number of clock cycles the device actually  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 153  ----------------------- Page 170----------------------- C++ Language Extensions  spent executing thread instructions. The former number is greater than the latter since threads are time sliced.  B.12. Atomic Functions  An atomic function performs a read-modify-write atomic operation on one 32-bit or 64- bit word residing in global or shared memory. For example, atomicAdd() reads a word at some address in global or shared memory, adds a number to it, and writes the result back to the same address. The operation is atomic in the sense that it is guaranteed to be performed without interference from other threads. In other words, no other thread can access this address until the operation is complete. Atomic functions do not act as memory fences and do not imply synchronization or ordering constraints for memory operations (see Memory Fence Functions for more details on memory fences). Atomic functions can only be used in device functions.  On GPU architectures with compute capability lower than 6.x, atomics operations done from the GPU are atomic only with respect to that GPU. If the GPU attempts an atomic operation to a peer GPU’s memory, the operation appears as a regular read followed by a write to the peer GPU, and the two operations are not done as one single atomic operation. Similarly, atomic operations from the GPU to CPU memory will not be atomic with respect to CPU initiated atomic operations.  Compute capability 6.x introduces new type of atomics which allows developers to widen or narrow the scope of an atomic operation. For example, atomicAdd_system guarantees that the instruction is atomic with respect to other CPUs and GPUs in the system. atomicAdd_block implies that the instruction is atomic only with respect atomics from other threads in the same thread block. In the following example both CPU and GPU can atomically update integer value at address addr:  \_\_global\_\_ void mykernel(int *addr) { atomicAdd_system(addr, 10);    // only available on devices with compute capability 6.x }  void foo() { int *addr; cudaMallocManaged(&addr, 4); *addr = 0;  mykernel<<<...>>> (addr);  __sync_fetch_and_add(addr, 10); // CPU atomic operation }  System wide atomics are not supported on Tegra devices with compute capability less than 7.2.  The new scoped versions of atomics are available for all atomics listed below only for compute capabilities 6.x and later.  www.nvidia.com CUDA C++ Programming Guide                           PG-02829-001_v10.2 | 154  ----------------------- Page 171----------------------- C++ Language Extensions  Note that any atomic operation can be implemented based on atomicCAS() (Compare And Swap). For example, atomicAdd() for double-precision floating-point numbers is not available on devices with compute capability lower than 6.0 but it can be implemented as follows:  #if __CUDA_ARCH__ < 600 \_\_device\_\_ double atomicAdd(double* address, double val) { unsigned long long int* address_as_ull = (unsigned long long int*)address; unsigned long long int old = *address_as_ull, assumed;  do { assumed = old; old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);  return __longlong_as_double(old); } #endif  B.12.1. Arithmetic Functions  B.12.1.1. atomicAdd()  int atomicAdd(int* address, int val); unsigned int atomicAdd(unsigned int* address, unsigned int val); unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val); float atomicAdd(float* address, float val); double atomicAdd(double* address, double val); __half2 atomicAdd(__half2 *address, __half2 val); __half atomicAdd(__half *address, __half val);  reads the 16-bit, 32-bit or 64-bit word old located at the address address in global or shared memory, computes (old + val), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old.  The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.  The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.  The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 add operation is guaranteed separately for each of the two __half elements; the entire __half2 is not guaranteed to be atomic as a single 32-bit access.  The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 155  ----------------------- Page 172----------------------- C++ Language Extensions  B.12.1.2. atomicSub()  int atomicSub(int* address, int val); unsigned int atomicSub(unsigned int* address, unsigned int val);  reads the 32-bit word old located at the address address in global or shared memory, computes (old - val), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old. 

 B.12.1.3. atomicExch()  int atomicExch(int* address, int val); unsigned int atomicExch(unsigned int* address, unsigned int val); unsigned long long int atomicExch(unsigned long long int* address, unsigned long long int val); float atomicExch(float* address, float val);  reads the 32-bit or 64-bit word old located at the address address in global or shared memory and stores val back to memory at the same address. These two operations are performed in one atomic transaction. The function returns old.  B.12.1.4. atomicMin()  int atomicMin(int* address, int val); unsigned int atomicMin(unsigned int* address, unsigned int val); unsigned long long int atomicMin(unsigned long long int* address, unsigned long long int val);  reads the 32-bit or 64-bit word old located at the address address in global or shared memory, computes the minimum of old and val, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old.  The 64-bit version of atomicMin() is only supported by devices of compute capability 3.5 and higher.  B.12.1.5. atomicMax()  int atomicMax(int* address, int val); unsigned int atomicMax(unsigned int* address, unsigned int val); unsigned long long int atomicMax(unsigned long long int* address, unsigned long long int val);  reads the 32-bit or 64-bit word old located at the address address in global or shared memory, computes the maximum of old and val, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old.  The 64-bit version of atomicMax() is only supported by devices of compute capability 3.5 and higher.  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 156  ----------------------- Page 173----------------------- C++ Language Extensions  B.12.1.6. atomicInc()  unsigned int atomicInc(unsigned int* address, unsigned int val);  reads the 32-bit word old located at the address address in global or shared memory, computes ((old >= val) ? 0 : (old+1)), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old.  B.12.1.7. atomicDec()  unsigned int atomicDec(unsigned int* address, unsigned int val);  reads the 32-bit word old located at the address address in global or shared memory, computes (((old == 0) || (old > val)) ? val : (old-1) ), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old.  B.12.1.8. atomicCAS()  int atomicCAS(int* address, int compare, int val); unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val); unsigned long long int atomicCAS(unsigned long long int* address, unsigned long long int compare, unsigned long long int val); unsigned short int atomicCAS(unsigned short int *address, unsigned short int compare, unsigned short int val);  reads the 16-bit, 32-bit or 64-bit word old located at the address address in global or shared memory, computes (old == compare ? val : old) , and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old (Compare And Swap).  B.12.2. Bitwise Functions  B.12.2.1. atomicAnd()  int atomicAnd(int* address, int val); unsigned int atomicAnd(unsigned int* address, unsigned int val); unsigned long long int atomicAnd(unsigned long long int* address, unsigned long long int val);  reads the 32-bit or 64-bit word old located at the address address in global or shared memory, computes (old & val), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old.  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 157  ----------------------- Page 174----------------------- C++ Language Extensions  The 64-bit version of atomicAnd() is only supported by devices of compute capability 3.5 and higher.  B.12.2.2. atomicOr()  int atomicOr(int* address, int val); unsigned int atomicOr(unsigned int* address, unsigned int val); unsigned long long int atomicOr(unsigned long long int* address, unsigned long long int val);  reads the 32-bit or 64-bit word old located at the address address in global or shared memory, computes (old | val), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old.  The 64-bit version of atomicOr() is only supported by devices of compute capability 3.5 and higher.  B.12.2.3. atomicXor()  int atomicXor(int* address, int val); unsigned int atomicXor(unsigned int* address, unsigned int val); unsigned long long int atomicXor(unsigned long long int* address, unsigned long long int val);  reads the 32-bit or 64-bit word old located at the address address in global or shared memory, computes (old ^ val), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old.  The 64-bit version of atomicXor() is only supported by devices of compute capability 3.5 and higher.  B.13. Address Space Predicate Functions  B.13.1. __isGlobal()  unsigned int __isGlobal(const void *ptr);  Returns 1 if ptr contains the generic address of an object in global memory space, otherwise returns 0.  B.13.2. __isShared()  unsigned int __isShared(const void *ptr);  Returns 1 if ptr contains the generic address of an object in shared memory space, otherwise returns 0.  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 158  ----------------------- Page 175----------------------- C++ Language Extensions  B.13.3. __isConstant()  unsigned int __isConstant(const void *ptr);  Returns 1 if ptr contains the generic address of an object in constant memory space, otherwise returns 0.  B.13.4. __isLocal()  unsigned int __isLocal(const void *ptr);  Returns 1 if ptr contains the generic address of an object in local memory space,  otherwise returns 0.  B.14. Warp Vote Functions  int __all_sync(unsigned mask, int predicate); int __any_sync(unsigned mask, int predicate); unsigned __ballot_sync(unsigned mask, int predicate); unsigned __activemask();   Deprecation notice: __any, __all, and __ballot have been deprecated in CUDA 9.0 for all devices.  Removal notice: When targeting devices with compute capability 7.x or higher, __any, __all, and __ballot are no longer available and their sync variants should be used instead.  The warp vote functions allow the threads of a given warp to perform a reduction-and- broadcast operation. These functions take as input an integer predicate from each thread in the warp and compare those values with zero. The results of the comparisons are combined (reduced) across the active threads of the warp in one of the following ways, broadcasting a single return value to each participating thread: __all_sync(unsigned mask, predicate): Evaluate predicate for all non-exited threads in mask and return non-zero if and only if predicate evaluates to non-zero for all of them. __any_sync(unsigned mask, predicate): Evaluate predicate for all non-exited threads in mask and return non-zero if and only if predicate evaluates to non-zero for any of them. __ballot_sync(unsigned mask, predicate): Evaluate predicate for all non-exited threads in mask and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. __activemask(): Returns a 32-bit integer mask of all currently active threads in the calling warp. The Nth bit is set if the Nth lane in the warp is active when __activemask() is called. Inactive threads are represented by 0 bits in the returned mask. Threads which have exited the program are always marked as inactive. Note that threads that are convergent at an __activemask() call are not guaranteed to be convergent at  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 159  ----------------------- Page 176----------------------- C++ Language Extensions 

 subsequent instructions unless those instructions are synchronizing warp-builtin functions.  Notes  For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the threads participating in the call. A bit, representing the thread's lane ID, must be set for each participating thread to ensure they are properly converged before the intrinsic is executed by the hardware. All active threads named in mask must execute the same intrinsic with the same mask, or the result is undefined.  B.15. Warp Match Functions  __match_any_sync and __match_all_sync perform a broadcast-and-compare operation of a variable between threads within a warp.  Supported by devices of compute capability 7.x or higher.  B.15.1. Synopsys   unsigned int __match_any_sync(unsigned mask, T value); unsigned int __match_all_sync(unsigned mask, T value, int *pred);    T can be int, unsigned int, long, unsigned long, long long, unsigned long long, float or double. __

__ B.15.2. Description 

 The __match_sync() intrinsics permit a broadcast-and-compare of a value value across threads in a warp after synchronizing threads named in mask.  __match_any_sync Returns mask of threads that have same value of value in mask __match_all_sync Returns mask if all threads in mask have the same value for value; otherwise 0 is returned. Predicate pred is set to true if all threads in mask have the same value of value; otherwise the predicate is set to false.  The new *_sync match intrinsics take in a mask indicating the threads participating in the call. A bit, representing the thread's lane id, must be set for each participating thread to ensure they are properly converged before the intrinsic is executed by the hardware. All non-exited threads named in mask must execute the same intrinsic with the same mask, or the result is undefined. __

__  B.16. Warp Shuffle Functions  

__shfl_sync, __shfl_up_sync, __shfl_down_sync, and __shfl_xor_sync exchange a variable between threads within a warp.  Supported by devices of compute capability 3.x or higher.  Deprecation Notice: __shfl, __shfl_up, __shfl_down, and __shfl_xor have been deprecated in CUDA 9.0 for all devices.  Removal Notice: When targeting devices with compute capability 7.x or higher, __shfl, __shfl_up, __shfl_down, and __shfl_xor are no longer available and their sync variants should be used instead.  B.16.1. Synopsis   T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize); T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize); T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize); T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);   T can be int, unsigned int, long, unsigned long, long long, unsigned long long, float or double. With the cuda_fp16.h header included, T can also be __half or __half2. __

__ B.16.2. Description 

 The __shfl_sync() intrinsics permit exchanging of a variable between threads within a warp without use of shared memory. The exchange occurs simultaneously for all active threads within the warp (and named in mask), moving 4 or 8 bytes of data per thread depending on the type.  Threads within a warp are referred to as lanes, and may have an index between 0 and warpSize-1 (inclusive). Four source-lane addressing modes are supported:  __shfl_sync() Direct copy from indexed lane __shfl_up_sync() Copy from a lane with lower ID relative to caller __shfl_down_sync() Copy from a lane with higher ID relative to caller __shfl_xor_sync() Copy from a lane based on bitwise XOR of own lane ID  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 161  ----------------------- Page 178----------------------- C++ Language Extensions  Threads may only read data from another thread which is actively participating in the __shfl_sync() command. If the target thread is inactive, the retrieved value is undefined.  All of the __shfl_sync() intrinsics take an optional width parameter which alters the behavior of the intrinsic. width must have a value which is a power of 2; results are undefined if width is not a power of 2, or is a number greater than warpSize.  __shfl_sync() returns the value of var held by the thread whose ID is given by srcLane. If width is less than warpSize then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0. If srcLane is outside the range [0:width-1], the value returned corresponds to the value of var held by the srcLane modulo width (i.e. within the same subsection).  __shfl_up_sync() calculates a source lane ID by subtracting delta from the caller's lane ID. The value of var held by the resulting lane ID is returned: in effect, var is shifted up the warp by delta lanes. If width is less than warpSize then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0. The source lane index will not wrap around the value of width, so effectively the lower delta lanes will be unchanged.  __shfl_down_sync() calculates a source lane ID by adding delta to the caller's lane ID. The value of var held by the resulting lane ID is returned: this has the effect of shifting var down the warp by delta lanes. If width is less than warpSize then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0. As for __shfl_up_sync(), the ID number of the source lane will not wrap around the value of width and so the upper delta lanes will remain unchanged.  __shfl_xor_sync() calculates a source line ID by performing a bitwise XOR of the caller's lane ID with laneMask: the value of var held by the resulting lane ID is returned. If width is less than warpSize then each group of width consecutive threads are able to access elements from earlier groups of threads, however if they attempt to access elements from later groups of threads their own value of var will be returned. This mode implements a butterfly addressing pattern such as is used in tree reduction and broadcast.  The new *_sync shfl intrinsics take in a mask indicating the threads participating in the call. A bit, representing the thread's lane id, must be set for each participating thread to ensure they are properly converged before the intrinsic is executed by the hardware. All non-exited threads named in mask must execute the same intrinsic with the same mask, or the result is undefined.  B.16.3. Notes  Threads may only read data from another thread which is actively participating in the __shfl_sync() command. If the target thread is inactive, the retrieved value is undefined.  __*













*__  width must be a power-of-2 (i.e., 2, 4, 8, 16 or 32). Results are unspecified for other values.  B.16.4. Examples  B.16.4.1. Broadcast of a single value across a warp  #include <stdio.h>  \_\_global\_\_ void bcast(int arg) { int laneId = threadIdx.x & 0x1f; int value; if (laneId == 0)    // Note unused variable for value = arg;    // all threads except lane 0 value = __shfl_sync(0xffffffff, value, 0);  // Synchronize all threads in warp, and get "value" from lane 0 if (value != arg) printf("Thread %d failed.\n", threadIdx.x); }  int main() { bcast<<< 1, 32 >>> (1234); cudaDeviceSynchronize();  return 0; }  B.16.4.2. Inclusive plus-scan across sub-partitions of 8 threads  #include <stdio.h>  \_\_global\_\_ void scan4() { int laneId = threadIdx.x & 0x1f; // Seed sample starting value (inverse of lane ID) int value = 31 - laneId;  // Loop to accumulate scan within my partition. // Scan requires log2(n) == 3 steps for 8 threads // It works by an accumulated sum up the warp // by 1, 2, 4, 8 etc. steps. for (int i=1; i<=4; i*=2) { // We do the __shfl_sync unconditionally so that we // can read even from threads which won't do a // sum, and then conditionally assign the result. int n = __shfl_up_sync(0xffffffff, value, i, 8); if ((laneId & 7) >= i) value += n; }  printf("Thread %d final value = %d\n", threadIdx.x, value);  }  int main() { scan4<<< 1, 32 >>> (); cudaDeviceSynchronize();  return 0; }  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 163  ----------------------- Page 180----------------------- C++ Language Extensions  B.16.4.3. Reduction across a warp  #include <stdio.h>  \_\_global\_\_ void warpReduce() { int laneId = threadIdx.x & 0x1f; // Seed starting value as inverse lane ID int value = 31 - laneId;  // Use XOR mode to perform butterfly reduction for (int i=16; i>=1; i/=2) value += __shfl_xor_sync(0xffffffff, value, i, 32);  // "value" now contains the sum across all threads printf("Thread %d final value = %d\n", threadIdx.x, value); }  int main() { warpReduce<<< 1, 32 >>> (); cudaDeviceSynchronize();  return 0; }  B.17. Warp matrix functions  C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form D=A*B+C. These operations are supported on mixed-precision floating point data for devices of compute capability 7.0 or higher. This requires co-operation from all threads in a warp. In addition, these operations are allowed in conditional code only if the condition evaluates identically across the entire warp, otherwise the code execution is likely to hang.  B.17.1. Description  All following functions and types are defined in the namespace nvcuda::wmma. Sub- byte operations are considered preview, i.e. the data structures and APIs for them are subject to change and may not be compatible with future releases. This extra functionality is defined in the nvcuda::wmma::experimental namespace.  template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;  void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm); void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout); void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout); void fill_fragment(fragment<...> &a, const T& v); void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 164  ----------------------- Page 181----------------------- C++ Language Extensions  fragment  An overloaded class containing a section of a matrix distributed across all threads in the warp. The mapping of matrix elements into fragment internal storage is unspecified and subject to change in future architectures.  Only certain combinations of template arguments are allowed. The first template parameter specifies how the fragment will participate in the matrix operation. Acceptable values for Use are:   ‣  matrix_a when the fragment is used as the first multiplicand, A, ‣  matrix_b when the fragment is used as the second multiplicand, B, or ‣  accumulator when the fragment is used as the source or destination accumulators (C or D, respectively).   The m, n and k sizes describe the shape of the warp-wide matrix tiles participating in the multiply-accumulate operation. The dimension of each tile depends on its role. For matrix_a the tile takes dimension m x k; for matrix_b the dimension is k x n, and accumulator tiles are m x n.  The data type, T, may be __half, char, or unsigned char for multiplicands and __half, float, or int for accumulators. As documented in Element Types & Matrix Sizes, limited combinations of accumulator and multiplicand types are supported. The Layout parameter must be specified for matrix_a and matrix_b fragments. row_major or col_major indicate that elements within a matrix row or column are contiguous in memory, respectively. The Layout parameter for an accumulator matrix should retain the default value of void. A row or column layout is specified only when the accumulator is loaded or stored as described below.  load_matrix_sync  Waits until all warp lanes have arrived at load_matrix_sync and then loads the matrix fragment a from memory. mptr must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. ldm describes the stride in elements between consecutive rows (for row major layout) or columns (for column major layout) and must be a multiple of 16 bytes (i.e., 8 __half elements or 4 float elements). If the fragment is an accumulator, the layout argument must be specified as either mem_row_major or mem_col_major. For matrix_a and matrix_b fragments, the layout is inferred from the fragment's layout parameter. The values of mptr, ldm, layout and all template parameters for a must be the same for all threads in the warp. This function must be called by all threads in the warp, or the result is undefined.  store_matrix_sync  Waits until all warp lanes have arrived at store_matrix_sync and then stores the matrix fragment a to memory. mptr must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. ldm describes the stride in elements between  consecutive rows (for row major layout) or columns (for column major layout) and  www.nvidia.com CUDA C++ Programming Guide                           PG-02829-001_v10.2 | 165  ----------------------- Page 182----------------------- C++ Language Extensions  must be a multiple of 16 bytes. The layout of the output matrix must be specified as either mem_row_major or mem_col_major. The values of mptr, ldm, layout and all template parameters for a must be the same for all threads in the warp.  fill_fragment  Fill a matrix fragment with a constant value v. Because the mapping of matrix elements to each fragment is unspecified, this function is ordinarily called by all threads in the warp with a common value for v.  mma_sync  Waits until all warp lanes have arrived at mma_sync, and then performs the warp- synchronous matrix multiply-accumulate operation D=A*B+C. The in-place operation, C=A*B+C, is also supported. The value of satf and template parameters for each matrix fragment must be the same for all threads in the warp. Also, the template parameters m, n and k must match between fragments A, B, C and D. This function must be called by all threads in the warp, or the result is undefined.  Saturation on integer accumulators will clamp the output to the maximum (or minimum) 32-bit signed integer value. Otherwise, if the accumulation would overflow, the value wraps.  For floating point accumulators, the satf (saturate-to-finite value) mode parameter is deprecated. Using it can lead to unexpected results.  For floating point accumulators, if satf (saturate to finite value) mode is true, the following additional numerical properties apply for the destination accumulator:  ‣  If an element result is +Infinity, the corresponding accumulator will contain +MAX_NORM  ‣  If an element result is -Infinity, the corresponding accumulator will contain - MAX_NORM  ‣  If an element result is NaN, the corresponding accumulator will contain +0  Because the map of matrix elements into each thread's fragment is unspecified, individual matrix elements must be accessed from memory (shared or global) after calling store_matrix_sync. In the special case where all threads in the warp will apply an element-wise operation uniformly to all fragment elements, direct element access can be implemented using the following fragment class members.  enum fragment<Use, m, n, k, T, Layout>::num_elements; T fragment<Use, m, n, k, T, Layout>::x[num_elements];  As an example, the following code scales an accumulator matrix tile by half.  wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag; float alpha = 0.5f; // Same value for all threads in warp ... for(int t=0; t<frag.num_elements; t++) frag.x[t] *= alpha;  www.nvidia.com CUDA C++ Programming Guide                           PG-02829-001_v10.2 | 166  ----------------------- Page 183----------------------- C++ Language Extensions  B.17.2. Sub-byte Operations  Sub-byte WMMA operations provide a way to access the low-precision capabilities of Tensor Cores. They are considered a preview feature i.e. the data structures and APIs for them are subject to change and may not be compatible with future releases. This functionality is available via the nvcuda::wmma::experimental namespace:  namespace experimental { namespace precision { struct u4; // 4-bit unsigned struct s4; // 4-bit signed struct b1; // 1-bit } enum bmmaBitOp { bmmaBitOpXOR = 1 }; enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 }; }  For 4 bit precision, the APIs available remain the same, but you must specify experimental::precision::u4 or experimental::precision::s4 as the fragment data type. Since the elements of the fragment are packed together, num_storage_elements will be smaller than num_elements for that fragment. This is true for single bit precision as well, in which case, the mapping from element_type<T> to storage_element_type<T> is as follows:  experimental::precision::u4 -> unsigned (8 elements in 1 storage element) experimental::precision::s4 -> int (8 elements in 1 storage element) experimental::precision::b1 -> unsigned (32 elements in 1 storage element) all other types T -> T  The allowed layouts for sub-byte fragments is always row_major for matrix_a and col_major for matrix_b.  bmma_sync Waits until all warp lanes have executed bmma_sync, and then performs the warp- synchronous bit matrix multiply-accumulate operation D = (A op B) + C, where op consists of a logical operation bmmaBitOp followed by the accumulation defined by bmmaAccumulateOp. The only available op is a bmmaBitOpXOR, a 128- bit XOR of a row in matrix_a with the 128-bit column of matrix_b, followed by a bmmaAccumulateOpPOPC which counts the number of set bits.



  B.17.3. Restrictions  

The special format required by tensor cores may be different for each major and minor device architecture. This is further complicated by threads holding only a fragment (opaque architecture-specific ABI data structure) of the overall matrix, with the developer not allowed to make assumptions on how the individual parameters are mapped to the registers participating in the matrix multiply-accumulate.  Since fragments are architecture-specific, it is unsafe to pass them from function A to function B if the functions have been compiled for different link-compatible architectures and linked together into the same device executable. In this case, the size  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 167  ----------------------- Page 184----------------------- C++ Language Extensions  and layout of the fragment will be specific to one architecture and using WMMA APIs in the other will lead to incorrect results or potentially, corruption.  An example of two link-compatible architectures, where the layout of the fragment differs, is sm_70 and sm_75.  fragA.cu: void foo() { wmma::fragment<...> mat_a; bar(&mat_a); } fragB.cu: void bar(wmma::fragment<...> *mat_a) { // operate on mat_a }   // sm_70 fragment layout $> nvcc -dc -arch=compute_70 -code=sm_70 fragA.cu -o fragA.o // sm_75 fragment layout $> nvcc -dc -arch=compute_75 -code=sm_75 fragB.cu -o fragB.o // Linking the two together $> nvcc -dlink -arch=sm_75 fragA.o fragB.o -o frag.o   This undefined behavior might also be undetectable at compilation time and by tools at runtime, so extra care is needed to make sure the layout of the fragments is consistent. This linking hazard is most likely to appear when linking with a legacy library that is both built for a different link-compatible architecture and expecting to be passed a WMMA fragment.  Note that in the case of weak linkages (for example, a CUDA C++ inline function), the linker may choose any available function definition which may result in implicit passes between compilation units.  To avoid these sorts of problems, the matrix should always be stored out to memory for transit through external interfaces (e.g. wmma::store_matrix_sync(dst, …);) and then it can be safely passed to bar() as a pointer type [e.g. float *dst].  Note that since sm_70 can run on sm_75, the above example sm_75 code can be changed to sm_70 and correctly work on sm_75. However, it is recommended to have sm_75 native code in your application when linking with other sm_75 separately compiled binaries.  

B.17.4. Element Types & Matrix Sizes  Tensor Cores support a variety of element types and matrix sizes. The following table presents the various combinations of matrix_a, matrix_b and accumulator matrix supported:  Matrix A             Matrix B        Accumulator      Matrix Size (m-n-k)  __half               __half          float          16x16x16  __half               __half          float          32x8x16  __half               __half          float          8x32x16  __half               __half          __half          16x16x16  www.nvidia.com CUDA C++ Programming Guide                           PG-02829-001_v10.2 | 168  ----------------------- Page 185----------------------- C++ Language Extensions  Matrix A             Matrix B        Accumulator      Matrix Size (m-n-k)  __half               __half           __half          32x8x16  __half               __half           __half          8x32x16  unsigned char          unsigned char         int           16x16x16  unsigned char          unsigned char         int           32x8x16  unsigned char          unsigned char         int           8x32x16  signed char           signed char          int           16x16x16  signed char           signed char          int           32x8x16  signed char           signed char          int           8x32x16  In addition, Tensor Cores have experimental support for the following sub-byte operations:  Matrix A             Matrix B        Accumulator      Matrix Size (m-n-k)  precision::u4          precision::u4         int           8x8x32  precision::s4          precision::s4         int           8x8x32  precision::b1          precision::b1         int           8x8x128  __



B.17.5. Example 

 The following code implements a 16x16x16 matrix multiplication in a single warp.  #include <mma.h>  using namespace nvcuda;  \_\_global\_\_ void wmma_ker(half *a, half *b, float *c) { // Declare the fragments wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag; wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag; wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;  // Initialize the output to zero wmma::fill_fragment(c_frag, 0.0f);  // Load the inputs wmma::load_matrix_sync(a_frag, a, 16); wmma::load_matrix_sync(b_frag, b, 16);  // Perform the matrix multiplication wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // Store the output wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major); }  www.nvidia.com CUDA C++ Programming Guide                            PG-02829-001_v10.2 | 169  ----------------------- Page 186----------------------- C++ Language Extensions  B.18. Profiler Counter Function  Each multiprocessor has a set of sixteen hardware counters that an application can increment with a single instruction by calling the __prof_trigger() function.  void __prof_trigger(int counter);  increments by one per warp the per-multiprocessor hardware counter of index counter. Counters 8 to 15 are reserved and should not be used by applications.  The value of counters 0, 1, ..., 7 can be obtained via nvprof by nvprof --events prof_trigger_0x where x is 0, 1, ..., 7. All counters are reset before each kernel launch (note that when collecting counters, kernel launches are synchronous as mentioned in Concurrent Execution between Host and Device).  B.19. Assertion  Assertion is only supported by devices of compute capability 2.x and higher. It is not supported on MacOS, regardless of the device, and loading a module that references the assert function on Mac OS will fail.  void assert(int expression);  stops the kernel execution if expression is equal to zero. If the program is run within a debugger, this triggers a breakpoint and the debugger can be used to inspect the current state of the device. Otherwise, each thread for which expression is equal to zero prints a message to stderr after synchronization with the host via cudaDeviceSynchronize(), cudaStreamSynchronize(), or cudaEventSynchronize(). The format of this message is as follows:  <filename>:<line number>:<function>: block: [blockId.x,blockId.x,blockIdx .z], thread: [threadIdx.x,threadIdx.y,threadIdx.z] Assertion `<expression>` failed.  Any subsequent host-side synchronization calls made for the same device will return cudaErrorAssert. No more commands can be sent to this device until cudaDeviceReset() is called to reinitialize the device.  If expression is different from zero, the kernel execution is unaffected.  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 170  ----------------------- Page 187----------------------- C++ Language Extensions  For example, the following program from source file test.cu  #include <assert.h>  \_\_global\_\_ void testAssert(void) { int is_one = 1; int should_be_one = 0;  // This will have no effect assert(is_one);  // This will halt kernel execution assert(should_be_one); }  int main(int argc, char* argv[]) { testAssert<<<1,1>>> (); cudaDeviceSynchronize();  return 0; }  will output:  test.cu:19: void testAssert(): block: [0,0,0], thread: [0,0,0] Assertion `should_be_one` failed.  Assertions are for debugging purposes. They can affect performance and it is therefore recommended to disable them in production code. They can be disabled at compile time by defining the NDEBUG preprocessor macro before including assert.h. Note that expression should not be an expression with side effects (something like (++i > 0), for example), otherwise disabling the assertion will affect the functionality of the code.  B.20. Formatted Output  Formatted output is only supported by devices of compute capability 2.x and higher.  int printf(const char *format[, arg, ...]);  prints formatted output from a kernel to a host-side output stream.  The in-kernel printf() function behaves in a similar way to the standard C-library printf() function, and the user is referred to the host system's manual pages for a complete description of printf() behavior. In essence, the string passed in as format is output to a stream on the host, with substitutions made from the argument list wherever a format specifier is encountered. Supported format specifiers are listed below.  The printf() command is executed as any other device-side function: per-thread, and in the context of the calling thread. From a multi-threaded kernel, this means that a straightforward call to printf() will be executed by every thread, using that thread's data as specified. Multiple versions of the output string will then appear at the host stream, once for each thread which encountered the printf().  It is up to the programmer to limit the output to a single thread if only a single output string is desired (see Examples for an illustrative example).  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 171  ----------------------- Page 188----------------------- C++ Language Extensions  Unlike the C-standard printf(), which returns the number of characters printed, CUDA's printf() returns the number of arguments parsed. If no arguments follow the format string, 0 is returned. If the format string is NULL, -1 is returned. If an internal error occurs, -2 is returned.  B.20.1. Format Specifiers  As for standard printf(), format specifiers take the form: %[flags][width] [.precision][size]type  The following fields are supported (see widely-available documentation for a complete description of all behaviors):  ‣  Flags: '#' ' ' '0' '+' '-' ‣  Width: '*' '0-9' ‣  Precision: '0-9' ‣  Size: 'h' 'l' 'll' ‣  Type: "%cdiouxXpeEfgGaAs"  Note that CUDA's printf()will accept any combination of flag, width, precision, size and type, whether or not overall they form a valid format specifier. In other words, "%hd" will be accepted and printf will expect a double-precision variable in the corresponding location in the argument list.  B.20.2. Limitations  Final formatting of the printf() output takes place on the host system. This means that the format string must be understood by the host-system's compiler and C library. Every effort has been made to ensure that the format specifiers supported by CUDA's printf function form a universal subset from the most common host compilers, but exact behavior will be host-OS-dependent.  As described in Format Specifiers, printf() will accept all combinations of valid flags and types. This is because it cannot determine what will and will not be valid on the host system where the final output is formatted. The effect of this is that output may be undefined if the program emits a format string which contains invalid combinations.  The printf() command can accept at most 32 arguments in addition to the format string. Additional arguments beyond this will be ignored, and the format specifier output as-is.  Owing to the differing size of the long type on 64-bit Windows platforms (four bytes on 64-bit Windows platforms, eight bytes on other 64-bit platforms), a kernel which is compiled on a non-Windows 64-bit machine but then run on a win64 machine will see corrupted output for all format strings which include "%ld". It is recommended that the compilation platform matches the execution platform to ensure safety.  www.nvidia.com CUDA C++ Programming Guide                           PG-02829-001_v10.2 | 172  ----------------------- Page 189----------------------- C++ Language Extensions  The output buffer for printf() is set to a fixed size before kernel launch (see Associated Host-Side API). It is circular and if more output is produced during kernel execution than can fit in the buffer, older output is overwritten. It is flushed only when one of these actions is performed:  ‣  Kernel launch via <<<>>> or cuLaunchKernel() (at the start of the launch, and if the CUDA_LAUNCH_BLOCKING environment variable is set to 1, at the end of the launch as well), ‣  Synchronization via cudaDeviceSynchronize(), cuCtxSynchronize(), cudaStreamSynchronize(), cuStreamSynchronize(), cudaEventSynchronize(), or cuEventSynchronize(), ‣  Memory copies via any blocking version of cudaMemcpy*() or cuMemcpy*(), ‣  Module loading/unloading via cuModuleLoad() or cuModuleUnload(), ‣  Context destruction via cudaDeviceReset() or cuCtxDestroy(). ‣  Prior to executing a stream callback added by cudaStreamAddCallback or cuStreamAddCallback.  Note that the buffer is not flushed automatically when the program exits. The user must call cudaDeviceReset() or cuCtxDestroy() explicitly, as shown in the examples below.  Internally printf() uses a shared data structure and so it is possible that calling printf() might change the order of execution of threads. In particular, a thread which calls printf() might take a longer execution path than one which does not call printf(), and that path length is dependent upon the parameters of the printf(). Note, however, that CUDA makes no guarantees of thread execution order except at explicit \_\_syncthreads() barriers, so it is impossible to tell whether execution order has been modified by printf() or by other scheduling behaviour in the hardware.  B.20.3. Associated Host-Side API  The following API functions get and set the size of the buffer used to transfer the printf() arguments and internal metadata to the host (default is 1 megabyte):  ‣  cudaDeviceGetLimit(size_t* size,cudaLimitPrintfFifoSize) ‣  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_t size)  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 173  ----------------------- Page 190----------------------- C++ Language Extensions__

B.20.4. Examples  The following code sample:  #include <stdio.h>  \_\_global\_\_ void helloCUDA(float f) {  printf("Hello thread %d, f=%f\n", threadIdx.x, f); }  int main() { helloCUDA<<<1, 5>>> (1.2345f); cudaDeviceSynchronize(); return 0; }  will output:  Hello thread 2, f=1.2345 Hello thread 1, f=1.2345 Hello thread 4, f=1.2345 Hello thread 0, f=1.2345 Hello thread 3, f=1.2345  Notice how each thread encounters the printf() command, so there are as many lines of output as there were threads launched in the grid. As expected, global values (i.e., float f) are common between all threads, and local values (i.e., threadIdx.x) are distinct per-thread.  The following code sample:  #include <stdio.h>  \_\_global\_\_ void helloCUDA(float f) { if (threadIdx.x == 0) printf("Hello thread %d, f=%f\n", threadIdx.x, f) ; }  int main() { helloCUDA<<<1, 5>>> (1.2345f); cudaDeviceSynchronize(); return 0; }  will output:  Hello thread 0, f=1.2345  Self-evidently, the if() statement limits which threads will call printf, so that only a single line of output is seen.  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 174  ----------------------- Page 191----------------------- C++ Language Extensions  

B.21. Dynamic Global Memory Allocation and Operations  Dynamic global memory allocation and operations are only supported by devices of compute capability 2.x and higher.  \_\_host\_\_ \_\_device\_\_ void* malloc(size_t size); \_\_device\_\_ void *__nv_aligned_device_malloc(size_t size, size_t align); \_\_host\_\_ \_\_device\_\_ void free(void* ptr);  allocate and free memory dynamically from a fixed-size heap in global memory.  \_\_host\_\_ \_\_device\_\_ void* memcpy(void* dest, const void* src, size_t size);  copy size bytes from the memory location pointed by src to the memory location  pointed by dest.  \_\_host\_\_ \_\_device\_\_ void* memset(void* ptr, int value, size_t size);  set size bytes of memory block pointed by ptr to value (interpreted as an unsigned char).  The CUDA in-kernel malloc() function allocates at least size bytes from the device heap and returns a pointer to the allocated memory or NULL if insufficient memory exists to fulfill the request. The returned pointer is guaranteed to be aligned to a 16-byte boundary.  The CUDA in-kernel __nv_aligned_device_malloc() function allocates at least size bytes from the device heap and returns a pointer to the allocated memory or NULL if insufficient memory exists to fulfill the requested size or alignment. The address of the allocated memory will be a multiple of align. align must be a non-zero power of 2.  The CUDA in-kernel free() function deallocates the memory pointed to by ptr, which must have been returned by a previous call to malloc() or __nv_aligned_device_malloc(). If ptr is NULL, the call to free() is ignored. Repeated calls to free() with the same ptr has undefined behavior.  The memory allocated by a given CUDA thread via malloc() or __nv_aligned_device_malloc() remains allocated for the lifetime of the CUDA context, or until it is explicitly released by a call to free(). It can be used by any other CUDA threads even from subsequent kernel launches. Any CUDA thread may free memory allocated by another thread, but care should be taken to ensure that the same pointer is not freed more than once.  B.21.1. Heap Memory Allocation  The device memory heap has a fixed size that must be specified before any program using malloc(), __nv_aligned_device_malloc() or free() is loaded into the  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 175  ----------------------- Page 192----------------------- C++ Language Extensions  context. A default heap of eight megabytes is allocated if any program uses malloc() or __nv_aligned_device_malloc() without explicitly specifying the heap size.  The following API functions get and set the heap size:  ‣  cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize) ‣  cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)  The heap size granted will be at least size bytes. cuCtxGetLimit()and cudaDeviceGetLimit() return the currently requested heap size.  The actual memory allocation for the heap occurs when a module is loaded into the context, either explicitly via the CUDA driver API (see Module), or implicitly via the CUDA runtime API (see CUDA Runtime). If the memory allocation fails, the module load will generate a CUDA_ERROR_SHARED_OBJECT_INIT_FAILED error.  Heap size cannot be changed once a module load has occurred and it does not resize dynamically according to need.  Memory reserved for the device heap is in addition to memory allocated through host- side CUDA API calls such as cudaMalloc().  __



B.21.2. Interoperability with Host Memory API  Memory allocated via device malloc() or __nv_aligned_device_malloc() cannot be freed using the runtime (i.e., by calling any of the free memory functions from Device Memory).  Similarly, memory allocated via the runtime (i.e., by calling any of the memory allocation functions from Device Memory) cannot be freed via free().  In addition, memory allocated by a call to malloc() or __nv_aligned_device_malloc() in device code cannot be used in any runtime or driver API calls (i.e. cudaMemcpy, cudaMemset, etc).  B.21.3. Examples  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 176  ----------------------- Page 193----------------------- C++ Language Extensions  

B.21.3.1. Per Thread Allocation  The following code sample:  #include <stdlib.h> #include <stdio.h>  \_\_global\_\_ void mallocTest() { size_t size = 123; char* ptr = (char*)malloc(size); memset(ptr, 0, size); printf("Thread %d got pointer: %p\n", threadIdx.x, ptr); free(ptr); }  int main() { // Set a heap size of 128 megabytes. Note that this must // be done before any kernel is launched. cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024); mallocTest<<<1, 5>>> (); cudaDeviceSynchronize(); return 0; }  will output:  Thread 0 got pointer: 00057020 Thread 1 got pointer: 0005708c Thread 2 got pointer: 000570f8 Thread 3 got pointer: 00057164 Thread 4 got pointer: 000571d0  Notice how each thread encounters the malloc() and memset() commands and so receives and initializes its own allocation. (Exact pointer values will vary: these are illustrative.) 

B.21.3.2. Per Thread Block Allocation  #include <stdlib.h>  \_\_global\_\_ void mallocTest() { \_\_shared\_\_ int* data;  // The first thread in the block does the allocation and then // shares the pointer with all other threads through shared memory, // so that access can easily be coalesced. // 64 bytes per thread are allocated. if (threadIdx.x == 0) { size_t size = blockDim .x * 64; data = (int*)malloc(size); } \_\_syncthreads();  // Check for failure if (data == NULL) return;  // Threads index into the memory, ensuring coalescence int* ptr = data; for (int i = 0; i < 64; ++i) ptr[i * blockDim .x + threadIdx.x] = threadIdx.x;  // Ensure all threads complete before freeing \_\_syncthreads();  // Only one thread may free the memory! if (threadIdx.x == 0) free(data); }  int main() { cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024); mallocTest<<<10, 128>>> (); cudaDeviceSynchronize(); return 0; }  www.nvidia.com CUDA C++ Programming Guide                      __



B.21.3.3. Allocation Persisting Between Kernel Launches

  #include <stdlib.h> #include <stdio.h>  #define NUM_BLOCKS 20  \_\_device\_\_ int* dataptr[NUM_BLOCKS]; // Per-block pointer  \_\_global\_\_ void allocmem() { // Only the first thread in the block does the allocation // since we want only one allocation per block. if (threadIdx.x == 0) dataptr[blockIdx .x] = (int*)malloc(blockDim .x * 4); \_\_syncthreads();  // Check for failure if (dataptr[blockIdx .x] == NULL) return;  // Zero the data with all threads in parallel dataptr[blockIdx .x][threadIdx.x] = 0; }  // Simple example: store thread ID into each element \_\_global\_\_ void usemem() { int* ptr = dataptr[blockIdx .x]; if (ptr != NULL) ptr[threadIdx.x] += threadIdx.x; }  // Print the content of the buffer before freeing it \_\_global\_\_ void freemem() { int* ptr = dataptr[blockIdx .x]; if (ptr != NULL) printf("Block %d, Thread %d: final value = %d\n", blockIdx .x, threadIdx.x, ptr[threadIdx.x]);  // Only free from one thread! if (threadIdx.x == 0) free(ptr); }  int main() { cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);  // Allocate memory allocmem<<< NUM_BLOCKS, 10 >>> ();  // Use memory usemem<<< NUM_BLOCKS, 10 >>> (); usemem<<< NUM_BLOCKS, 10 >>> (); usemem<<< NUM_BLOCKS, 10 >>> ();  // Free memory freemem<<< NUM_BLOCKS, 10 >>> ();  cudaDeviceSynchronize();  return 0; }  www.nvidia.com CUDA C++ Programming Guide                         PG-02829-001_v10.2 | 179  ----------------------- Page 196----------------------- C++ Language Extensions __

B.22. Execution Configuration  

Any call to a \_\_global\_\_ function must specify the execution configuration for that call. The execution configuration defines the dimension of the grid and blocks that will be used to execute the function on the device, as well as the associated stream (see CUDA Runtime for a description of streams).  The execution configuration is specified by inserting an expression of the form <<< Dg, Db, Ns, S >>> between the function name and the parenthesized argument list, where:  ‣  Dg is of type dim3 (see dim3) and specifies the dimension and size of the grid, such that Dg.x * Dg.y * Dg.z equals the number of blocks being launched; ‣  Db is of type dim3 (see dim3) and specifies the dimension and size of each block, such that Db.x * Db.y * Db.z equals the number of threads per block; ‣  Ns is of type size_t and specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory; this dynamically allocated memory is used by any of the variables declared as an external array as mentioned in \_\_shared\_\_; Ns is an optional argument which defaults to 0; ‣  S is of type cudaStream_t and specifies the associated stream; S is an optional argument which defaults to 0.  As an example, a function declared as  \_\_global\_\_ void Func(float* parameter);  must be called like this:  Func<<< Dg, Db, Ns >>> (parameter);  The arguments to the execution configuration are evaluated before the actual function arguments.  The function call will fail if Dg or Db are greater than the maximum sizes allowed for the device as specified in Compute Capabilities, or if Ns is greater than the maximum amount of shared memory available on the device, minus the amount of shared memory required for static allocation.  B.23. Launch Bounds  As discussed in detail in Multiprocessor Level, the fewer registers a kernel uses, the more threads and thread blocks are likely to reside on a multiprocessor, which can improve performance.  Therefore, the compiler uses heuristics to minimize register usage while keeping register spilling (see Device Memory Accesses) and instruction count to a minimum.  www.nvidia.com CUDA C++ Programming Guide                           PG-02829-001_v10.2 | 180  ----------------------- Page 197----------------------- C++ Language Extensions  An application can optionally aid these heuristics by providing additional information to the compiler in the form of launch bounds that are specified using the __launch_bounds__() qualifier in the definition of a \_\_global\_\_ function:  \_\_global\_\_ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) MyKernel(...) { ... }  ‣  maxThreadsPerBlock specifies the maximum number of threads per block with which the application will ever launch MyKernel(); it compiles to the .maxntid PTX directive; ‣  minBlocksPerMultiprocessor is optional and specifies the desired minimum number of resident blocks per multiprocessor; it compiles to the .minnctapersm PTX directive.  If launch bounds are specified, the compiler first derives from them the upper limit L on the number of registers the kernel should use to ensure that minBlocksPerMultiprocessor blocks (or a single block if minBlocksPerMultiprocessor is not specified) of maxThreadsPerBlock threads can reside on the multiprocessor (see Hardware Multithreading for the relationship between the number of registers used by a kernel and the number of registers allocated per block). The compiler then optimizes register usage in the following way:  ‣  If the initial register usage is higher than L, the compiler reduces it further until it becomes less or equal to L, usually at the expense of more local memory usage and/ or higher number of instructions; ‣  If the initial register usage is lower than L  ‣  If maxThreadsPerBlock is specified and minBlocksPerMultiprocessor is not, the compiler uses maxThreadsPerBlock to determine the register usage thresholds for the transitions between n and n+1 resident blocks (i.e., when using one less register makes room for an additional resident block as in the example of Multiprocessor Level) and then applies similar heuristics as when no launch bounds are specified; ‣  If both minBlocksPerMultiprocessor and maxThreadsPerBlock are specified, the compiler may increase register usage as high as L to reduce the number of instructions and better hide single thread instruction latency.  A kernel will fail to launch if it is executed with more threads per block than its launch bound maxThreadsPerBlock.  www.nvidia.com CUDA C++ Programming Guide                           PG-02829-001_v10.2 | 181  ----------------------- Page 198----------------------- C++ Language Extensions  Optimal launch bounds for a given kernel will usually differ across major architecture revisions. The sample code below shows how this is typically handled in device code using the __CUDA_ARCH__ macro introduced in Application Compatibility  #define THREADS_PER_BLOCK     256 #if __CUDA_ARCH__ >= 200 #define MY_KERNEL_MAX_THREADS (2 * THREADS_PER_BLOCK) #define MY_KERNEL_MIN_BLOCKS  3 #else #define MY_KERNEL_MAX_THREADS THREADS_PER_BLOCK #define MY_KERNEL_MIN_BLOCKS  2 #endif  // Device code \_\_global\_\_ void __launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS) MyKernel(...) { ... }  In the common case where MyKernel is invoked with the maximum number of threads per block (specified as the first parameter of __launch_bounds__()), it is tempting to use MY_KERNEL_MAX_THREADS as the number of threads per block in the execution configuration:  // Host code MyKernel<<<blocksPerGrid, MY_KERNEL_MAX_THREADS>>> (...);  This will not work however since __CUDA_ARCH__ is undefined in host code as mentioned in Application Compatibility, so MyKernel will launch with 256 threads per block even when __CUDA_ARCH__ is greater or equal to 200. Instead the number of threads per block should be determined:  ‣  Either at compile time using a macro that does not depend on __CUDA_ARCH__, for example  // Host code MyKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>> (...);  ‣  Or at runtime based on the compute capability  // Host code cudaGetDeviceProperties(&deviceProp, device); int threadsPerBlock = (deviceProp.major >= 2 ? 2 * THREADS_PER_BLOCK : THREADS_PER_BLOCK); MyKernel<<<blocksPerGrid, threadsPerBlock>>> (...);  Register usage is reported by the --ptxas options=-v compiler option. The number of resident blocks can be derived from the occupancy reported by the CUDA profiler (see Device Memory Accessesfor a definition of occupancy).  Register usage can also be controlled for all \_\_global\_\_ functions in a file using the maxrregcount compiler option. The value of maxrregcount is ignored for functions with launch bounds.  www.nvidia.com CUDA C++ Programming Guide                       



B.24. #pragma unroll  

By default, the compiler unrolls small loops with a known trip count. The #pragma unroll directive however can be used to control unrolling of any given loop. It must be placed immediately before the loop and only applies to that loop. It is optionally 9 followed by an integral constant expression (ICE) . If the ICE is absent, the loop will be completely unrolled if its trip count is constant. If the ICE evaluates to 1, the compiler will not unroll the loop. The pragma will be ignored if the ICE evaluates to a non- positive integer or to an integer greater than the maximum value representable by the int data type.  Examples:  struct S1_t { static const int value = 4; }; template <int X, typename T2> \_\_device\_\_ void foo(int *p1, int *p2) {  // no argument specified, loop will be completely unrolled #pragma unroll for (int i = 0; i < 12; ++i) p1[i] += p2[i]*2;  // unroll value = 8 #pragma unroll (X+1) for (int i = 0; i < 12; ++i) p1[i] += p2[i]*4;  // unroll value = 1, loop unrolling disabled #pragma unroll 1 for (int i = 0; i < 12; ++i) p1[i] += p2[i]*8;  // unroll value = 4 #pragma unroll (T2::value) for (int i = 0; i < 12; ++i) p1[i] += p2[i]*16; }  \_\_global\_\_ void bar(int *p1, int *p2) { foo<7, S1_t>(p1, p2); }  B.25. SIMD Video Instructions  PTX ISA version 3.0 includes SIMD (Single Instruction, Multiple Data) video instructions which operate on pairs of 16-bit values and quads of 8-bit values. These are available on devices of compute capability 3.0.  The SIMD video instructions are:  ‣  vadd2, vadd4 ‣  vsub2, vsub4  9 See the C++ Standard for definition of integral constant expression.  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 183  ----------------------- Page 200----------------------- C++ Language Extensions  ‣  vavrg2, vavrg4 ‣  vabsdiff2, vabsdiff4 ‣  vmin2, vmin4 ‣  vmax2, vmax4 ‣  vset2, vset4  PTX instructions, such as the SIMD video instructions, can be included in CUDA programs by way of the assembler, asm(), statement.  The basic syntax of an asm() statement is:  asm("template-string" : "constraint"(output) : "constraint"(input)"));  An example of using the vabsdiff4 PTX instruction is:  asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;": "=r" (result):"r" (A), "r" (B), "r" (C));  This uses the vabsdiff4 instruction to compute an integer quad byte SIMD sum of absolute differences. The absolute difference value is computed for each byte of the unsigned integers A and B in SIMD fashion. The optional accumulate operation ( .add) is specified to sum these differences.  Refer to the document "Using Inline PTX Assembly in CUDA" for details on using the assembly statement in your code. Refer to the PTX ISA documentation ("Parallel Thread Execution ISA Version 3.0" for example) for details on the PTX instructions for the version of PTX that you are using.  www.nvidia.com CUDA C++ Programming Guide                           PG-02829-001_v10.2 | 184  ----------------------- Page 201----------------------- 

 

## Appendix C. COOPERATIVE GROUPS 

### C.1. Introduction  

Cooperative Groups is an extension to the CUDA programming model, introduced in CUDA 9, for organizing groups of communicating threads. Cooperative Groups allows developers to express the granularity at which threads are communicating, helping them to express richer, more efficient parallel decompositions.  Historically, the CUDA programming model has provided a single, simple construct for synchronizing cooperating threads: a barrier across all threads of a thread block, as implemented with the \_\_syncthreads() intrinsic function. However, programmers would like to define and synchronize groups of threads at other granularities to enable greater performance, design flexibility, and software reuse in the form of “collective” group-wide function interfaces. In an effort to express broader patterns of parallel interaction, many performance-oriented programmers have resorted to writing their own ad hoc and unsafe primitives for synchronizing threads within a single warp, or across sets of thread blocks running on a single GPU. Whilst the performance improvements achieved have often been valuable, this has resulted in an ever-growing collection of brittle code that is expensive to write, tune, and maintain over time and across GPU generations. Cooperative Groups addresses this by providing a safe and future-proof mechanism to enable performant code.  The Cooperative Groups programming model extension describes synchronization patterns both within and across CUDA thread blocks. It provides both the means for applications to define their own groups of threads, and the interfaces to synchronize them. It also provides new launch APIs that enforce certain restrictions and therefore can guarantee the synchronization will work. These primitives enable new patterns of cooperative parallelism within CUDA, including producer-consumer parallelism, opportunistic parallelism, and global synchronization across the entire Grid.  The expression of groups as first-class program objects improves software composition, since collective functions can receive an explicit object representing the group of participating threads. This object also makes programmer intent explicit, which eliminates unsound architectural assumptions that result in brittle code, undesirable __

 Cooperative Groups  restrictions upon compiler optimizations, and better compatibility with new GPU generations.  The Cooperative Groups programming model consists of the following elements:  ‣  data types for representing groups of cooperating threads; ‣  operations to obtain intrinsic groups defined by the CUDA launch API (e.g., thread blocks); ‣  operations for partitioning existing groups into new groups; ‣  a barrier operation to synchronize a given group; ‣  and operations to inspect the group properties as well as group-specific collectives. 

###  C.2. Intra-block Groups 

In this section we describe the functionality available to create groups of threads within a thread block that can synchronize and collaborate. Note that the use of Cooperative Groups for synchronization across thread blocks or devices requires some additional considerations, as described later.  Cooperative Groups requires CUDA 9.0 or later. To use Cooperative Groups, include the header file:  #include <cooperative_groups.h>  and use the Cooperative Groups namespace:  using namespace cooperative_groups;  Then code containing any intra-block Cooperative Groups functionality can be compiled in the normal way using nvcc.  

C.2.1. Thread Groups and Thread Blocks  Any CUDA programmer is already familiar with a certain group of threads: the thread block. The Cooperative Groups extension introduces a new datatype, thread_block, to explicitly represent this concept within the kernel. The group can be initialized as follows:  thread_block g = this_thread_block();  The thread_block datatype is derived from the more generic thread_group datatype, which can be used to represent a wider class of groups. thread_group provides the following functionality:  void sync(); // Synchronize the threads in the group unsigned size(); // Total number of threads in the group unsigned thread_rank(); // Rank of the calling thread within [0, size] bool is_valid(); // Whether the group violated any APIconstraints  whereas thread_block provides the following additional block-specific functionality:  dim3 group_index(); // 3-dimensional block index within the grid dim3 thread_index(); // 3-dimensional thread index within the block  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 186  ----------------------- Page 203----------------------- Cooperative Groups  For example, if the group g is initialized as above, then  g.sync();  will synchronize all threads in the block (i.e. equivalent to \_\_syncthreads();).  Note that all threads in the group must participate in collective operations, or the behavior is undefined.  __

C.2.2. Tiled Partitions  

The tiled_partition() function can be used to decompose the thread block into multiple smaller groups of cooperative threads. For example, if we first create a group containing all the threads in the block:  thread_block wholeBlock = this_thread_block();  then we can partition this into smaller groups, each of size 32 threads:  thread_group tile32 = tiled_partition(wholeBlock, 32);  and, furthermore, we can partition each of these groups into even smaller groups, each of size 4 threads:  thread_group tile4 = tiled_partition(tile32, 4);  If, for instance, if we were to then include the following line of code:  if (tile4.thread_rank()==0) printf(“Hello from tile4 rank 0\n”);  then the statement would be printed by every fourth thread in the block: the threads of rank 0 in each tile4 group, which correspond to those threads with ranks 0,4,8,12… in the wholeBlock group.   Note that, currently, only supported are tile sizes which are a power of 2 and no larger than 32.  C.2.3. Thread Block Tiles  An alternative templated version of the tiled_partition function is available, where a template parameter is used to specify the size of the tile: with this known at compile time there is the potential for more optimal execution. Analogous to that in the previous section, the following code will create two sets of tiled groups, of size 32 and 4 respectively:  thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block()); thread_block_tile<4> tile4 = tiled_partition<4>(this_thread_block());  Note that the thread_block_tile templated data structure is being used here, and that the size of the group is passed to the tiled_partition call as a template parameter rather than an argument.  www.nvidia.com CUDA C++ Programming Guide                           PG-02829-001_v10.2 | 187  ----------------------- Page 204----------------------- Cooperative Groups  Thread Block Tiles also expose additional functionality as follows:  .shfl() .shfl_down() .shfl_up() .shfl_xor() .any() .all() .ballot() .match_any() .match_all()  where these cooperative synchronous operations are analogous to those described in Warp Shuffle Functions and Warp Vote Functions. However their use here, in the context of these user-defined Cooperative Groups, offers enhanced flexibility and productivity. This functionality will be demonstrated later in this appendix.  As mentioned above, only supported are tile sizes which are a power of 2 and no larger than 32. 

 C.2.4. Coalesced Groups 

 In CUDA’s SIMT architecture, at the hardware level the multiprocessor executes threads in groups of 32 called warps. If there exists a data-dependent conditional branch in the application code such that threads within a warp diverge, then the warp serially executes each branch disabling threads not on that path. The threads that remain active on the path are referred to as coalesced. Cooperative Groups has functionality to discover, and create, a group containing all coalesced threads as follows:  coalesced_group active = coalesced_threads();  For example, consider a situation whereby there is a branch in the code in which only the 2nd, 4th and 8th threads in each warp are active. The above call, placed in that branch, will create (for each warp) a group, active, that has three threads (with ranks 0-2 inclusive). 

 C.2.5. Uses of Intra-block Cooperative Groups  In this section, Cooperative Group functionality is illustrated through some usage examples.  C.2.5.1. Discovery Pattern  Commonly developers need to work with the current active set of threads. No assumption is made about the threads that are present, and instead developers work with the threads that happen to be there. This is seen in the following “aggregating  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 188  ----------------------- Page 205----------------------- Cooperative Groups  atomic increment across threads in a warp” example (written using the correct CUDA 9.0 set of intrinsics):  { unsigned int writemask = __activemask(); unsigned int total = __popc(writemask); unsigned int prefix = __popc(writemask & __lanemask_lt()); // Find the lowest-numbered active lane int elected_lane = __ffs(writemask) - 1; int base_offset = 0; if (prefix == 0) { base_offset = atomicAdd(p, total); } base_offset = __shfl_sync(writemask, base_offset, elected_lane); int thread_offset = prefix + base_offset; return thread_offset; }  This can be re-written with Cooperative Groups as follows:  { cg::coalesced_group g = cg::coalesced_threads(); int prev; if (g.thread_rank() == 0) { prev = atomicAdd(p, g.size()); } prev = g.thread_rank() + g.shfl(prev, 0); return prev; }  

C.2.5.2. Warp-Synchronous Code Pattern  Developers might have had warp-synchronous codes that they previously made implicit assumptions about the warp size and would code around that number. Now this needs to be specified explicitly.  // If the size is known statically auto g = tiled_partition<16>(this_thread_block()); // Can use g.shfl and all other warp-synchronous builtins  However, the user might want to better partition his algorithm, but without needing the advantage of warp-synchronous builtins.  auto g = tiled_partition(this_thread_block(), 8);  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 189  ----------------------- Page 206----------------------- Cooperative Groups  In this case, the group g can still synchronize and you can still build varied parallel algorithms on top, but shfl() etc. are not accessible.  \_\_global\_\_ void cooperative_kernel(...) { // obtain default "current thread block" group thread_group my_block = this_thread_block();  // subdivide into 32-thread, tiled subgroups // Tiled subgroups evenly partition a parent group into // adjacent sets of threads - in this case each one warp in size thread_group my_tile = tiled_partition(my_block, 32);  // This operation will be performed by only the // first 32-thread tile of each block if (my_block.thread_rank() < 32) { // ... my_tile.sync(); } } 

 C.2.5.3. Composition  Previously, there were hidden constraints on the implementation when writing certain code. Take this example:  \_\_device\_\_ int sum(int *x, int n) { // ... \_\_syncthreads(); return total; }  \_\_global\_\_ void parallel_kernel(float *x){ // ... // Entire thread block must call sum sum(x, n); }  All threads in the thread block must arrive at the \_\_syncthreads() barrier, however, this constraint is hidden from the developer who might want to use sum(…). With Cooperative Groups, a better way of writing this would be:  \_\_device\_\_ int sum(const thread_group& g, int *x, int n) { // ... g.sync() return total; }  \_\_global\_\_ void parallel_kernel(...) { // ... // Entire thread block must call sum sum(this_thread_block(), x, n); // ... }  C.3. Grid Synchronization  Prior to the introduction of Cooperative Groups, the CUDA programming model only allowed synchronization between thread blocks at a kernel completion boundary. The  www.nvidia.com CUDA C++ Programming Guide                     

Cooperative Groups  kernel boundary carries with it an implicit invalidation of state, and with it, potential performance implications.  For example, in certain use cases, applications have a large number of small kernels, with each kernel representing a stage in a processing pipeline. The presence of these kernels is required by the current CUDA programming model to ensure that the thread blocks operating on one pipeline stage have produced data before the thread block operating on the next pipeline stage is ready to consume it. In such cases, the ability to provide global inter thread block synchronization would allow the application to be restructured to have persistent thread blocks, which are able to synchronize on the device when a given stage is complete.  To synchronize across the grid, from within a kernel, you would simply use the group:  grid_group grid = this_grid();  and call:  grid.sync();  To enable grid synchronization, when launching the kernel it is necessary to use, instead of the <<<...>>> execution configuration syntax, the cudaLaunchCooperativeKernel CUDA runtime launch API:  cudaLaunchCooperativeKernel( const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem = 0, cudaStream_t stream = 0 )  (or the CUDA driver equivalent).  To guarantee co-residency of the thread blocks on the GPU, the number of blocks launched needs to be carefully considered. For example, as many blocks as there are SMs can be launched as follows:  cudaDeviceProp deviceProp; cudaGetDeviceProperties(&deviceProp, dev); // initialize, then launch cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);  Alternatively, you can maximize the exposed parallelism by calculating how many blocks can fit simultaneously per-SM using the occupancy calculator as follows:   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0); // initialize, then launch cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount*numBlocksPerSm, numThreads, args);  Note also that to use grid synchronization, the device code must be compiled in separate compilation (see the "Using Separate Compilation in CUDA" section in the CUDA  www.nvidia.com CUDA C++ Programming Guide                          PG-02829-001_v10.2 | 191  ----------------------- Page 208----------------------- Cooperative Groups  Compiler Driver NVCC documentation) and the device runtime linked in. The simplest example is:  nvcc -arch=sm_61 -rdc=true mytestfile.cu -o mytest  You should also ensure the device supports the cooperative launch property, as can be determined by usage of the cuDeviceGetAttribute CUDA driver API :  int pi=0; cuDevice dev; cuDeviceGet(&dev,0) // get handle to device 0 cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, dev);  which will set pi to 1 if the property is supported on device 0. Only devices with compute capability of 6.0 and higher are supported. In addition, you need to be running on either of these:  ‣  The Linux platform without MPS, or ‣  The Linux platform with MPS and on a device with compute capability 7.0 or higher, or ‣  On current versions of Windows with the device in TCC mode. 

 C.4. Multi-Device Synchronization  In order to enable synchronization across multiple devices with Cooperative Groups, use of the cuLaunchCooperativeKernelMultiDevice CUDA API is required. This, a significant departure from existing CUDA APIs, will allow a single host thread to launch a kernel across multiple devices. In addition to the constraints and guarantees made by cuLaunchCooperativeKernel, this API has the additional semantics:  ‣  This API will ensure that a launch is atomic, i.e. if the API call succeeds, then the provided number of thread blocks will launch on all specified devices. ‣  The functions launched via this API must be identical. No explicit checks are done by the driver in this regard because it is largely not feasible. It is up to the application to ensure this. ‣  No two entries in the provided launchParamsList may map to the same device. ‣  All devices being targeted by this launch must be of the same compute capability - major and minor versions. ‣  The block size, grid size and amount of shared memory per grid must be the same across all devices. Note that this means the maximum number of blocks that can be launched per device will be limited by the device with the least number of SMs. ‣  Any user defined \_\_device\_\_, \_\_constant\_\_ or \_\_managed\_\_ device global variables present in the module that owns the CUfunction being launched are independently instantiated on every device. The user is responsible for initializing such device global variables appropriately.  www.nvidia.com CUDA C++ Programming Guide                            PG-02829-001_v10.2 | 192  ----------------------- Page 209----------------------- Cooperative Groups  The launch parameters should be defined using an array of structs (one per device):  typedef struct CUDA_LAUNCH_PARAMS_st { CUfunction function; unsigned int gridDimX; unsigned int gridDimY; unsigned int gridDimZ; unsigned int blockDimX; unsigned int blockDimY; unsigned int blockDimZ; unsigned int sharedMemBytes; CUstream hStream; void **kernelParams; } CUDA_LAUNCH_PARAMS;  and passed into the launch API:  cudaLaunchCooperativeKernelMultiDevice( CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags = 0);  in a similar fashion to that for grid-wide synchronization described above. Also, as with grid-wide synchronization, the resulting device code looks very similar:  multi_grid_group multi_grid = this_multi_grid(); multi_grid.sync();  and needs to be compiled in separate compilation.  You should also ensure the device supports the cooperative multi device launch property in a similar way to that described in the previous section, but with use of CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH instead of CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH. Only devices with compute capability of 6.0 and higher are supported. In addition, you need to be running on the Linux platform (without MPS) or on current versions of Windows with the device in TCC mode.  