## chapter5 Performance Guidelines

### 5.1. Overall Performance Optimization Strategies

​        Performance optimization revolves around three basic strategies:
1、Maximize parallel execution to achieve maximum utilization;
2、Optimize memory usage to achieve maximum memory throughput;
3、Optimize instruction usage to achieve maximum instruction throughput.
​        Which strategies will yield the best performance gain for a particular portion of an application depends on the performance limiters for that portion; optimizing instruction usage of a kernel that is mostly limited by memory accesses will not yield any significant performance gain, for example. Optimization efforts should therefore be constantly directed by measuring and monitoring the performance limiters, for example using the CUDA profiler. Also, comparing the floating-point operation throughput or memory throughput - whichever makes more sense - of a particular kernel to the corresponding peak theoretical throughput of the device indicates how much room for improvement there is for the kernel.  

### 5.2. Maximize Utilization

​        To maximize utilization the application should be structured in a way that it exposes as much parallelism as possible and efficiently maps this parallelism to the various components of the system to keep them busy most of the time.

#### 5.2.1. Application Level

​        At a high level, the application should maximize parallel execution between the host, the
devices, and the bus connecting the host to the devices, by using asynchronous functions
calls and streams as described in Asynchronous Concurrent Execution. It should assign
to each processor the type of work it does best: serial workloads to the host; parallel
workloads to the devices.  

​        For the parallel workloads, at points in the algorithm where parallelism is broken
because some threads need to synchronize in order to share data with each other,
there are two cases: Either these threads belong to the same block, in which case they
should use __syncthreads() and share data through shared memory within the same
kernel invocation, or they belong to different blocks, in which case they must share
data through global memory using two separate kernel invocations, one for writing to
and one for reading from global memory. The second case is much less optimal since it
adds the overhead of extra kernel invocations and global memory traffic. Its occurrence
should therefore be minimized by mapping the algorithm to the CUDA programming
model in such a way that the computations that require inter-thread communication are
performed within a single thread block as much as possible.

#### 5.2.2. Device Level

​        At a lower level, the application should maximize parallel execution between the
multiprocessors of a device.
​        Multiple kernels can execute concurrently on a device, so maximum utilization can
also be achieved by using streams to enable enough kernels to execute concurrently as
described in Asynchronous Concurrent Execution.

#### 5.2.3. Multiprocessor Level

​    At an even lower level, the application should maximize parallel execution between the various functional units within a multiprocessor.
​        As described in Hardware Multithreading, a GPU multiprocessor primarily relies on
thread-level parallelism to maximize utilization of its functional units. Utilization is
therefore directly linked to the number of resident warps. At every instruction issue
time, a warp scheduler selects an instruction that is ready to execute. This instruction
can be another independent instruction of the same warp, exploiting instruction-level
parallelism, or more commonly an instruction of another warp, exploiting thread-level
parallelism. If a ready to execute instruction is selected it is issued to the active threads
of the warp. The number of clock cycles it takes for a warp to be ready to execute its next
instruction is called the latency, and full utilization is achieved when all warp schedulers
always have some instruction to issue for some warp at every clock cycle during that
latency period, or in other words, when latency is completely "hidden". The number
of instructions required to hide a latency of L clock cycles depends on the respective
throughputs of these instructions (see Arithmetic Instructions for the throughputs of
various arithmetic instructions). If we assume instructions with maximum throughput, it
is equal to:
1、4L for devices of compute capability 5.x, 6.1, 6.2 and 7.x since for these devices, a
multiprocessor issues one instruction per warp over one clock cycle for four warps
at a time, as mentioned in Compute Capabilities.
2、2L for devices of compute capability 6.0 since for these devices, the two instructions
issued every cycle are one instruction for two different warps.
2、8L for devices of compute capability 3.x since for these devices, the eight instructions
issued every cycle are four pairs for four different warps, each pair being for the
same warp.

​        The most common reason a warp is not ready to execute its next instruction is that the instruction's input operands are not available yet.

​        If all input operands are registers, latency is caused by register dependencies, i.e., some of the input operands are written by some previous instruction(s) whose execution has not completed yet. In this case, the latency is equal to the execution time of the previous instruction and the warp schedulers must schedule instructions of other warps during that time. Execution time varies depending on the instruction. On devices of compute capability 7.x, for most arithmetic instructions, it is typically 4 clock cycles. This means that 16 active warps per multiprocessor (4 cycles, 4 warp schedulers) are required to hide arithmetic instruction latencies (assuming that warps execute instructions with maximum throughput, otherwise fewer warps are needed). If the individual warps exhibit instruction-level parallelism, i.e. have multiple independent instructions in their instruction stream, fewer warps are needed because multiple independent instructions from a single warp can be issued back to back.

​          If some input operand resides in off-chip memory, the latency is much higher: typically hundreds of clock cycles. The number of warps required to keep the warp schedulers busy during such high latency periods depends on the kernel code and its degree of instruction-level parallelism. In general, more warps are required if the ratio of the number of instructions with no off-chip memory operands (i.e., arithmetic instructions most of the time) to the number of instructions with off-chip memory operands is low (this ratio is commonly called the arithmetic intensity of the program). 

​          Another reason a warp is not ready to execute its next instruction is that it is waiting at some memory fence (Memory Fence Functions) or synchronization point (Memory Fence Functions). A synchronization point can force the multiprocessor to idle as more and more warps wait for other warps in the same block to complete execution of instructions prior to the synchronization point. Having multiple resident blocks per multiprocessor can help reduce idling in this case, as warps from different blocks do not need to wait for each other at synchronization points. 

​         The number of blocks and warps residing on each multiprocessor for a given kernel call depends on the execution configuration of the call (Execution Configuration), the memory resources of the multiprocessor, and the resource requirements of the kernel as described in Hardware Multithreading. Register and shared memory usage are reported by the compiler when compiling with the -ptxas-options=-v option. 

​        The total amount of shared memory required for a block is equal to the sum of the amount of statically allocated shared memory and the amount of dynamically allocated shared memory. 

​        The number of registers used by a kernel can have a significant impact on the number of resident warps. For example, for devices of compute capability 6.x, if a kernel uses 64 registers and each block has 512 threads and requires very little shared memory, then two blocks (i.e., 32 warps) can reside on the multiprocessor since they require 2x512x64 registers, which exactly matches the number of registers available on the multiprocessor. But as soon as the kernel uses one more register, only one block (i.e., 16 warps) can be resident since two blocks would require 2x512x65 registers, which are more registers than are available on the multiprocessor. Therefore, the compiler attempts to minimize register usage while keeping register spilling (see Device Memory Accesses)  and the number of instructions to a minimum. Register usage can be controlled using the maxrregcount compiler option or launch bounds as described in Launch Bounds.  

​        The register file is organized as 32-bit registers. So, each variable stored in a register needs at least one 32-bit register, e.g. a double variable uses two 32-bit registers.

​        The effect of execution configuration on performance for a given kernel call generally depends on the kernel code. Experimentation is therefore recommended. Applications can also parameterize execution configurations based on register file size and shared memory size, which depends on the compute capability of the device, as well as on the number of multiprocessors and memory bandwidth of the device, all of which can be queried using the runtime (see reference manual). 

​        The number of threads per block should be chosen as a multiple of the warp size to avoid wasting computing resources with under-populated warps as much as possible.

#####  5.2.3.1. Occupancy Calculator 

​        Several API functions exist to assist programmers in choosing thread block size based on register and shared memory requirements. 

1、 The occupancy calculator API, cudaOccupancyMaxActiveBlocksPerMultiprocessor, can provide an occupancy prediction based on the block size and shared memory usage of a kernel. This function reports occupancy in terms of the number of concurrent thread blocks per multiprocessor.
    1.1、Note that this value can be converted to other metrics. Multiplying by the number of warps per block yields the number of concurrent warps per multiprocessor; further dividing concurrent warps by max warps per multiprocessor gives the occupancy as a percentage.
2、The occupancy-based launch configurator APIs, cudaOccupancyMaxPotentialBlockSize and cudaOccupancyMaxPotentialBlockSizeVariableSMem, heuristically calculate an execution configuration that achieves the maximum multiprocessor-level occupancy. 

The following code sample calculates the occupancy of MyKernel. It then reports the occupancy level with the ratio between concurrent warps versus maximum warps per multiprocessor.  

```c++
// Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}
// Host code
int main()
{
    int numBlocks; // Occupancy in terms of active blocks
    int blockSize = 32;
    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
    											  MyKernel,
    											  blockSize,
    											  0);
    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" <<
    std::endl;
    return 0;
}
```

​         The following code sample configures an occupancy-based kernel launch of MyKernel according to the user input.  

```c++
// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < arrayCount) {
		array[idx] *= array[idx];
	}
}
// Host code
int launchMyKernel(int *array, int arrayCount)
{
	int blockSize; // The launch configurator returned block size
	int minGridSize; // The minimum grid size needed to achieve the
	// maximum occupancy for a full device
	// launch
	int gridSize; // The actual grid size needed, based on input
	// size
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)MyKernel, 0, arrayCount);
	// Round up according to array size
	gridSize = (arrayCount + blockSize - 1) / blockSize;
	MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
	cudaDeviceSynchronize();
	// If interested, the occupancy can be calculated with
	// cudaOccupancyMaxActiveBlocksPerMultiprocessor
	return 0;
}
```

​        The CUDA Toolkit also provides a self-documenting, standalone occupancy calculator and launch configurator implementation in <CUDA_Toolkit_Path>/include/ cuda_occupancy.h for any use cases that cannot depend on the CUDA software stack. A spreadsheet version of the occupancy calculator is also provided. The spreadsheet version is particularly useful as a learning tool that visualizes the impact of changes to the parameters that affect occupancy (block size, registers per thread, and shared memory per thread). 

### 5.3. Maximize Memory Throughput 

​       The first step in maximizing overall memory throughput for the application is to minimize data transfers with low bandwidth. 

​        That means minimizing data transfers between the host and the device, as detailed in Data Transfer between Host and Device, since these have much lower bandwidth than data transfers between global memory and the device.  

​        That also means minimizing data transfers between global memory and the device by maximizing use of on-chip memory: shared memory and caches (i.e., L1 cache and L2 cache available on devices of compute capability 2.x and higher, texture cache and constant cache available on all devices).

​        Shared memory is equivalent to a user-managed cache: The application explicitly allocates and accesses it. As illustrated in CUDA Runtime, a typical programming pattern is to stage data coming from device memory into shared memory; in other words, to have each thread of a block: 

1、 Load data from device memory to shared memory, 

2、Synchronize with all the other threads of the block so that each thread can safely read shared memory locations that were populated by different threads, 

3、Process the data in shared memory, 

4、Synchronize again if necessary to make sure that shared memory has been updated with the results, 

5、Write the results back to device memory.

​        For some applications (e.g., for which global memory access patterns are datadependent), a traditional hardware-managed cache is more appropriate to exploit data locality. As mentioned in Compute Capability 3.x and Compute Capability 7.x, for devices of compute capability 3.x and 7.x, the same on-chip memory is used for both L1 and shared memory, and how much of it is dedicated to L1 versus shared memory is configurable for each kernel call. 

​        The throughput of memory accesses by a kernel can vary by an order of magnitude depending on access pattern for each type of memory. The next step in maximizing memory throughput is therefore to organize memory accesses as optimally as possible based on the optimal memory access patterns described in Device Memory Accesses. This optimization is especially important for global memory accesses as global memory bandwidth is low compared to available on-chip bandwidths and arithmetic instruction throughput, so non-optimal global memory accesses generally have a high impact on performance.

####         5.3.1. Data Transfer between Host and Device 

​         Applications should strive to minimize data transfer between the host and the device. One way to accomplish this is to move more code from the host to the device, even if that means running kernels that do not expose enough parallelism to execute on the device with full efficiency. Intermediate data structures may be created in device memory, operated on by the device, and destroyed without ever being mapped by the host or copied to host memory. 

​        Also, because of the overhead associated with each transfer, batching many small transfers into a single large transfer always performs better than making each transfer separately. 

​        On systems with a front-side bus, higher performance for data transfers between host and device is achieved by using page-locked host memory as described in Page-Locked Host Memory.

​         In addition, when using mapped page-locked memory (Mapped Memory), there is no need to allocate any device memory and explicitly copy data between device and host memory. Data transfers are implicitly performed each time the kernel accesses the mapped memory. For maximum performance, these memory accesses must be coalesced as with accesses to global memory (see Device Memory Accesses). Assuming that they are and that the mapped memory is read or written only once, using mapped pagelocked memory instead of explicit copies between device and host memory can be a win for performance. 

​        On integrated systems where device memory and host memory are physically the same, any copy between host and device memory is superfluous and mapped page-locked memory should be used instead. Applications may query a device is integrated by checking that the integrated device property (see Device Enumeration) is equal to 1. 

#### 5.3.2. Device Memory Accesses 

​        An instruction that accesses addressable memory (i.e., global, local, shared, constant, or texture memory) might need to be re-issued multiple times depending on the distribution of the memory addresses across the threads within the warp. How the distribution affects the instruction throughput this way is specific to each type of memory and described in the following sections. For example, for global memory, as a general rule, the more scattered the addresses are, the more reduced the throughput is. 

##### Global Memory 

​        Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned: Only the 32-, 64-, or 128-byte segments of device memory that are aligned to their size (i.e., whose first address is a multiple of their size) can be read or written by memory transactions. 

​        When a warp executes an instruction that accesses global memory, it coalesces the memory accesses of the threads within the warp into one or more of these memory transactions depending on the size of the word accessed by each thread and the distribution of the memory addresses across the threads. In general, the more transactions are necessary, the more unused words are transferred in addition to the words accessed by the threads, reducing the instruction throughput accordingly. For example, if a 32-byte memory transaction is generated for each thread's 4-byte access, throughput is divided by 8. 

How many transactions are necessary and how much throughput is ultimately affected varies with the compute capability of the device. Compute Capability 3.x, Compute Capability 5.x, Compute Capability 6.x and Compute Capability 7.x give more details on how global memory accesses are handled for various compute capabilities. 

To maximize global memory throughput, it is therefore important to maximize coalescing by: 

1、Following the most optimal access patterns based on Compute Capability 3.x, Compute Capability 5.x, Compute Capability 6.x and Compute Capability 7.x, 

2、Using data types that meet the size and alignment requirement detailed in the section Size and Alignment Requirement below,

3、Padding data in some cases, for example, when accessing a two-dimensional array as described in the section Two-Dimensional Arrays below. 

##### Size and Alignment Requirement 

​        Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to data residing in global memory compiles to a single global memory instruction if and only if the size of the data type is 1, 2, 4, 8, or 16 bytes and the data is naturally aligned (i.e., its address is a multiple of that size).

​        If this size and alignment requirement is not fulfilled, the access compiles to multiple instructions with interleaved access patterns that prevent these instructions from fully coalescing. It is therefore recommended to use types that meet this requirement for data that resides in global memory. 

​        The alignment requirement is automatically fulfilled for the Built-in Vector Types. 

​        For structures, the size and alignment requirements can be enforced by the compiler using the alignment specifiers \_\_align\_\_(8) or \_\_align\_\_(16), such as 

```c++
struct __align__(8) {
    float x;
    float y;
};
```

or

```c++
struct __align__(16) {
    float x;
    float y;
    float z;
};
```

​        Any address of a variable residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is always aligned to at least 256 bytes.

​        Reading non-naturally aligned 8-byte or 16-byte words produces incorrect results (off by a few words), so special care must be taken to maintain alignment of the starting address of any value or array of values of these types. A typical case where this might be easily overlooked is when using some custom global memory allocation scheme, whereby the allocations of multiple arrays (with multiple calls to cudaMalloc() or cuMemAlloc()) is replaced by the allocation of a single large block of memory partitioned into multiple arrays, in which case the starting address of each array is offset from the block's starting address. 

##### Two-Dimensional Arrays 

A common global memory access pattern is when each thread of index (tx,ty) uses the following address to access one element of a 2D array of width width, located at address BaseAddress of type type* (where type meets the requirement described in Maximize Utilization): 

For these accesses to be fully coalesced, both the width of the thread block and the width of the array must be a multiple of the warp size.

 In particular, this means that an array whose width is not a multiple of this size will be accessed much more efficiently if it is actually allocated with a width rounded up to the closest multiple of this size and its rows padded accordingly. The cudaMallocPitch() and cuMemAllocPitch() functions and associated memory copy functions described in the reference manual enable programmers to write non-hardware-dependent code to allocate arrays that conform to these constraints. 

##### Local Memory 

Local memory accesses only occur for some automatic variables as mentioned in Variable Memory Space Specifiers. Automatic variables that the compiler is likely to place in local memory are:

1、Arrays for which it cannot determine that they are indexed with constant quantities, 

2、Large structures or arrays that would consume too much register space, 

3、Any variable if the kernel uses more registers than available (this is also known as register spilling). 

Inspection of the PTX assembly code (obtained by compiling with the -ptx orkeep option) will tell if a variable has been placed in local memory during the first compilation phases as it will be declared using the .local mnemonic and accessed using the ld.local and st.local mnemonics. Even if it has not, subsequent compilation phases might still decide otherwise though if they find it consumes too much register space for the targeted architecture: Inspection of the cubin object using cuobjdump will tell if this is the case. Also, the compiler reports total local memory usage per kernel (lmem) when compiling with the --ptxas-options=-v option. Note that some mathematical functions have implementation paths that might access local memory. 

The local memory space resides in device memory, so local memory accesses have the same high latency and low bandwidth as global memory accesses and are subject to the same requirements for memory coalescing as described in Device Memory Accesses. Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address (e.g., same index in an array variable, same member in a structure variable). 

On some devices of compute capability 3.x local memory accesses are always cached in L1 and L2 in the same way as global memory accesses (see Compute Capability 3.x). On devices of compute capability 5.x and 6.x, local memory accesses are always cached in L2 in the same way as global memory accesses (see Compute Capability 5.x and Compute Capability 6.x). 

##### Shared Memory 

​        Because it is on-chip, shared memory has much higher bandwidth and much lower latency than local or global memory. 

​       To achieve high bandwidth, shared memory is divided into equally-sized memory modules, called banks, which can be accessed simultaneously. Any memory read or write request made of n addresses that fall in n distinct memory banks can therefore be serviced simultaneously, yielding an overall bandwidth that is n times as high as the bandwidth of a single module. 

​        However, if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized. The hardware splits a memory request with bank conflicts into as many separate conflict-free requests as necessary, decreasing throughput by a factor equal to the number of separate memory requests. If the number of separate memory requests is n, the initial memory request is said to cause n-way bank conflicts. 

​         To get maximum performance, it is therefore important to understand how memory addresses map to memory banks in order to schedule the memory requests so as to minimize bank conflicts. This is described in Compute Capability 3.x, Compute Capability 5.x, Compute Capability 6.x, and Compute Capability 7.x for devices of compute capability 3.x, 5.x, 6.x and 7.x, respectively. 

##### Constant Memory

​        The constant memory space resides in device memory and is cached in the constant cache. 

​        A request is then split into as many separate requests as there are different memory addresses in the initial request, decreasing throughput by a factor equal to the number of separate requests. 

​         The resulting requests are then serviced at the throughput of the constant cache in case of a cache hit, or at the throughput of device memory otherwise.

##### Texture and Surface Memory 

​        The texture and surface memory spaces reside in device memory and are cached in texture cache, so a texture fetch or surface read costs one memory read from device memory only on a cache miss, otherwise it just costs one read from texture cache. The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture or surface addresses that are close together in 2D will achieve best performance. Also, it is designed for streaming fetches with a constant latency; a cache hit reduces DRAM bandwidth demand but not fetch latency.

​        Reading device memory through texture or surface fetching present some benefits that can make it an advantageous alternative to reading device memory from global or constant memory: 

1、 If the memory reads do not follow the access patterns that global or constant memory reads must follow to get good performance, higher bandwidth can be achieved providing that there is locality in the texture fetches or surface reads; 

2、Addressing calculations are performed outside the kernel by dedicated units; 

3、Packed data may be broadcast to separate variables in a single operation; 

4、8-bit and 16-bit integer input data may be optionally converted to 32 bit floatingpoint values in the range [0.0, 1.0] or [-1.0, 1.0] (see Texture Memory). 

### 5.4. Maximize Instruction Throughput 

​        To maximize instruction throughput the application should: 

1、Minimize the use of arithmetic instructions with low throughput; this includes trading precision for speed when it does not affect the end result, such as using intrinsic instead of regular functions (intrinsic functions are listed in Intrinsic Functions), single-precision instead of double-precision, or flushing denormalized numbers to zero; 

2、Minimize divergent warps caused by control flow instructions as detailed in Control Flow Instructions 

3、Reduce the number of instructions, for example, by optimizing out synchronization points whenever possible as described in Synchronization Instruction or by using restricted pointers as described in \_\_restrict\_\_. 

​       In this section, throughputs are given in number of operations per clock cycle per multiprocessor. For a warp size of 32, one instruction corresponds to 32 operations, so if N is the number of operations per clock cycle, the instruction throughput is N/32 instructions per clock cycle. 

​        All throughputs are for one multiprocessor. They must be multiplied by the number of multiprocessors in the device to get throughput for the whole device. 

#### 5.4.1. Arithmetic Instructions 

Table 3 gives the throughputs of the arithmetic instructions that are natively supported in hardware for devices of various compute capabilities. 

Table 3 Throughput of Native Arithmetic Instructions (Number of Results per Clock Cycle per Multiprocessor) 

there is a chart here

​        Other instructions and functions are implemented on top of the native instructions. The implementation may be different for devices of different compute capabilities, and the number of native instructions after compilation may fluctuate with every compiler version. For complicated functions, there can be multiple code paths depending on input. cuobjdump can be used to inspect a particular implementation in a cubin object.
​        The implementation of some functions are readily available on the CUDA header files
(math_functions.h, device_functions.h, ...).

​        In general, code compiled with -ftz=true (denormalized numbers are flushed to zero) tends to have higher performance than code compiled with -ftz=false. Similarly, code compiled with -prec div=false (less precise division) tends to have higher performance code than code compiled with -prec div=true, and code compiled with -prec-sqrt=false (less precise square root) tends to have higher performance  than code compiled with -prec-sqrt=true. The nvcc user manual describes these compilation flags in more details.

​        Other instructions and functions are implemented on top of the native instructions. The implementation may be different for devices of different compute capabilities, and the number of native instructions after compilation may fluctuate with every compiler version. For complicated functions, there can be multiple code paths depending on input. cuobjdump can be used to inspect a particular implementation in a cubin object. 

​        The implementation of some functions are readily available on the CUDA header files (math_functions.h, device_functions.h, ...). 

​        In general, code compiled with -ftz=true (denormalized numbers are flushed to zero) tends to have higher performance than code compiled with -ftz=false. Similarly, code compiled with -prec div=false (less precise division) tends to have higher performance code than code compiled with -prec div=true, and code compiled with -prec-sqrt=false (less precise square root) tends to have higher performance than code compiled with -prec-sqrt=true. The nvcc user manual describes these compilation flags in more details. 

#####  Single-Precision Floating-Point Division

​         __fdividef(x, y) (see Intrinsic Functions) provides faster single-precision floatingpoint division than the division operator.

##### Single-Precision Floating-Point Reciprocal Square Root 

​        To preserve IEEE-754 semantics the compiler can optimize 1.0/sqrtf() into rsqrtf() only when both reciprocal and square root are approximate, (i.e., with -precdiv=false and -prec-sqrt=false). It is therefore recommended to invoke rsqrtf() directly where desired.

 Single-Precision Floating-Point Square Root 

Single-precision floating-point square root is implemented as a reciprocal square root followed by a reciprocal instead of a reciprocal square root followed by a multiplication so that it gives correct results for 0 and infinity. 

##### Sine and Cosine 

​         sinf(x), cosf(x), tanf(x), sincosf(x), and corresponding double-precision instructions are much more expensive and even more so if the argument x is large in magnitude.

​        More precisely, the argument reduction code (see Mathematical Functions for implementation) comprises two code paths referred to as the fast path and the slow path, respectively.

​        The fast path is used for arguments sufficiently small in magnitude and essentially consists of a few multiply-add operations. The slow path is used for arguments large in magnitude and consists of lengthy computations required to achieve correct results over the entire argument range. 

​        At present, the argument reduction code for the trigonometric functions selects the fast path for arguments whose magnitude is less than 105615.0f for the single-precision functions, and less than 2147483648.0 for the double-precision functions.

​         As the slow path requires more registers than the fast path, an attempt has been made to reduce register pressure in the slow path by storing some intermediate variables in local memory, which may affect performance because of local memory high latency and bandwidth (see Device Memory Accesses). At present, 28 bytes of local memory are used by single-precision functions, and 44 bytes are used by double-precision functions. However, the exact amount is subject to change. 

​        Due to the lengthy computations and use of local memory in the slow path, the throughput of these trigonometric functions is lower by one order of magnitude when the slow path reduction is required as opposed to the fast path reduction.

#####  Integer Arithmetic

​        Integer division and modulo operation are costly as they compile to up to 20 instructions. They can be replaced with bitwise operations in some cases: If n is a power of 2, (i/n) is equivalent to (i>>log2(n)) and (i%n) is equivalent to (i&(n-1)); the compiler will perform these conversions if n is literal.

​        \_\_brev and \_\_popc map to a single instruction and \_\_brevll and \_\_popcll to a few instructions. 

​        \_\_[u]mul24 are legacy intrinsic functions that no longer have any reason to be used. 

##### Half Precision Arithmetic 

​        In order to achieve good half precision floating-point add, multiply or multiply-add throughput it is recommended that the half2 datatype is used. Vector intrinsics (eg. __hadd2, __hsub2, __hmul2, __hfma2) can then be used to do two operations in a single instruction. Using half2 in place of two calls using half may also help performance of other intrinsics, such as warp shuffles. 

​        The intrinsic __halves2half2 is provided to convert two half precision values to the half2 datatype.

##### Type Conversion 

​       Sometimes, the compiler must insert conversion instructions, introducing additional execution cycles. This is the case for: 

1、Functions operating on variables of type char or short whose operands generally need to be converted to int, 

2、Double-precision floating-point constants (i.e., those constants defined without any type suffix) used as input to single-precision floating-point computations (as mandated by C/C++ standards). 

​        This last case can be avoided by using single-precision floating-point constants, defined with an f suffix such as 3.141592653589793f, 1.0f, 0.5f. 

#### 5.4.2. Control Flow Instructions 

​        Any flow control instruction (if, switch, do, for, while) can significantly impact the effective instruction throughput by causing threads of the same warp to diverge (i.e., to follow different execution paths). If this happens, the different executions paths have to be serialized, increasing the total number of instructions executed for this warp. 

​        To obtain best performance in cases where the control flow depends on the thread ID, the controlling condition should be written so as to minimize the number of divergent warps. This is possible because the distribution of the warps across the block is deterministic as mentioned in SIMT Architecture. A trivial example is when the controlling condition only depends on (threadIdx / warpSize) where warpSize is the warp size. In this case, no warp diverges since the controlling condition is perfectly aligned with the warps. 

​        Sometimes, the compiler may unroll loops or it may optimize out short if or switch blocks by using branch predication instead, as detailed below. In these cases, no warp can ever diverge. The programmer can also control loop unrolling using the #pragma unroll directive (see #pragma unroll). 

​         When using branch predication none of the instructions whose execution depends on the controlling condition gets skipped. Instead, each of them is associated with a perthread condition code or predicate that is set to true or false based on the controlling condition and although each of these instructions gets scheduled for execution, only the instructions with a true predicate are actually executed. Instructions with a false predicate do not write results, and also do not evaluate addresses or read operands. 

​        5.4.3. Synchronization Instruction 

​        Throughput for \_\_syncthreads() is 128 operations per clock cycle for devices of compute capability 3.x, 32 operations per clock cycle for devices of compute capability 6.0, 16 operations per clock cycle for devices of compute capability 7.x and 64 operations per clock cycle for devices of compute capability 5.x, 6.1 and 6.2.

​        Note that __syncthreads() can impact performance by forcing the multiprocessor to idle as detailed in Device Memory Accesses. 











