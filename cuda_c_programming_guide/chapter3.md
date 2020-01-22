## chapter3 Programming Interface

### 3.2 CUDA Runtime

静态链接库 cudart.lib libcudart.a

动态链接库 cudart.dll cudart.so



page-locked host memory 页锁定的主机内存

asynchronous concurrent execution (异步并发执行)

multi-device system 单机多卡系统

error checking

call stack

texture and surface memory

graphics interoperability

#### 3.2.1 Initialization

        运行时没有显式的初始化函数,当地一个运行时函数被调用时直接就初始化了(除了错误处理和版本管理的函数).当计算运行时函数的运行时间和解释第一次调用运行时函数的错误码时必须时刻记住这一点.
    
        在初始化过程中,运行时在系统中为每个设备(gpu)创建了一个CUDA上下文.这个上下文只是这个设备的一个初始上下文,并且在应用的所有主机线程中共享.作为上下文创建的一部分,如果需要的话,设备的代码是实时编译的,并且加载到设备的内存中. 所有这一切全是透明的.如果需要,对于驱动API的互操作性(interoperability),设备的基础上下文能够从驱动的API中获取
    
       当一个主机线程调用cudaDeviceReset(), 这个操作销毁了主机线程当前操作的设备的初始上下文.下一次任何将这个设备作为current的主机线程调用运行时函数都会为这个设备创建一个新的初始上下文.

```
cuda接口使用的全局状态(global state)是在主机程序初始化时初始化的,在主程序终结时被销毁的.CUDA运行时和驱动不能检测到全局状态(global state)是否有效,所以在主程序初始化或者main函数结束后使用这些接口都会导致未定义行为
```

#### 3.2.2 Device Memory

         在异构编程中,CUDA编程模型假定系统由一个主机(host)和一个设备(device)组成, 而且每个都有自己独立的内存.核函数运行在设备内存中,所以运行时需要提供函数来分配,回收和复制设备的内存,同时在主机内存和设备内存之间传输数据.
    
         设备内存被分配成线性内存或者CUDA数组.
    
         CUDA数组的内存布局时不透明的,专门为了获取纹理做了优化. 这一点在章节3.2.11 Texture and Surface Memory中描述.
    
         线性内存是在单个统一的地址空间中分配的,这就意味着独立分配的实体能够通过指针来引用,例如二叉树和链表.地址空间的大小取决于主机(host)系统和GPU的compute capability

```
在compute capability 为5.3或者更早的显卡中,CUDA驱动创建一个未提交的40bit虚拟保留地址来保证分配的内存在支持的范围中.这个保留呈现为一个保留的虚拟地址,但是并不占用任何物理地址直到程序真正分配了内存
```

        线性存储器一般使用cudaMalloc()来分配内存, 使用cudaFree()来释放内存, 使用cudaMemcpy()来在主机内存和设备内存之间传输数据.在向量加的代码例子中,向量需要从主机内存拷贝到设备内存.

```c++
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] + B[i];
}
// Host code
int main()
{
	int N = ...;
	size_t size = N * sizeof(float);
	// Allocate input vectors h_A and h_B in host memory
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	// Initialize input vectors
	 ...
	// Allocate vectors in device memory
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);
	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	// Free host memory
	...
}
```

        线性存储器也可以通过cudaMallocPitch()和cudaMalloc3D()来分配.推荐使用这两个函数用来分配2D和3D的数组,因为这能确保分配的空间会通过padding来满足章节 5.3.2 Device memory Accesses中描述的字节对齐的要求, 也因此能保证在获取行地址或者使用别的API(cudaMemcpy2D和cudaMemcpy3D)来在2D数组和设备内存的其他区域之间复制数据时获得最好的性能.返回的pitch或者stride必须用来获取数组元素.
    
        以下代码是在设备中分配一个二维数组并且循环遍历数组元素的设备代码

```c++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);
// Device code
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
	for (int r = 0; r < height; ++r) {
		float* row = (float*)((char*)devPtr + r * pitch);
		for (int c = 0; c < width; ++c) {
			float element = row[c];
		}
	}
}
```

        以下代码是在设备中分配一个三维数组并且循环遍历数组元素的设备代码

```c++
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);
// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
int width, int height, int depth)
{
	char* devPtr = devPitchedPtr.ptr;
	size_t pitch = devPitchedPtr.pitch;
	size_t slicePitch = pitch * height;
	for (int z = 0; z < depth; ++z) {
		char* slice = devPtr + z * slicePitch;
		for (int y = 0; y < height; ++y) {
			float* row = (float*)(slice + y * pitch);
			for (int x = 0; x < width; ++x) {
			    float element = row[x];
            }             
        }
    }
}
```

    参考手册列举了...

下列代码列举通过运行时API访问全局变量的各种方式

```c++
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));
__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));
__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

#### 3.2.3 Shared Memory

​        共享存储器在分配时使用\_\_share\__存储空间限定符.

​        共享存储器比全局存储器的速度快得多(这一点在Thread HIerarchy中有提到).它可以用来作为scratchpad memory(或者software managed cache)来使得CUDA block对全局存储器的访问达到最低,后面有矩阵乘法的示例代码来阐述这个问题.

​       下面这部分代码时矩阵乘法的直接实现,并未使用shared_memory.每个线程读取A的一行和B的一列,计算C中对应的元素.因此,A会从全局存储器中被读取B.width次,同理,B也会被读取A.height次.

```c++
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
	cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
	cudaMemcpyHostToDevice);
	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	// Read C from device memory
	cudaMemcpy(C.elements, Cd.elements, size,
	cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}
```

​       下面这部分代码是使用了shared_memory的矩阵乘法实现.在这个实现中,每个线程块,负责计算一个C的一个子矩阵Csub,线程块的每个线程负责计算Csub中的一个元素.如图所示,Csub等于两个矩形矩阵的乘积;A的子矩阵的维度(A.width,block_size)和Csub的行索引数量相同,B的子矩阵的维度(block_size, B.height)和Csub的列索引数量相同.为了适配设备的资源,这两个矩形的矩阵分割成尽可能多的维度为(block_size, block_size)的方阵,并且Csub最后的结果是这些方阵的和.在计算这些方矩阵的乘积时, 第一步是从全局存储器中加载两个相关的方阵的数据,每个线程加载这些方阵中的一个元素, 然后每个线程计算一个元素的乘积.每个线程累加这些和的结果到寄存器, 而且最后写回到全局存储器中.

​        通过分块计算的方式, 我们使用了高速的共享存储器并且节省了大量的全局存储器带宽, 因为A从全局存储器中只被读取B.width/block_size次, B从全局存储器中只被读取了(A.height/block_size)次.

​        相较于之前的代码示例, 接下来的示例中, 矩阵类型增加了一个域stride,因此子矩阵能够高效的表示成相同的类型.\_\_device\_\_函数是用来从矩阵中获取和设置元素, 创建任何子矩阵. 

```c++
// Matrices are stored in row-major order:

// M(row, col) = *(M.elements + row * M.str
typedef struct {
	int width;
	int height;
	int stride; 
	float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
{
	A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}

// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = d_B.stride = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);

	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	float Cvalue = 0;
	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;
	// Loop over all the sub-matrices of A and B that are
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		// Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		// Get sub-matrix Bsub of B
		Matrix Bsub = GetSubMatrix(B, m, blockCol);
		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();

		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e)
		Cvalue += As[row][e] * Bs[e][col];
		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	// Write Csub to device memory
	// Each thread writes one element
	SetElement(Csub, row, col, Cvalue);
}
```

#### 3.2.4 Page-Locked Host Memory (页锁定主机内存)

​        运行时提供了函数来允许使用页锁定主机内存(也叫pinned memory), 这是相对于malloc()分配的regular pageable host memory(可分页内存)

​        1、cudaHostAlloc()和cudaFreeHost()分配和释放页锁定主机内存

​        2、cudaHostRegister() 页锁定由malloc()分配的内存

​         使用页锁定内存有以下几个好处:

​         1、在页锁定内存和设备内存之间复制数据(Copies between page-locked host memory and device memory can be performed concurrently with kernel execution for some devices )

​          2、在某些设备上,页锁定内存可以映射到设备的地址空间上,这样可以省略往设备存储器读或者写的时间

​          3、在一个有前端总线(front-side bus)的系统上, 主机存储器和设备存储器之间的带宽会更高,如果主机存储器是页锁定的,甚至还要高些如果这些页锁定内存被分配成write-combining

​         页锁定存储器是一项稀缺的资源, 如果持续分配页锁定存储器, 那么后续继续分配存储器可能会失败.另外,页锁定存储器减少了操作系统可用于分页的物理存储器,如果可分页存储器消耗太多会导致整体系统的性能下降. 

```
Page-locked host memory is not cached on non I/O coherent Tegra devices. Also, cudaHostRegister() is not supported on non I/O coherent Tegra devices.
```

##### 3.2.4.1 Portable Memory

​        页锁定存储器的块能用来连接系统中的任意设备,但是默认情况下,页锁定存储器带来的性能优势只适用于当页锁定存储器块分配时,当前正在使用的设备(the device that was current).为了使得所有的设备获取这种性能优势,存储器块在分配的时候需要传入flag cudaHostAllocPortable 给cudaHostAlloc()或者通过传入cudaHostRegister() flag cudaHostRegisterPortable

##### 3.2.4.2 Write-Combining Memory

​        默认情况下页锁定内存分配的时候是可缓存的(cacheable).它可以选择性的分配成write-combining通过给cu daHostAlloc()传入flag cudahostAllocWriteCombined.Write-combining存储器不占用主机的L1缓存和L2缓存,可以留给给多缓存给其他应用.In addition, write-combining memory is not snooped during transfers across the PCI Express bus, which can improve transfer performance by up to 40%.

从host 的 write-combining存储器读取数据不可避免的慢,所以write-combining存储器通常应该只让主机写

##### 3.2.4.3 Mapped Memory

​        通过给cudaHostAlloc()传入flag cudaHostAllocMapped(或者把cudaHostRegisterMapped传入cudaHostRegister()), 页锁定存储器块也可以映射到设备存储器的地址空间. 因此这样的存储器块通常有两个地址:一个是主机存储器地址,通常是malloc()或者cudaHostAlloc()返回的, 另一个是设备存储器的地址, 可以通过cudaHostGetDevicePointer()获取. 然后这个地址可以在kernel函数中用来访问存储器块.  The only exception is for pointers allocated with cudaHostAlloc() and when a unified address space is used for the host and the device as mentioned in **Unified Virtual Address Space**.

​         设备通过kernel函数直接访问主机存储器的带宽和直接访问设备存储器不同,但是有一些优势:

​         1、没有必要在设备存储器中分配一个块,并且在这个存储器块和主机存储器块之间复制数据,数据传输只在和函数需要的时候隐式执行

​         2、There is no need to use streams (see Concurrent Data Transfers) to overlap data transfers with kernel execution; the kernel-originated data transfers automatically overlap with kernel execution

​        既然映射的页锁定存储器在主机和设备之间共享,但是,应用必须使用steams和events来同步存储器访问来避免任何潜在的read-after-write, write-after-read或者write-after-write的风险.

​        To be able to retrieve the device pointer to any mapped page-locked memory, page-locked memory mapping must be enabled by calling **cudaSetDeviceFlags**() with the *cudaDeviceMapHost* flag before any other CUDA call is performed. Otherwise, **cudaHostGetDevicePointer**() will return an error.
​        **cudaHostGetDevicePointer**() also returns an error if the device does not support mapped page-locked host memory. Applications may query this capability by checking the canMapHostMemory device property (see Device Enumeration), which is equal to 1 for devices that support mapped page-locked host memory.
​        Note that atomic functions (see Atomic Functions) operating on mapped page-locked memory are not atomic from the point of view of the host or other devices. 
​        Also note that CUDA runtime requires that 1-byte, 2-byte, 4-byte, and 8-byte naturally aligned loads and stores to host memory initiated from the device are preserved as single accesses from the point of view of the host and other devices. On some platforms, atomics to memory may be broken by the hardware into separate load and store operations. These component load and store operations have the same requirements on preservation of naturally aligned accesses. As an example, the CUDA runtime does not support a PCI Express bus topology where a PCI Express bridge splits 8-byte naturally aligned writes into two 4-byte writes between the device and the host.

#### 3.2.5 Asynchronous Concurrent Execution

​        CUDA exposes the following operations as independent tasks that can operate
concurrently with one another:(可同时进行的操作)
1、Computation on the host;
2、Computation on the device;
3、Memory transfers from the host to the device;
4、Memory transfers from the device to the host;
5、Memory transfers within the memory of a given device;
6、Memory transfers among devices.
​        The level of concurrency achieved between these operations will depend on the feature
set and compute capability of the device as described below.

##### 3.2.5.1. Concurrent Execution between Host and Device

​        Concurrent host execution is facilitated through asynchronous(异步) library functions that return control to the host thread before the device completes the requested task. Using asynchronous calls, many device operations can be queued up together to be executed by the CUDA driver when appropriate device resources are available. This relieves the host thread of much of the responsibility to manage the device, leaving it free for other tasks. The following device operations are asynchronous with respect to the host:
1、Kernel launches;
2、Memory copies within a single device's memory;
3、Memory copies from host to device of a memory block of 64 KB or less;
4、Memory copies performed by functions that are suffixed with Async;
5、Memory set function calls.
​        Programmers can globally <u>disable</u> asynchronicity of kernel launches for all CUDA
applications running on a system by setting the **CUDA_LAUNCH_BLOCKING** environment
variable to 1. This feature is provided for debugging purposes only and <u>should not be</u>
<u>used as a way to make production software</u> run reliably.
​        Kernel launches are synchronous if hardware counters are collected via a profiler (Nsight, Visual Profiler) unless concurrent kernel profiling is enabled. Async memory copies will also be synchronous if they involve host memory that is not page-locked.

##### 3.2.5.2. Concurrent Kernel Execution

​        Some devices of compute capability 2.x and higher can execute multiple kernels concurrently. Applications may query this capability by checking the **concurrentKernels** device property (see Device Enumeration), which is equal to 1 for devices that support it.

​        The maximum number of kernel launches that a device can execute concurrently
depends on its compute capability and is listed in Table 15.

​       A kernel from one CUDA context cannot execute concurrently with a kernel from
another CUDA context.

​      Kernels that use many textures or a large amount of local memory are less likely to execute concurrently with other kernels.

##### 3.2.5.3. Overlap of Data Transfer and Kernel Execution

​        Some devices can perform an asynchronous memory copy to or from the GPU concurrently with kernel execution. Applications may query this capability by checking the **asyncEngineCount** device property (see Device Enumeration), which is greater than zero for devices that support it. If host memory is involved in the copy, it must be page-locked. (通过查询asyncEngineCount来看GPU是不是支持同时从kernel中异步并发拷贝内存)

​        It is also possible to perform an intra-device copy simultaneously with kernel execution (on devices that support the concurrentKernels device property) and/or with copies to or from the device (for devices that support the asyncEngineCount property). Intra-device copies are initiated using the standard memory copy functions with destination and source addresses residing on the same device.

##### 3.2.5.4. Concurrent Data Transfers

​        Some devices of compute capability 2.x and higher can overlap copies to and from the device. Applications may query this capability by checking the **asyncEngineCount** device property (see Device Enumeration), which is equal to 2 for devices that support it. In order to be overlapped, any host memory involved in the transfers must be page-locked.

##### 3.2.5.5. Streams

​        Applications manage the concurrent operations described above through streams. (应用通过streams来管理并发的操作的). A stream is a sequence of commands (possibly issued by different host threads) that execute in order. (一个stream是一个命令队列)Different streams, on the other hand, may execute their commands out of order with respect to one another or concurrently; this behavior is not guaranteed and should therefore not be relied upon for correctness (e.g., inter-kernel communication is undefined). The commands issued on a stream may execute when all the dependencies of the command are met. The dependencies could be previously launched commands on same stream or dependencies from other streams. The successful completion of synchronize call guarantees that all the commands launched are completed.

###### 3.2.5.5.1. Creation and Destruction

​        A stream is defined by creating a stream object and specifying it as the stream parameter to a sequence of kernel launches and host <-> device memory copies. The following code sample creates two streams and allocates an array hostPtr of float in page-locked memory

```c++
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);
```

​        Each of these streams is defined by the following code sample as a sequence of one memory copy from host to device, one kernel launch, and one memory copy from device to host:

```c++
for (int i = 0; i < 2; ++i) {
	cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
	MyKernel <<<100, 512, 0, stream[i]>>>(outputDevPtr + i * size, inputDevPtr + i * size, size);
	cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
}
```

Each stream copies its portion of input array hostPtr to array inputDevPtr in device memory, processes inputDevPtr on the device by calling MyKernel(), and copies the result outputDevPtr back to the same portion of hostPtr. Overlapping Behavior describes how the streams overlap in this example depending on the capability of the device. Note that hostPtr must point to page-locked host memory for any overlap to occur.

Streams are released by calling cudaStreamDestroy().

```c++
for (int i = 0; i < 2; ++i)
 cudaStreamDestroy(stream[i]);
```

​        In case the device is still doing work in the stream when cudaStreamDestroy() is called, the function will return immediately and the resources associated with the stream will be released automatically once the device has completed all work in the stream.

###### 3.2.5.5.2. Default Stream

​        Kernel launches and host <-> device memory copies that do not specify any stream parameter, or equivalently that set the stream parameter to zero, are issued to the default stream. They are therefore executed in order. 
​        For code that is compiled using the --default-stream per-thread compilation flag(or that defines the CUDA_API_PER_THREAD_DEFAULT_STREAM macro before including CUDA headers (cuda.h and cuda_runtime.h)), the default stream is a regular stream and each host thread has its own default stream.

```c++
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1 cannot be used to
enable this behavior when the code is compiled by nvcc as nvcc implicitly
includes cuda_runtime.h at the top of the translation unit. In this case the
--default-stream per-thread compilation flag needs to be used or the
CUDA_API_PER_THREAD_DEFAULT_STREAM macro needs to be defined with the -
DCUDA_API_PER_THREAD_DEFAULT_STREAM=1 compiler flag.
```

​        For code that is compiled using the --default-stream legacy compilation flag, the default stream is a special stream called the NULL stream and each device has a single NULL stream used for all host threads. The NULL stream is special as it causes implicit synchronization as described in Implicit Synchronization.

​        For code that is compiled without specifying a --default-stream compilation flag, -- default-stream legacy is assumed as the default.

###### 3.2.5.5.3. Explicit Synchronization

​        There are various ways to explicitly synchronize streams with each other.

​        **cudaDeviceSynchronize**() waits until all preceding commands in all streams of all host threads have completed. 

​        **cudaStreamSynchronize**() takes a stream as a parameter and waits until all preceding commands in the given stream have completed. It can be used to synchronize the host with a specific stream, allowing other streams to continue executing on the device. 

​        **cudaStreamWaitEvent**() takes a stream and an event as parameters (see Events for a description of events) and makes all the commands added to the given stream after the call to cudaStreamWaitEvent() delay their execution until the given event has completed. 

​        **cudaStreamQuery**() provides applications with a way to know if all preceding commands in a stream have completed.

###### 3.2.5.5.4. Implicit Synchronization

​        Two commands from different streams cannot run concurrently if any one of the following operations is issued in-between them by the host thread: 

​        1、a page-locked host memory allocation, 

​        2、a device memory allocation,

​        3、a device memory set, 

​        4、a memory copy between two addresses to the same device memory, 

​        5、any CUDA command to the NULL stream, 

​        6、a switch between the L1/shared memory configurations described in Compute Capability 3.x and Compute Capability 7.x. 

​        For devices that support concurrent kernel execution and are of compute capability 3.0 or lower, any operation that requires a dependency check to see if a streamed kernel launch is complete:
​        1、Can start executing only when all thread blocks of all prior kernel launches from any stream in the CUDA context have started executing;
​        2、Blocks all later kernel launches from any stream in the CUDA context until the kernel launch being checked is complete.

​        Operations that require a dependency check include any other commands within the same stream as the launch being checked and any call to cudaStreamQuery() on that stream. Therefore, applications should follow these guidelines to improve their potential for concurrent kernel execution:
​        1、All independent operations should be issued before dependent operations,
​        2、Synchronization of any kind should be delayed as long as possible.





































