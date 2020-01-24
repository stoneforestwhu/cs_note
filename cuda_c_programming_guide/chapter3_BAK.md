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

###### 3.2.5.5.5 Overlapping Behavior

​         The amount of execution overlap between two streams depends on the order in which the commands are issued to each stream and whether or not the device supports overlap of data transfer and kernel execution (see Overlap of Data Transfer and Kernel Execution), concurrent kernel execution (see Concurrent Kernel Execution), and/or concurrent data transfers (see Concurrent Data Transfers).

​         For example, on devices that do not support concurrent data transfers, the two streams of the code sample of Creation and Destruction do not overlap at all because the memory copy from host to device is issued to stream[1] after the memory copy from device to host is issued to stream[0], so it can only start once the memory copy from device to host issued to stream[0] has completed. If the code is rewritten the following way (and assuming the device supports overlap of data transfer and kernel execution)

```c++
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
```

​        then the memory copy from host to device issued to stream[1] overlaps with the kernel launch issued to stream[0].
​        On devices that do support concurrent data transfers, the two streams of the code sample of Creation and Destruction do overlap: The memory copy from host to device issued to stream[1] overlaps with the memory copy from device to host issued to stream[0] and even with the kernel launch issued to stream[0] (assuming the device supports overlap of data transfer and kernel execution). However, for devices of compute capability 3.0 or lower, the kernel executions cannot possibly overlap because the second kernel launch is issued to stream[1] after the memory copy from device to host is issued to stream[0], so it is blocked until the first kernel launch issued to stream[0] is complete as per Implicit Synchronization. If the code is rewritten as above, the kernel executions overlap (assuming the device supports concurrent kernel execution) since the second kernel launch is issued to stream[1] before the memory copy from device to host is issued to stream[0]. In that case however, the memory copy from device to host issued to stream[0] only overlaps with the last thread blocks of the kernel launch issued to stream[1] as per Implicit Synchronization, which can represent only a small portion of the total execution time of the kernel.

###### 3.2.5.5.6 Host Functions (Callbacks)

​        The runtime provides a way to insert a CPU function call at any point into a stream via cudaLaunchHostFunc(). The provided function is executed on the host once all commands issued to the stream before the callback have completed. The following code sample adds the host function MyCallback to each of two streams after issuing a host-to-device memory copy, a kernel launch and a device-to-host

memory copy into each stream. The function will begin execution on the host after each of the device-to-host memory copies completes. 

```c++
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, 
void *data){ 
    printf("Inside callback %d\n", (size_t)data); } 
...
for (size_t i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size, cudaMemcpyHostToDevice, stream[i]);       MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
    cudaLaunchHostFunc(stream[i], MyCallback, (void*)i); 
} 
```

​         The commands that are issued in a stream after a host function do not start executing before the function has completed. 
​          A host function enqueued into a stream must not make CUDA API calls (directly or indirectly), as it might end up waiting on itself if it makes such a call leading to a deadlock.  

###### 3.2.5.5.7. Stream Priorities 

​         The relative priorities of streams can be specified at creation using cudaStreamCreateWithPriority(). The range of allowable priorities, ordered as [ highest priority, lowest priority ] can be obtained using the cudaDeviceGetStreamPriorityRange() function. At runtime, pending work in higher-priority streams takes preference over pending work in low-priority streams. 
​         The following code sample obtains the allowable range of priorities for the current device, and creates streams with the highest and lowest available priorities. 

```C++
// get the range of stream priorities for this device 
int priority_high, priority_low; 
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high); 
// create streams with highest and lowest available priorities 
cudaStream_t st_high, st_low; 
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high); 
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
```

##### 3.2.5.6. Graphs 

​        Graphs present a new model for work submission in CUDA. A graph is a series of operations, such as kernel launches, connected by dependencies, which is defined separately from its execution. This allows a graph to be defined once and then launched repeatedly. Separating out the definition of a graph from its execution enables a number of optimizations: first, CPU launch costs are reduced compared to streams, because much of the setup is done in advance; second, presenting the whole workflow to CUDA enables optimizations which might not be possible with the piecewise work submission mechanism of streams. 
​        To see the optimizations possible with graphs, consider what happens in a stream: when you place a kernel into a stream, the host driver performs a sequence of operations in preparation for the execution of the kernel on the GPU. These operations, necessary for setting up and launching the kernel, are an overhead cost which must be paid for each kernel that is issued. For a GPU kernel with a short execution time, this overhead cost can be a significant fraction of the overall end-to-end execution time.

Work submission using graphs is separated into three distinct stages: definition, instantiation, and execution.
1、During the definition phase, a program creates a description of the operations in the graph along with the dependencies between them.
2、Instantiation takes a snapshot of the graph template, validates it, and performs much of the setup and initialization of work with the aim of minimizing what needs to be done at launch. The resulting instance is known as an executable graph.
3、 An executable graph may be launched into a stream, similar to any other CUDA work. It may be launched any number of times without repeating the instantiation.

###### 3.2.5.6.1. Graph Structure

An operation forms a node in a graph. The dependencies between the operations are the edges. These dependencies constrain the execution sequence of the operations.
An operation may be scheduled at any time once the nodes on which it depends are complete. Scheduling is left up to the CUDA system.

3.2.5.6.1.1 Node Types

A graph node can be one of:
1、kernel
2、CPU function call
3、memory copy
4、memset
5、empty node
6、child graph: To execute a separate nested graph. See Figure 11.

###### 3.2.5.6.2. Creating a Graph Using Graph APIs

​         Graphs can be created via two mechanisms: explicit API and stream capture. The following is an example of creating and executing the below graph.

```c++
// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);
// For the purpose of this example, we'll create
// the nodes separately from the dependencies to
// demonstrate that it can be done in two stages.
// Note that dependencies can also be specified
// at node creation.
cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);
// Now set up dependencies on each node
cudaGraphAddDependencies(graph, &a, &b, 1); // A->B
cudaGraphAddDependencies(graph, &a, &c, 1); // A->C
cudaGraphAddDependencies(graph, &b, &d, 1); // B->D
cudaGraphAddDependencies(graph, &c, &d, 1); // C->D
```

###### 3.2.5.6.3. Creating a Graph Using Stream Capture

​        Stream capture provides a mechanism to create a graph from existing streambased APIs. A section of code which launches work into streams, including existing code, can be bracketed with calls to cudaStreamBeginCapture() and cudaStreamEndCapture(). See below

```c++
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);
cudaStreamEndCapture(stream, &graph);
```

​         A call to cudaStreamBeginCapture() places a stream in capture mode. When a stream is being captured, work launched into the stream is not enqueued for execution. It is instead appended to an internal graph that is progressively being built up. This graph is then returned by calling cudaStreamEndCapture(), which also ends capture mode for the stream. A graph which is actively being constructed by stream capture is referred to as a capture graph. 

​        Stream capture can be used on any CUDA stream except cudaStreamLegacy (the "NULL stream"). Note that it can be used on cudaStreamPerThread. If a program is using the legacy stream, it may be possible to redefine stream 0 to be the per-thread stream with no functional change. See Default Stream. 

​        Whether a stream is being captured can be queried with cudaStreamIsCapturing(). 

3.2.5.6.3.1. Cross-stream Dependencies and Events 

​        Stream capture can handle cross-stream dependencies expressed with cudaEventRecord() and cudaStreamWaitEvent(), provided the event being waited upon was recorded into the same capture graph. 

​        When an event is recorded in a stream that is in capture mode, it results in a captured event. A captured event represents a set of nodes in a capture graph. 

​       When a captured event is waited on by a stream, it places the stream in capture mode if it is not already, and the next item in the stream will have additional dependencies on the nodes in the captured event. The two streams are then being captured to the same capture graph.

​       When cross-stream dependencies are present in stream capture, cudaStreamEndCapture() must still be called in the same stream where cudaStreamBeginCapture() was called; this is the origin stream. Any other streams which are being captured to the same capture graph, due to event-based dependencies, must also be joined back to the origin stream. This is illustrated below. All streams being captured to the same capture graph are taken out of capture mode upon cudaStreamEndCapture(). Failure to rejoin to the origin stream will result in failure of the overall capture operation.             

```c++
// stream1 is the origin stream
cudaStreamBeginCapture(stream1);
kernel_A<<< ..., stream1 >>>(...);
// Fork into stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);
kernel_B<<< ..., stream1 >>>(...);
kernel_C<<< ..., stream2 >>>(...);
// Join stream2 back to origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);
kernel_D<<< ..., stream1 >>>(...);
// End capture in the origin stream
cudaStreamEndCapture(stream1, &graph);
// stream1 and stream2 no longer in capture mode
```

​        Graph returned by the above code is shown in Figure 12.

```
When a stream is taken out of capture mode, the next non-captured item in the stream (if any) will still have a dependency on the most recent prior non-captured item, despite intermediate items having been removed.
```

3.2.5.6.3.2. Prohibited and Unhandled Operations

​        It is invalid to synchronize or query the execution status of a stream which is being captured or a captured event, because they do not represent items scheduled for execution. It is also invalid to query the execution status of or synchronize a broader handle which encompasses an active stream capture, such as a device or context handle when any associated stream is in capture mode.

​        When any stream in the same context is being captured, and it was not created with cudaStreamNonBlocking, any attempted use of the legacy stream is invalid. This is because the legacy stream handle at all times encompasses these other streams; enqueueing to the legacy stream would create a dependency on the streams being captured, and querying it or synchronizing it would query or synchronize the streams being captured. 

​        It is therefore also invalid to call synchronous APIs in this case. Synchronous APIs, such as cudaMemcpy(), enqueue work to the legacy stream and synchronize it before returning. 

```
       As a general rule, when a dependency relation would connect something that is captured with something that was not captured and instead enqueued for execution, CUDA prefers to return an error rather than ignore the dependency. An exception is made for placing a stream into or out of capture mode; this severs a dependency relation between items added to the stream immediately before and after the mode transition. 
```

​        It is invalid to merge two separate capture graphs by waiting on a captured event from a stream which is being captured and is associated with a different capture graph than the event. It is invalid to wait on a non-captured event from a stream which is being captured. 

​        A small number of APIs that enqueue asynchronous operations into streams are not currently supported in graphs and will return an error if called with a stream which is being captured, such as cudaStreamAttachMemAsync(). 

3.2.5.6.3.3. Invalidation

​        When an invalid operation is attempted during stream capture, any associated capture graphs are invalidated. When a capture graph is invalidated, further use of any streams which are being captured or captured events associated with the graph is invalid and will return an error, until stream capture is ended with cudaStreamEndCapture(). This call will take the associated streams out of capture mode, but will also return an error value and a NULL graph. 

###### 3.2.5.6.4. Using Graph APIs 

​        cudaGraph_t objects are not thread-safe. It is the responsibility of the user to ensure that multiple threads do not concurrently access the same cudaGraph_t. 

​        A cudaGraphExec_t cannot run concurrently with itself. A launch of a cudaGraphExec_t will be ordered after previous launches of the same executable graph.

​        Graph execution is done in streams for ordering with other asynchronous work. However, the stream is for ordering only; it does not constrain the internal parallelism of the graph, nor does it affect where graph nodes execute.

​        See Graph API. 

##### 3.2.5.7. Events 

​        The runtime also provides a way to closely monitor the device's progress, as well as perform accurate timing, by letting the application asynchronously record events at any point in the program, and query when these events are completed. An event has completed when all tasks - or optionally, all commands in a given stream - preceding the event have completed. Events in stream zero are completed after all preceding tasks and commands in all streams are completed.

###### 3.2.5.7.1. Creation and Destruction 

​        The following code sample creates two events: 

```c++
cudaEvent_t start, stop; 
cudaEventCreate(&start); 
cudaEventCreate(&stop); 
```

They are destroyed this way: 

```c++
cudaEventDestroy(start); 
cudaEventDestroy(stop); 
```

###### 3.2.5.7.2. Elapsed Time

​        The events created in Creation and Destruction can be used to time the code sample of Creation and Destruction the following way:  

```c++
cudaEventRecord(start, 0);
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDev + i * size, inputHost + i * size, size,                                         cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>> (outputDev + i * size, inputDev + i * size,                                                size);
    cudaMemcpyAsync(outputHost + i * size, outputDev + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
```

3.2.5.8. Synchronous Calls

 When a synchronous function is called, control is not returned to the host thread before the device has completed the requested task. Whether the host thread will then yield, block, or spin can be specified by calling cudaSetDeviceFlags()with some specific flags (see reference manual for details) before any other CUDA call is performed by the host thread. 

3.2.6. Multi-Device System 

3.2.6.1. Device Enumeration 

​        A host system can have multiple devices. The following code sample shows how to enumerate these devices, query their properties, and determine the number of CUDAenabled devices. 

```c++
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	printf("Device %d has compute capability %d.%d.\n",
	device, deviceProp.major, deviceProp.minor);
}
```

3.2.6.2. Device Selection
         A host thread can set the device it operates on at any time by calling cudaSetDevice(). Device memory allocations and kernel launches are made on the currently set device; streams and events are created in association with the currently set device. If no call to cudaSetDevice() is made, the current device is device 0.

​        The following code sample illustrates how setting the current device affects memory allocation and kernel execution.

```c++
size_t size = 1024 * sizeof(float);
cudaSetDevice(0); // Set device 0 as current
float* p0;
cudaMalloc(&p0, size); // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0
cudaSetDevice(1); // Set device 1 as current
float* p1;
cudaMalloc(&p1, size); // Allocate memory on device 1
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
```

3.2.6.3. Stream and Event Behavior
        A kernel launch will fail if it is issued to a stream that is not associated to the current device as illustrated in the following code sample.

```c++
cudaSetDevice(0); // Set device 0 as current
cudaStream_t s0;
cudaStreamCreate(&s0); // Create stream s0 on device 0
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 0 in s0
cudaSetDevice(1); // Set device 1 as current
cudaStream_t s1;
cudaStreamCreate(&s1); // Create stream s1 on device 1
MyKernel<<<100, 64, 0, s1>>>(); // Launch kernel on device 1 in s1
// This kernel launch will fail:
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 1 in s0
```

​        A memory copy will succeed even if it is issued to a stream that is not associated to the
current device.
​        cudaEventRecord() will fail if the input event and input stream are associated to different devices.
​        cudaEventElapsedTime() will fail if the two input events are associated to different devices.
​        cudaEventSynchronize() and cudaEventQuery() will succeed even if the input event is associated to a device that is different from the current device.
​        cudaStreamWaitEvent() will succeed even if the input stream and input event are associated to different devices. cudaStreamWaitEvent() can therefore be used to synchronize multiple devices with each other.
​         Each device has its own default stream (see Default Stream), so commands issued to the default stream of a device may execute out of order or concurrently with respect to commands issued to the default stream of any other device.

3.2.6.4. Peer-to-Peer Memory Access

​        Depending on the system properties, specifically the PCIe and/or NVLINK topology, devices are able to address each other's memory (i.e., a kernel executing on one device can dereference a pointer to the memory of the other device). This peer-to-peer memory access feature is supported between two devices if cudaDeviceCanAccessPeer() returns true for these two devices. 

​        Peer-to-peer memory access is only supported in 64-bit applications and must be enabled between two devices by calling cudaDeviceEnablePeerAccess() as illustrated in the following code sample. On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections.

​        A unified address space is used for both devices (see Unified Virtual Address Space), so the same pointer can be used to address memory from both devices as shown in the code sample below. 

```c++
cudaSetDevice(0); // Set device 0 as current 
float* p0; 
size_t size = 1024 * sizeof(float); 
cudaMalloc(&p0, size); // Allocate memory on device 0 
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0 
cudaSetDevice(1); // Set device 1 as current 
cudaDeviceEnablePeerAccess(0, 0); // Enable peer-to-peer access 
                                  // with device 0 
// Launch kernel on device 1 
// This kernel launch can access memory on device 0 at address p0 
MyKernel<<<1000, 128>>>(p0); 
```

3.2.6.4.1. IOMMU on Linux 

​        On Linux only, CUDA and the display driver does not support IOMMU-enabled bare-metal PCIe peer to peer memory copy. However, CUDA and the display driver does support IOMMU via VM pass through. As a consequence, users on Linux, when running on a native bare metal system, should disable the IOMMU. The IOMMU should be enabled and the VFIO driver be used as a PCIe pass through for virtual machines.

​        On Windows the above limitation does not exist. 

​        See also Allocating DMA Buffers on 64-bit Platforms. 

3.2.6.5. Peer-to-Peer Memory Copy 

​        Memory copies can be performed between the memories of two different devices. 

​        When a unified address space is used for both devices (see Unified Virtual Address Space), this is done using the regular memory copy functions mentioned in Device Memory.

​        Otherwise, this is done using cudaMemcpyPeer(), cudaMemcpyPeerAsync(), cudaMemcpy3DPeer(), or cudaMemcpy3DPeerAsync() as illustrated in the following code sample.

```c++
cudaSetDevice(0); // Set device 0 as current 
float* p0; 
size_t size = 1024 * sizeof(float); 
cudaMalloc(&p0, size); // Allocate memory on device 0
cudaSetDevice(1); // Set device 1 as current float* p1; 
cudaMalloc(&p1, size); // Allocate memory on device 1 
cudaSetDevice(0); // Set device 0 as current 
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0 
cudaSetDevice(1); // Set device 1 as current 
cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1 
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1 
```

​        A copy (in the implicit NULL stream) between the memories of two different devices:   

```
1、does not start until all commands previously issued to either device have completed and（缺点什么？？？？？？？） 
2、runs to completion before any commands (see Asynchronous Concurrent Execution) issued after the copy to either device can start.
```

​        Consistent with the normal behavior of streams, an asynchronous copy between the memories of two devices may overlap with copies or kernels in another stream. 

​        Note that if peer-to-peer access is enabled between two devices via cudaDeviceEnablePeerAccess() as described in Peer-to-Peer Memory Access, peerto-peer memory copy between these two devices no longer needs to be staged through the host and is therefore faster. 

3.2.7. Unified Virtual Address Space 

​         When the application is run as a 64-bit process, a single address space is used for the host and all the devices of compute capability 2.0 and higher. All host memory allocations made via CUDA API calls and all device memory allocations on supported devices are within this virtual address range. As a consequence: 

1、The location of any memory on the host allocated through CUDA, or on any of the devices which use the unified address space, can be determined from the value of the pointer using cudaPointerGetAttributes(). 

2、When copying to or from the memory of any device which uses the unified address space, the cudaMemcpyKind parameter of cudaMemcpy*() can be set to cudaMemcpyDefault to determine locations from the pointers. This also works for host pointers not allocated through CUDA, as long as the current device uses unified addressing. 

3、Allocations via cudaHostAlloc() are automatically portable (see Portable Memory) across all the devices for which the unified address space is used, and pointers returned by cudaHostAlloc() can be used directly from within kernels running on these devices (i.e., there is no need to obtain a device pointer via cudaHostGetDevicePointer() as described in Mapped Memory. 

​         Applications may query if the unified address space is used for a particular device by checking that the unifiedAddressing device property (see Device Enumeration) is equal to 1. 

3.2.8. Interprocess Communication 

​        Any device memory pointer or event handle created by a host thread can be directly referenced by any other thread within the same process. It is not valid outside this process however, and therefore cannot be directly referenced by threads belonging to a different process.

​         To share device memory pointers and events across processes, an application must use the Inter Process Communication API, which is described in detail in the reference manual. The IPC API is only supported for 64-bit processes on Linux and for devices of compute capability 2.0 and higher. Note that the IPC API is not supported for cudaMallocManaged allocations. 

​         Using this API, an application can get the IPC handle for a given device memory pointer using cudaIpcGetMemHandle(), pass it to another process using standard IPC mechanisms (e.g., interprocess shared memory or files), and use cudaIpcOpenMemHandle() to retrieve a device pointer from the IPC handle that is a valid pointer within this other process. Event handles can be shared using similar entry points. 

​        An example of using the IPC API is where a single master process generates a batch of input data, making the data available to multiple slave processes without requiring regeneration or copying.    

​         Applications using CUDA IPC to communicate with each other should be compiled, linked, and run with the same CUDA driver and runtime. 

```
CUDA IPC calls are not supported on Tegra devices.
```

3.2.9. Error Checking 

​        All runtime functions return an error code, but for an asynchronous function (see Asynchronous Concurrent Execution), this error code cannot possibly report any of the asynchronous errors that could occur on the device since the function returns before the device has completed the task; the error code only reports errors that occur on the host prior to executing the task, typically related to parameter validation; if an asynchronous error occurs, it will be reported by some subsequent unrelated runtime function call. 

​        The only way to check for asynchronous errors just after some asynchronous function call is therefore to synchronize just after the call by calling cudaDeviceSynchronize() (or by using any other synchronization mechanisms described in Asynchronous Concurrent Execution) and checking the error code returned by cudaDeviceSynchronize(). 

​        The runtime maintains an error variable for each host thread that is initialized to cudaSuccess and is overwritten by the error code every time an error occurs (be it a parameter validation error or an asynchronous error). cudaPeekAtLastError() returns this variable. cudaGetLastError() returns this variable and resets it to cudaSuccess.

​        Kernel launches do not return any error code, so cudaPeekAtLastError() or cudaGetLastError() must be called just after the kernel launch to retrieve any pre-launch errors. To ensure that any error returned by cudaPeekAtLastError() or cudaGetLastError() does not originate from calls prior to the kernel launch, one has to make sure that the runtime error variable is set to cudaSuccess just before the kernel launch, for example, by calling cudaGetLastError() just before the kernel launch. Kernel launches are asynchronous, so to check for asynchronous errors, the application must synchronize in-between the kernel launch and the call to cudaPeekAtLastError() or cudaGetLastError().

​        Note that cudaErrorNotReady that may be returned by cudaStreamQuery() and cudaEventQuery() is not considered an error and is therefore not reported by cudaPeekAtLastError() or cudaGetLastError(). 

3.2.10. Call Stack 

​        On devices of compute capability 2.x and higher, the size of the call stack can be queried using cudaDeviceGetLimit() and set using cudaDeviceSetLimit(). 

​        When the call stack overflows, the kernel call fails with a stack overflow error if the application is run via a CUDA debugger (cuda-gdb, Nsight) or an unspecified launch error, otherwise. 

3.2.11. Texture and Surface Memory 

​        CUDA supports a subset of the texturing hardware that the GPU uses for graphics to access texture and surface memory. Reading data from texture or surface memory instead of global memory can have several performance benefits as described in Device Memory Accesses. 

​        There are two different APIs to access texture and surface memory: 

1、The texture reference API that is supported on all devices, 

2、The texture object API that is only supported on devices of compute capability 3.x and higher. 

​        The texture reference API has limitations that the texture object API does not have. They are mentioned in Texture Reference API. 

3.2.11.1. Texture Memory 

​        Texture memory is read from kernels using the device functions described in Texture Functions. The process of reading a texture calling one of these functions is called a texture fetch. Each texture fetch specifies a parameter called a texture object for the texture object API or a texture reference for the texture reference API. 

​        The texture object or the texture reference specifies:

1、The texture, which is the piece of texture memory that is fetched. Texture objects are created at runtime and the texture is specified when creating the texture object as described in Texture Object API. Texture references are created at compile time and the texture is specified at runtime by bounding the texture reference to the texture through runtime functions as described in Texture Reference API; several distinct texture references might be bound to the same texture or to textures that overlap in memory. A texture can be any region of linear memory or a CUDA array (described in CUDA Arrays).

2、Its dimensionality that specifies whether the texture is addressed as a one dimensional array using one texture coordinate, a two-dimensional array using two texture coordinates, or a three-dimensional array using three texture coordinates. Elements of the array are called texels, short for texture elements. The texture width, height, and depth refer to the size of the array in each dimension. Table 15 lists the maximum texture width, height, and depth depending on the compute capability of the device. 

3、The type of a texel, which is restricted to the basic integer and single-precision floating-point types and any of the 1-, 2-, and 4-component vector types defined in Built-in Vector Types that are derived from the basic integer and single-precision floating-point types.

4、The read mode, which is equal to cudaReadModeNormalizedFloat or cudaReadModeElementType. If it is cudaReadModeNormalizedFloat and the type of the texel is a 16-bit or 8-bit integer type, the value returned by the texture fetch is actually returned as floating-point type and the full range of the integer type is mapped to [0.0, 1.0] for unsigned integer type and [-1.0, 1.0] for signed integer type; for example, an unsigned 8-bit texture element with the value 0xff reads as 1. If it is cudaReadModeElementType, no conversion is performed. 

5、Whether texture coordinates are normalized or not. By default, textures are referenced (by the functions of Texture Functions) using floating-point coordinates in the range [0, N-1] where N is the size of the texture in the dimension corresponding to the coordinate. For example, a texture that is 64x32 in size will be referenced with coordinates in the range [0, 63] and [0, 31] for the x and y dimensions, respectively. Normalized texture coordinates cause the coordinates to be specified in the range [0.0, 1.0-1/N] instead of [0, N-1], so the same 64x32 texture would be addressed by normalized coordinates in the range [0, 1-1/N] in both the x and y dimensions. Normalized texture coordinates are a natural fit to some applications' requirements, if it is preferable for the texture coordinates to be independent of the texture size. 

6、The addressing mode. It is valid to call the device functions of Section B.8 with coordinates that are out of range. The addressing mode defines what happens in that case. The default addressing mode is to clamp the coordinates to the valid range: [0, N) for non-normalized coordinates and [0.0, 1.0) for normalized coordinates. If the border mode is specified instead, texture fetches with outof-range texture coordinates return zero. For normalized coordinates, the wrap mode and the mirror mode are also available. When using the wrap mode, each coordinate x is converted to frac(x)=x floor(x) where floor(x) is the largest integer not greater than x. When using the mirror mode, each coordinate x is converted to frac(x) if floor(x) is even and 1-frac(x) if floor(x) is odd. The addressing mode is specified as an array of size three whose first, second, and third elements specify the addressing mode for the first, second, and third texture coordinates, respectively; the addressing mode are cudaAddressModeBorder, cudaAddressModeClamp, cudaAddressModeWrap, and cudaAddressModeMirror; cudaAddressModeWrap and cudaAddressModeMirror are only supported for normalized texture coordinates 

7、The filtering mode which specifies how the value returned when fetching the texture is computed based on the input texture coordinates. Linear texture filtering may be done only for textures that are configured to return floating-point data. It performs low-precision interpolation between neighboring texels. When enabled, the texels surrounding a texture fetch location are read and the return value of the texture fetch is interpolated based on where the texture coordinates fell between the texels. Simple linear interpolation is performed for one-dimensional textures, bilinear interpolation for two-dimensional textures, and trilinear interpolation for threedimensional textures. Texture Fetching gives more details on texture fetching. The filtering mode is equal to cudaFilterModePoint or cudaFilterModeLinear. If it is cudaFilterModePoint, the returned value is the texel whose texture coordinates are the closest to the input texture coordinates. If it is cudaFilterModeLinear, the returned value is the linear interpolation of the two (for a one-dimensional texture), four (for a two dimensional texture), or eight (for a three dimensional texture) texels whose texture coordinates are the closest to the input texture coordinates. cudaFilterModeLinear is only valid for returned values of floating-point type.

Texture Object API introduces the texture object API.
Texture Reference API introduces the texture reference API.
16-Bit Floating-Point Textures explains how to deal with 16-bit floating-point textures.
Textures can also be layered as described in Layered Textures.
Cubemap Textures and Cubemap Layered Textures describe a special type of texture,
the cubemap texture.
Texture Gather describes a special texture fetch, texture gather.

3.2.11.1.1. Texture Object API
A texture object is created using cudaCreateTextureObject() from a resource description of type struct cudaResourceDesc, which specifies the texture, and from a texture description defined as such:

1、 addressMode specifies the addressing mode;
2、 filterMode specifies the filter mode;
3、 readMode specifies the read mode;
4、 normalizedCoords specifies whether texture coordinates are normalized or not;
5、 See reference manual for sRGB, maxAnisotropy, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, and maxMipmapLevelClamp.

```c++
struct cudaTextureDesc
{
	enum cudaTextureAddressMode addressMode[3];
	enum cudaTextureFilterMode filterMode;
	enum cudaTextureReadMode readMode;
	int sRGB;
	int normalizedCoords;
	unsigned int maxAnisotropy;
	enum cudaTextureFilterMode mipmapFilterMode;
	float mipmapLevelBias;
	float minMipmapLevelClamp;
	float maxMipmapLevelClamp;
};
```

1、 addressMode specifies the addressing mode;
2、 filterMode specifies the filter mode;
3、 readMode specifies the read mode;
4、 normalizedCoords specifies whether texture coordinates are normalized or not;
5、 See reference manual for sRGB, maxAnisotropy, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, and maxMipmapLevelClamp.

The following code sample applies some simple transformation kernel to a texture.

```c++
The following code sample applies some simple transformation kernel to a texture.
// Simple transformation kernel
__global__ void transformKernel(float* output,
								cudaTextureObject_t texObj,
								int width, int height,
								float theta)
{
	// Calculate normalized texture coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	float u = x / (float)width;
	float v = y / (float)height;
	// Transform coordinates
	u -= 0.5f;
	v -= 0.5f;
	float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
	float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;
	// Read from texture and write to global memory
	output[y * width + x] = tex2D<float>(texObj, tu, tv);
}

// Host code
int main()
{
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
	cudaCreateChannelDesc(32, 0, 0, 0,
	cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);
	// Copy to device memory some data located at address h_data
	// in host memory
	cudaMemcpyToArray(cuArray, 0, 0, h_data, size, cudaMemcpyHostToDevice);
	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;
	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;
	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	// Allocate result of transformation in device memory
	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	// Invoke kernel
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
	             (height + dimBlock.y - 1) / dimBlock.y);
	transformKernel<<<dimGrid, dimBlock>>>(output, texObj, width, height, angle);
	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	cudaFree(output);
	return 0;
}
```



