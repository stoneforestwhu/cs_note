

## chapter3 Programming Interface        

CUDA C++ provides a simple path for users familiar with the C++ programming language to easily write programs for execution by the device. 

​        It consists of a minimal set of extensions to the C++ language and a runtime library. 

​        The core language extensions have been introduced in Programming Model. They allow programmers to define a kernel as a C++ function and use some new syntax to specify the grid and block dimension each time the function is called. A complete description of all extensions can be found in C++ Language Extensions(Appendix B). Any source file that contains some of these extensions must be compiled with **nvcc** as outlined in Compilation with NVCC (chapter3.1). 

​        The runtime is introduced in CUDA Runtime(chapter3.2). It provides C and C++ functions that execute on the host to allocate and deallocate device memory, transfer data between host memory and device memory, manage systems with multiple devices, etc. A complete description of the runtime can be found in the *CUDA reference manual*.

​        The runtime is built on top of a lower-level C API, the CUDA driver API, which is also accessible by the application. The driver API provides an additional level of control by exposing lower-level concepts such as CUDA contexts - the analogue(相似的事物) of host processes for the device - and CUDA modules - the analogue of dynamically loaded libraries for the device. (CUDA context类比主机中的进程这个概念，CUDA module类比主机中的动态链接库) Most applications do not use the driver API as they do not need this additional level of control and when using the runtime, context and module management are implicit, resulting in more concise code. As the runtime is interoperable(互相操作，互相配合) with the driver API, most applications that need some driver API features can default to use the runtime API and only use the driver API where needed. The driver API is introduced in Driver API(Appendix I) and fully described in the reference manual. 

### 3.1. Compilation with NVCC 

​        Kernels can be written using the CUDA instruction set architecture, called PTX, which is described in the *PTX reference manual*. It is however usually more effective to use a high-level programming language such as C++. In both cases, kernels must be compiled into binary code by nvcc to execute on the device.(kernel函数可以用PTX指令集或者high level的语言来写)

​        nvcc is a compiler driver that simplifies the process of compiling C++ or PTX code: It provides simple and familiar command line options and executes them by invoking the collection of tools that implement the different compilation stages. This section gives an overview of nvcc workflow and command options. A complete description can be found in the *nvcc user manual*. 

#### 3.1.1. Compilation Workflow 

##### 3.1.1.1. Offline Compilation 

​        Source files compiled with nvcc can include a mix of host code (i.e., code that executes on the host) and device code (i.e., code that executes on the device). nvcc's basic workflow consists in separating device code from host code and then: 

1 compiling the device code into an assembly form (PTX code) and/or binary form (cubin object), 

2 and modifying the host code by replacing the <<<...>>> syntax introduced in Kernels (and described in more details in Execution Configuration) by the necessary CUDA runtime function calls to load and launch each compiled kernel from the PTX code and/or cubin object.

(过程就三步，先把host code和device code分离，然后把device code编译成二进制文件，把文件中<<<...>>>替换成CUDA runtime函数调用，用来加载从PTXcode和cubin对象中编译的核函数)

​        The modified host code is output either as C++ code that is left to be compiled using another tool or as object code directly by letting nvcc invoke the host compiler during the last compilation stage. 

(修改完的host code要么作为c++代码由其他工具编译，要么由nvcc启动主机的编译器来编译)

Applications can then: 

1、Either link to the compiled host code (this is the most common case), 

2、Or ignore the modified host code (if any) and use the CUDA driver API (see Driver API) to load and execute the PTX code or cubin object. 

(应用然后去链接编译好的主机代码，或者忽略修改后的主机代码，而去使用CUDA driver API来加载和执行PTX code或者cubin对象)

##### 3.1.1.2. Just-in-Time Compilation 

​        Any PTX code loaded by an application at runtime is compiled further to binary code by the device driver. This is called just-in-time compilation. Just-in-time compilation increases application load time, but allows the application to benefit from any new compiler improvements coming with each new device driver. It is also the only way for applications to run on devices that did not exist at the time the application was compiled, as detailed in Application Compatibility(chapter3.1.4). 

​        When the device driver just-in-time compiles some PTX code for some application, it automatically caches a copy of the generated binary code in order to avoid repeating the compilation in subsequent invocations of the application. The cache - referred to as compute cache - is automatically invalidated when the device driver is upgraded, so that applications can benefit from the improvements in the new just-in-time compiler built into the device driver.

(应用在运行时直接加载PTX代码，然后编译成二进制代码被称为实时编译。实时编译增加了应用的加载时间，但是好处是能够享受新编译器提升带来的好处，当这些代码被编译之后就被缓存起来了避免重复编译。当编译器升级后，这些缓存会自动失效)

​        Environment variables are available to control just-in-time compilation as described in CUDA Environment Variables 

（有一些环境变量能够控制实时编译）

​        As an alternative to using nvcc to compile CUDA C++ device code, NVRTC can be used to compile CUDA C++ device code to PTX at runtime. NVRTC is a runtime compilation library for CUDA C++; more information can be found in the NVRTC User guide. (NVRTC能够把high level的代码编译成PTX代码)

#### 3.1.2. Binary Compatibility 

​        Binary code is architecture-specific. A cubin object is generated using the compiler option *-code* that specifies the targeted architecture: For example, compiling with -code=sm_35 produces binary code for devices of compute capability 3.5. Binary compatibility is guaranteed from one minor revision to the next one, but not from one minor revision to the previous one or across major revisions. In other words, a cubin object generated for compute capability X.y will only execute on devices of compute capability X.z where z≥y. （这一段主要讨论的是二进制码的兼容性）

```
Binary compatibility is supported only for the desktop. It is not supported for Tegra.
Also, the binary compatibility between desktop and Tegra is not supported.(桌面产品才会有二进制的兼容性，Tegra没有)
```

#### 3.1.3. PTX Compatibility

​         Some PTX instructions are only supported on devices of higher compute capabilities. For example, Warp Shuffle Functions are only supported on devices of compute capability 3.0 and above. The *-arch* compiler option specifies the compute capability that is assumed when compiling C++ to PTX code. So, code that contains warp shuffle, for example, must be compiled with -arch=compute_30 (or higher).

编译选项-arch指定计算兼容性。举例来说，warp shuffle函数只支持计算兼容性大于等于3.0的，所以如果high level代码中由warp shuffle，编译时必须添加编译选项 -arch=compute_30

​        PTX code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability. (如果PTX代码有特定的compute capability编译，那么一定可以编译出等于或者更高版本compute capability的二进制代码) Note that a binary compiled from an earlier PTX version may not make use of some hardware features. (但是可能使用不了新的feature) For example, a binary targeting devices of compute capability 7.0 (Volta) compiled from PTX generated for compute capability 6.0 (Pascal) will not make use of Tensor Core instructions, since these were not available on Pascal. As a result, the final binary may perform worse than would be possible if the binary were generated using the latest version of PTX.

####  3.1.4. Application Compatibility 

​        To execute code on devices of specific compute capability, an application must load binary or PTX code that is compatible with this compute capability as described in Binary Compatibility and PTX Compatibility. In particular, to be able to execute code on future architectures with higher compute capability (for which no binary code can be generated yet), an application must load PTX code that will be just-in-time compiled for these devices (see Just-in-Time Compilation). 

​        Which PTX and binary code gets embedded in a CUDA C++ application is controlled by the *-arch* and *-code* compiler options or the *-gencode* compiler option as detailed in the nvcc user manual. For example, 

```shell
nvcc x.cu 
       -gencode arch=compute_35,code=sm_35 
       -gencode arch=compute_50,code=sm_50 
       -gencode arch=compute_60,code=\'compute_60,sm_60\' 
```

​        embeds binary code compatible with compute capability 3.5 and 5.0 (first and second -gencode options) and PTX and binary code compatible with compute capability 6.0 (third -gencode option). 

​        Host code is generated to automatically select at runtime the most appropriate code to load and execute, which, in the above example, will be: (尽管随意编译，选择依旧由主机代码自己定)

1、3.5 binary code for devices with compute capability 3.5 and 3.7, 

2、5.0 binary code for devices with compute capability 5.0 and 5.2, 

3、6.0 binary code for devices with compute capability 6.0 and 6.1, 

4、PTX code which is compiled to binary code at runtime for devices with compute capability 7.0 and higher. 

​        x.cu can have an optimized code path that uses warp shuffle operations, for example, which are only supported in devices of compute capability 3.0 and higher. The \_\_CUDA_ARCH\_\_ macro can be used to differentiate various code paths based on compute capability. It is only defined for device code. When compiling with *-arch=compute_35* for example, \_\_CUDA_ARCH\_\_ is equal to 350. 

​        Applications using the driver API must compile code to separate files and explicitly load and execute the most appropriate file at runtime. 

​        The Volta architecture introduces Independent Thread Scheduling which changes the way threads are scheduled on the GPU. For code relying on specific behavior of SIMT scheduling in previous architecures, Independent Thread Scheduling may alter the set of participating threads, leading to incorrect results. To aid migration while implementing the corrective actions detailed in Independent Thread Scheduling, Volta developers can opt-in to Pascal's thread scheduling with the compiler option combination *- arch=compute_60   -code=sm_70*.

​        The nvcc user manual lists various shorthands for the -arch, -code, and -gencode compiler options. For example, -arch=sm_35 is a shorthand for -arch=compute_35 - code=compute_35,sm_35 (which is the same as -gencode arch=compute_35,code= \'compute_35,sm_35\'). 

#### 3.1.5. C++ Compatibility 

​        The front end of the compiler processes CUDA source files according to C++ syntax rules. Full C++ is supported for the host code. However, only a subset of C++ is fully supported for the device code as described in C++ Language Support. (device code不支持全部的c++，只是一个子集)

#### 3.1.6. 64-Bit Compatibility 

​        The 64-bit version of nvcc compiles device code in 64-bit mode (i.e., pointers are 64-bit). Device code compiled in 64-bit mode is only supported with host code compiled in 64- bit mode. 

​        Similarly, the 32-bit version of nvcc compiles device code in 32-bit mode and device code compiled in 32-bit mode is only supported with host code compiled in 32-bit mode. 

​        The 32-bit version of nvcc can compile device code in 64-bit mode also using the -m64 compiler option. The 64-bit version of nvcc can compile device code in 32-bit mode also using the -m32 compiler option. (32位的nvcc可以编译出64位的device code，同理，64位的nvcc可以编译出32位的device code)

### 3.2 CUDA Runtime

​        The runtime is implemented in the cudart library, which is linked to the application, either statically via cudart.lib or libcudart.a, or dynamically via cudart.dll or libcudart.so. Applications that require cudart.dll and/or cudart.so for dynamic linking typically include them as part of the application installation package. It is only safe to pass the address of CUDA runtime symbols between components that link to the same instance of the CUDA runtime.
​        All its entry points(函数入口) are prefixed with cuda.
​        As mentioned in **Heterogeneous Programming**, the CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. **Device Memory**(chapter3.2.2) gives an overview of the runtime functions used to manage device memory.
​        **Shared Memory**(chapter3.2.2)illustrates the use of shared memory, introduced in **Thread Hierarchy,**
to maximize performance.
​        **Page-Locked Host Memory**(页锁定的主机内存, chapter3.2.4) introduces page-locked host memory that is required to overlap(这里的overlap到底是什么意思) kernel execution with data transfers between host and device memory.
​        **Asynchronous Concurrent Execution**(异步并发执行, chapter3.2.5) describes the concepts and API used to enable asynchronous concurrent execution at various levels in the system.
​        **Multi-Device System**(单机多卡系统, chapter3.2.6) shows how the programming model extends to a system with multiple devices attached to the same host.
​        **Error Checking**(chapter3.2.9) describes how to properly check the errors generated by the runtime.
​        **Call Stack**(chapter3.2.10) mentions the runtime functions used to manage the CUDA C++ call stack.
​        **Texture and Surface Memory**(chapter3.2.11) presents the texture and surface memory spaces that provide another way to access device memory; they also expose a subset of the GPU texturing hardware.
​        **Graphics Interoperability**(chapter3.2.12) introduces the various functions the runtime provides to
interoperate with the two main graphics APIs, OpenGL and Direct3D.  

#### 3.2.1 Initialization

​        There is no explicit initialization function for the runtime; it initializes the first time a runtime function is called (more specifically any function other than functions from the error handling and version management sections of the reference manual). One needs to keep this in mind when timing runtime function calls and when interpreting the error code from the first call into the runtime.

​       运行时没有显式的初始化函数,当第一个运行时函数被调用时直接就初始化了(除了错误处理和版本管理的函数).当计算运行时函数的运行时间和解释第一次调用运行时函数的错误码时必须时刻记住这一点.

​        During initialization, the runtime creates a CUDA context for each device in the system(see Context for more details on CUDA contexts). This context is the primary context for this device and it is shared among all the host threads of the application. As part of this  context creation, the device code is just-in-time compiled if necessary (see Just-in-Time Compilation) and loaded into device memory. This all happens transparently. If needed, e.g. for driver API interoperability, the primary context of a device can be accessed from the driver API as described in Interoperability between Runtime and Driver APIs.  

​       在初始化过程中,运行时在系统中为每个设备(gpu)创建了一个CUDA上下文.这个上下文只是这个设备的一个初始上下文,并且在应用的所有主机线程中共享.作为上下文创建的一部分,如果需要的话,设备的代码是实时编译的,并且加载到设备的内存中. 所有这一切全是透明的.如果需要,对于驱动API的互操作性(interoperability),设备的初始上下文能够从驱动的API中获取

​        When a host thread calls cudaDeviceReset(), this destroys the primary context of the device the host thread currently operates on (i.e., the current device as defined in Device Selection). The next runtime function call made by any host thread that has this device as current will create a new primary context for this device.  

​         当一个主机线程调用cudaDeviceReset(), 这个操作销毁了主机线程当前操作的设备的初始上下文.下一次任何将这个设备作为current的主机线程调用运行时函数都会为这个设备创建一个新的初始上下文.

```
    The CUDA interfaces use global state that is initialized during host program initiation and destroyed during host program termination. The CUDA runtime and driver cannot detect if this state is invalid, so using any of these interfaces (implicitly or explicity) during program initiation or termination after main) will result in undefined behavior.
    cuda接口使用的全局状态(global state)是在主机程序初始化时初始化的,在主程序终结时被销毁的.CUDA运行时和驱动不能检测到全局状态(global state)是否有效,所以在主程序初始化或者main函数结束后使用这些接口都会导致未定义行为
    什么是global state???
```

#### 3.2.2 Device Memory

​        As mentioned in **Heterogeneous Programming**, the CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. Kernels operate out of device memory, so the runtime provides functions to allocate, deallocate, and copy device memory, as well as transfer data between host memory and device memory.

​       正如之前在异构编程这一节中提到的,CUDA编程模型假定系统由一个主机(host)和一个设备(device)组成, 而且每个都有自己独立的存储器.核函数运行在设备存储器中,所以运行时需要提供函数来分配,回收和复制设备的内存,同时在主机存储器和设备存储器之间传输数据.

​        Device memory can be allocated either as linear memory or as CUDA arrays. 

​        设备存储器能够被分配成线性存储器或者CUDA数组.

​        CUDA arrays are opaque memory layouts optimized for texture fetching. They are described in Texture and Surface Memory.

​         CUDA数组的内存布局是不透明的,专门为了获取纹理做了优化. 这一点在**Texture and Surface Memory**(chapter3.2.11)中描述.

​        Linear memory is allocated in a **single unified address space**, which means that separately allocated entities can reference one another via pointers, for example, in a binary tree or linked list. The size of the address space depends on the host system (CPU) and the compute capability  

​        线性存储器是在单个统一的地址空间中分配的,这就意味着独立分配的实体能够通过指针来引用,例如二叉树和链表.地址空间的大小取决于主机(host)系统和GPU的compute capability

![table_1](D:\ComputerScience\note\cs_note\cuda_c_programming_guide\graph\table_1.PNG)

```
    On devices of compute capability 5.3 (Maxwell) and earlier, the CUDA driver creates
an uncommitted 40bit virtual address reservation to ensure that memory allocations
(pointers) fall into the supported range. This reservation appears as reserved virtual
memory, but does not occupy any physical memory until the program actually
allocates memory.
    在compute capability 为5.3或者更早的显卡中,CUDA驱动创建一个未提交的40bit虚拟保留地址来保证分配的内存在支持的范围中.这个保留呈现为一个保留的虚拟地址,但是并不占用任何物理地址直到程序真正分配了内存
```

​        Linear memory is typically allocated using cudaMalloc() and freed using cudaFree() and data transfer between host memory and device memory are typically done using  cudaMemcpy(). In the vector addition code sample of Kernels, the vectors need to be copied from host memory to device memory:  

​        线性存储器一般使用cudaMalloc()来分配内存, 使用cudaFree()来释放内存, 使用cudaMemcpy()来在主机内存和设备内存之间传输数据.在向量加的代码例子中,向量需要从主机内存拷贝到设备内存.

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

​        Linear memory can also be allocated through cudaMallocPitch() and cudaMalloc3D(). These functions are recommended for allocations of 2D or 3D arrays as it makes sure that the allocation is appropriately padded to meet the alignment requirements described in **Device Memory Accesses**(chapter5.3.2), therefore ensuring best performance when accessing the row addresses or performing copies between 2D arrays and other regions of device memory (using the cudaMemcpy2D() and cudaMemcpy3D() functions). The returned pitch (or stride) must be used to access array elements. The  following code sample allocates a width x height 2D array of floating-point values and shows how to loop over the array elements in device code:  

​        线性存储器也可以通过cudaMallocPitch()和cudaMalloc3D()来分配.推荐使用这两个函数用来分配2D和3D的数组,因为这能确保分配的空间会通过padding来满足章节 5.3.2 Device memory Accesses中描述的字节对齐的要求, 也因此能保证在获取行地址或者使用别的API(cudaMemcpy2D和cudaMemcpy3D)来在2D数组和设备内存的其他区域之间复制数据时获得最好的性能.返回的pitch或者stride必须用来获取数组元素.以下代码是在设备中分配一个二维数组并且循环遍历数组元素的设备代码

```c++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);
// Device code
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
	for (int r = 0; r < height; ++r) {
        #  pitch是一行占据的字节数，padding之后的大小不是width*sizeof(float)，而由pitch指定
		float* row = (float*)((char*)devPtr + r * pitch);
		for (int c = 0; c < width; ++c) {
			float element = row[c];
		}
	}
}
```

​        The following code sample allocates a width x height x depth 3D array of floating-point values and shows how to loop over the array elements in device code:  

​       以下代码是在设备中分配一个三维数组并且循环遍历数组元素的设备代码

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

​        The reference manual lists all the various functions used to copy memory between
linear memory allocated with cudaMalloc(), linear memory allocated with cudaMallocPitch() or cudaMalloc3D(), CUDA arrays, and memory allocated for variables declared in global or constant memory space.  

​        参考手册列举了...

​        The following code sample illustrates various ways of accessing global variables via the
runtime API:  

​        下列代码列举通过运行时API访问全局变量的各种方式

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

​        cudaGetSymbolAddress() is used to retrieve the address pointing to the memory allocated for a variable declared in global memory space. The size of the allocated memory is obtained through cudaGetSymbolSize().  

#### 3.2.3 Shared Memory

​        As detailed in Variable Memory Space Specifiers(appendix B.2) shared memory is allocated using the
\_\_shared\_\_ memory space specifier.        

​        共享存储器在分配时使用\_\_share\__存储空间限定符.

​        Shared memory is expected to be much faster than global memory as mentioned in Thread Hierarchy(chapter2.2) and detailed in Shared Memory. It can be used as scratchpad memory (or software managed cache) to minimize global memory accesses from a CUDA block as illustrated by the following matrix multiplication example.  

​        共享存储器比全局存储器的速度快得多(这一点在Thread HIerarchy中有提到).它可以用来作为scratchpad memory(或者software managed cache)来使得CUDA block对全局存储器的访问达到最低,后面有矩阵乘法的示例代码来阐述这个问题.

​        The following code sample is a straightforward implementation of matrix multiplication
that does not take advantage of shared memory. Each thread reads one row of A and one column of B and computes the corresponding element of C as illustrated in Figure 9. A is therefore read B.width times from global memory and B is read A.height times.  

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

​        The following code sample is an implementation of matrix multiplication that does take
advantage of shared memory. In this implementation, each thread block is responsible for computing one square sub-matrix Csub of C and each thread within the block is responsible for computing one element of Csub. As illustrated in Figure 10, Csub is equal to the product of two rectangular matrices: the sub-matrix of A of dimension (A.width, block_size) that has the same row indices as Csub, and the sub-matrix of B of dimension (block_size, A.width )that has the same column indices as Csub. In order to fit into the
device's resources, these two rectangular matrices are divided into as many square matrices of dimension block_size as necessary and Csub is computed as the sum of the products of these square matrices. Each of these products is performed by first loading the two corresponding square matrices from global memory to shared memory with one thread loading one element of each matrix, and then by having each thread compute one element of the product. Each thread accumulates the result of each of these products into a register and once done writes the result to global memory         

​        下面这部分代码是使用了shared_memory的矩阵乘法实现.在这个实现中,每个线程块,负责计算一个C的一个子矩阵Csub,线程块的每个线程负责计算Csub中的一个元素.如图所示,Csub等于两个矩形矩阵的乘积;A的子矩阵的维度(A.width,block_size)和Csub的行索引数量相同,B的子矩阵的维度(block_size, B.height)和Csub的列索引数量相同.为了适配设备的资源,这两个矩形的矩阵分割成尽可能多的维度为(block_size, block_size)的方阵,并且Csub最后的结果是这些方阵的和.在计算这些方矩阵的乘积时, 第一步是从全局存储器中加载两个相关的方阵的数据,每个线程加载这些方阵中的一个元素, 然后每个线程计算一个元素的乘积.每个线程累加这些和的结果到寄存器, 而且最后写回到全局存储器中.

​        By blocking the computation this way, we take advantage of fast shared memory and
save a lot of global memory bandwidth since A is only read (B.width / block_size) times
from global memory and B is read (A.height / block_size) times.  

​        通过分块计算的方式, 我们使用了高速的共享存储器并且节省了大量的全局存储器带宽, 因为A从全局存储器中只被读取B.width/block_size次, B从全局存储器中只被读取了(A.height/block_size)次.

​        The Matrix type from the previous code sample is augmented with a stride field, so that sub-matrices can be efficiently represented with the same type. __device__ functions are used to get and set elements and build any sub-matrix from a matrix.  

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

​        The runtime provides functions to allow the use of page-locked (also known as pinned) host memory (as opposed to regular pageable host memory allocated by malloc()):

​         运行时提供了函数来允许使用页锁定主机内存(也叫pinned memory), 这是相对于malloc()分配的regular pageable host memory(可分页内存)

1、cudaHostAlloc() and cudaFreeHost() allocate and free page-locked host memory; 

2、cudaHostRegister() page-locks a range of memory allocated by malloc() (see reference manual for limitations). 

1、cudaHostAlloc()和cudaFreeHost()分配和释放页锁定主机内存

2、cudaHostRegister() 页锁定由malloc()分配的内存

​        Using page-locked host memory has several benefits: 

​        使用页锁定内存有以下几个好处:

1、Copies between page-locked host memory and device memory can be performed concurrently with kernel execution for some devices as mentioned in Asynchronous Concurrent Execution. 

2、On some devices, page-locked host memory can be mapped into the address space of the device, eliminating the need to copy it to or from device memory as detailed in Mapped Memory. 

3、On systems with a front-side bus, bandwidth between host memory and device memory is higher if host memory is allocated as page-locked and even higher if in addition it is allocated as write-combining as described in Write-Combining Memory.

   1、在页锁定内存和设备内存之间复制数据(Copies between page-locked host memory and device memory can be performed concurrently with kernel execution for some devices )

   2、在某些设备上,页锁定内存可以映射到设备的地址空间上,这样可以省略往设备存储器读或者写的时间

   3、在一个有前端总线(front-side bus)的系统上, 主机存储器和设备存储器之间的带宽会更高,如果主机存储器是页锁定的,甚至还要高些如果这些页锁定内存被分配成write-combining

​        Page-locked host memory is a scarce resource however, so allocations in page-locked memory will start failing long before allocations in pageable memory. In addition, by reducing the amount of physical memory available to the operating system for paging, consuming too much page-locked memory reduces overall system performance.  The simple zero-copy CUDA sample comes with a detailed document on the pagelocked memory APIs. 

​         页锁定存储器是一项稀缺的资源, 如果持续分配页锁定存储器, 那么后续继续分配存储器可能会失败.另外,页锁定存储器减少了操作系统可用于分页的物理存储器,如果可分页存储器消耗太多会导致整体系统的性能下降. 

```
Page-locked host memory is not cached on non I/O coherent Tegra devices. Also, cudaHostRegister() is not supported on non I/O coherent Tegra devices.
```

​       The simple zero-copy CUDA sample comes with a detailed document on the page-locked memory APIs.  

##### 3.2.4.1 Portable Memory

​        A block of page-locked memory can be used in conjunction with any device in the system (see Multi-Device System for more details on multi-device systems), but by default, the benefits of using page-locked memory described above are only available in conjunction with the device that was current when the block was allocated (and with all devices sharing the same unified address space, if any, as described in Unified Virtual Address Space). To make these advantages available to all devices, the block needs to be allocated by passing the flag cudaHostAllocPortable to cudaHostAlloc() or pagelocked by passing the flag cudaHostRegisterPortable to cudaHostRegister().          

​        页锁定存储器的块能用来连接系统中的任意设备,但是默认情况下,页锁定存储器带来的性能优势只适用于当页锁定存储器块分配时,当前正在使用的设备(the device that was current).为了使得所有的设备获取这种性能优势,存储器块在分配的时候需要传入flag cudaHostAllocPortable 给cudaHostAlloc()或者通过传入cudaHostRegister() flag cudaHostRegisterPortable

##### 3.2.4.2 Write-Combining Memory

​        By default page-locked host memory is allocated as cacheable. It can optionally be allocated as write-combining instead by passing flag cudaHostAllocWriteCombined to cudaHostAlloc(). Write-combining memory frees up the host's L1 and L2 cache resources, making more cache available to the rest of the application. In addition, writecombining memory is not **snooped** during transfers across the PCI Express bus, which can improve transfer performance by up to 40%.  

​        默认情况下页锁定内存分配的时候是可缓存的(cacheable).它可以选择性的分配成write-combining通过给cu daHostAlloc()传入flag cudahostAllocWriteCombined.Write-combining存储器不占用主机的L1缓存和L2缓存,可以留给给多缓存给其他应用.In addition, write-combining memory is not **snooped**(窥探) during transfers across the PCI Express bus, which can improve transfer performance by up to 40%.

​        Reading from write-combining memory from the host is prohibitively slow, so write-combining memory should in general be used for memory that the host only writes to.  

​        从host 的 write-combining存储器读取数据不可避免的慢,所以write-combining存储器通常应该只让主机写

##### 3.2.4.3 Mapped Memory

​        A block of page-locked host memory can also be mapped into the address space of the device by passing flag cudaHostAllocMapped to cudaHostAlloc() or by passing flag cudaHostRegisterMapped to cudaHostRegister(). Such a block has therefore in general two addresses: one in host memory that is returned by cudaHostAlloc() or malloc(), and one in device memory that can be retrieved using cudaHostGetDevicePointer() and then used to access the block from within a kernel. The only exception is for pointers allocated with cudaHostAlloc() and when a unified address space is used for the host and the device as mentioned in Unified Virtual Address Space.  

​        通过给cudaHostAlloc()传入flag cudaHostAllocMapped(或者把cudaHostRegisterMapped传入cudaHostRegister()), 页锁定存储器块也可以映射到设备存储器的地址空间. 因此这样的存储器块通常有两个地址:一个是主机存储器地址,通常是malloc()或者cudaHostAlloc()返回的, 另一个是设备存储器的地址, 可以通过cudaHostGetDevicePointer()获取. 然后这个地址可以在kernel函数中用来访问存储器块.  The only exception is for pointers allocated with cudaHostAlloc() and when a unified address space is used for the host and the device as mentioned in **Unified Virtual Address Space**.

​         Accessing host memory directly from within a kernel does not provide the same bandwidth as device memory, but does have some advantages:  

​         设备通过kernel函数直接访问主机存储器的带宽和直接访问设备存储器不同,但是有一些优势:

​        1、There is no need to allocate a block in device memory and copy data between this block and the block in host memory; data transfers are implicitly performed as needed by the kernel;

​        2、There is no need to use streams (see Concurrent Data Transfers) to overlap data
transfers with kernel execution; the kernel-originated data transfers automatically
overlap with kernel execution.(没懂)

​         1、没有必要在设备存储器中分配一个块,并且在这个存储器块和主机存储器块之间复制数据,数据传输只在和函数需要的时候隐式执行

​         2、There is no need to use streams (see Concurrent Data Transfers) to overlap data transfers with kernel execution; the kernel-originated data transfers automatically overlap with kernel execution

​        Since mapped page-locked memory is shared between host and device however,
the application must synchronize memory accesses using streams or events (see
Asynchronous Concurrent Execution) to avoid any potential read-after-write, write-after-read, or write-after-write hazards.         

​         既然映射的页锁定存储器在主机和设备之间共享,但是,应用必须使用streams和events来同步存储器访问来避免任何潜在的read-after-write, write-after-read或者write-after-write的风险.

​        To be able to retrieve the device pointer to any mapped page-locked memory, page-locked memory mapping must be enabled by calling **cudaSetDeviceFlags**() with the *cudaDeviceMapHost* flag before any other CUDA call is performed. Otherwise, **cudaHostGetDevicePointer**() will return an error.
​        **cudaHostGetDevicePointer**() also returns an error if the device does not support mapped page-locked host memory. Applications may query this capability by checking the canMapHostMemory device property (see Device Enumeration), which is equal to 1 for devices that support mapped page-locked host memory.
​        Note that atomic functions (see Atomic Functions) operating on mapped page-locked memory are not atomic from the point of view of the host or other devices. 
​        Also note that CUDA runtime requires that 1-byte, 2-byte, 4-byte, and 8-byte naturally aligned loads and stores to host memory initiated from the device are preserved as single accesses from the point of view of the host and other devices. On some platforms, atomics to memory may be broken by the hardware into separate load and store operations. These component load and store operations have the same requirements on preservation of naturally aligned accesses. As an example, the CUDA runtime does not support a PCI Express bus topology where a PCI Express bridge splits 8-byte naturally aligned writes into two 4-byte writes between the device and the host.

#### 3.2.5 Asynchronous Concurrent Execution

​        CUDA exposes the following operations as independent tasks that can operate
concurrently with one another:(可同时进行的操作)
1、Computation on the host;
2、Computation on the device;
3、Memory transfers from the host to the device;(数据从host到device)
4、Memory transfers from the device to the host;(数据从device到host)
5、Memory transfers within the memory of a given device;(设备内部存储器数据传输)
6、Memory transfers among devices.(设备之间存储器数据传输)
​        The level of concurrency achieved between these operations will depend on the feature
set and compute capability of the device as described below.

##### 3.2.5.1. Concurrent Execution between Host and Device

​        Concurrent host execution is facilitated through asynchronous(异步) library functions that return control to the host thread before the device completes the requested task. Using asynchronous calls, many device operations can be queued up together to be executed by the CUDA driver when appropriate device resources are available. This relieves the host thread of much of the responsibility to manage the device, leaving it free for other tasks. The following device operations are asynchronous with respect to the host:
1、Kernel launches;
2、Memory copies within a single device's memory;
3、Memory copies from host to device of a memory block of 64 KB or less;
4、Memory copies performed by functions that are suffixed with Async;
5、Memory set function calls.

​        异步执行是把程序的执行权返回给主机线程，主机发出的指令通过cuda driver管理的命令队列来实现管理

​        Programmers can globally <u>disable</u> asynchronicity of kernel launches for all CUDA
applications running on a system by setting the **CUDA_LAUNCH_BLOCKING** environment
variable to 1. This feature is provided for debugging purposes only and <u>should not be</u>
<u>used as a way to make production software</u> run reliably.

​       kernel的异步执行可以disable,通过把环境变量**CUDA_LAUNCH_BLOCKING**设为1

​        Kernel launches are synchronous if hardware counters are collected via a profiler (Nsight, Visual Profiler) unless concurrent kernel profiling is enabled. Async memory copies will also be synchronous if they involve host memory that is not page-locked.

​       如果硬件计数器是通过分析器得到的，kernel是同步launch的，除非concurrent kernel profiling使能了(concurrent kernel profiling是什么？) 异步存储器的复制会变成同步操作如果它们的操作对象的host memory不是页锁定的。

##### 3.2.5.2. Concurrent Kernel Execution

​        Some devices of compute capability 2.x and higher can execute multiple kernels concurrently. Applications may query this capability by checking the **concurrentKernels** device property (see Device Enumeration), which is equal to 1 for devices that support it.

​         concurrentKernels能决定设备是不是支持并发执行多个kernel

​        The maximum number of kernel launches that a device can execute concurrently
depends on its compute capability and is listed in Table 15.

​       A kernel from one CUDA context cannot execute concurrently with a kernel from
another CUDA context.

​     不同上下文的kernel不能并发执行

​      Kernels that use many textures or a large amount of local memory are less likely to execute concurrently with other kernels.

##### 3.2.5.3. Overlap of Data Transfer and Kernel Execution(数据传输和kernel函数执行交叠操作)

​        Some devices can perform an asynchronous memory copy to or from the GPU concurrently with kernel execution. Applications may query this capability by checking the **asyncEngineCount** device property (see Device Enumeration), which is greater than zero for devices that support it. If host memory is involved in the copy, it must be page-locked. (通过查询asyncEngineCount来看GPU是不是支持同时从kernel中异步并发拷贝内存)

​        It is also possible to perform an intra-device copy simultaneously with kernel execution (on devices that support the concurrentKernels device property) and/or with copies to or from the device (for devices that support the asyncEngineCount property). Intra-device copies are initiated using the standard memory copy functions with destination and source addresses residing on the same device.

##### 3.2.5.4. Concurrent Data Transfers

​        Some devices of compute capability 2.x and higher can overlap copies to and from the device. Applications may query this capability by checking the **asyncEngineCount** device property (see Device Enumeration), which is equal to 2 for devices that support it. In order to be overlapped, any host memory involved in the transfers must be page-locked.(从设备往外拷贝数据操作和往设备写数据操作可以同步进行，只要硬件支持这个feature)

##### 3.2.5.5. Streams

​        Applications manage the concurrent operations described above through streams. (应用通过streams来管理并发的操作的). A stream is a sequence of commands (possibly issued by different host threads) that execute in order. (一个stream是一个命令队列)Different streams, on the other hand, may execute their commands out of order with respect to one another or concurrently; this behavior is not guaranteed and should therefore not be relied upon for correctness (e.g., inter-kernel communication is undefined). The commands issued on a stream may execute when all the dependencies of the command are met. The dependencies could be previously launched commands on same stream or dependencies from other streams. **The successful completion of synchronize call guarantees** that all the commands launched are completed.

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

​        Each stream copies its portion of input array hostPtr to array inputDevPtr in device memory, processes inputDevPtr on the device by calling MyKernel(), and copies the result outputDevPtr back to the same portion of hostPtr. Overlapping Behavior describes how the streams overlap in this example depending on the capability of the device. Note that hostPtr must point to **page-locked host memory** for any overlap to occur.

​        Streams are released by calling cudaStreamDestroy().

```c++
for (int i = 0; i < 2; ++i)
  cudaStreamDestroy(stream[i]);
```

​        In case the device is still doing work in the stream when cudaStreamDestroy() is called, the function will return immediately and the resources associated with the stream will be released automatically once the device has completed all work in the stream.

###### 3.2.5.5.2. Default Stream

​        Kernel launches and host <-> device memory copies that do not specify any stream parameter, or equivalently that set the stream parameter to zero, are issued to the default stream. They are therefore executed in order. 
​        For code that is compiled using the --default-stream per-thread compilation flag(or that defines the CUDA_API_PER_THREAD_DEFAULT_STREAM macro before including CUDA headers (cuda.h and cuda_runtime.h)), the default stream is a regular stream and each host thread has its own default stream.(这里是每个线程都有一个default stream)

```c++
    #define CUDA_API_PER_THREAD_DEFAULT_STREAM 1 cannot be used to enable this behavior when the code is compiled by nvcc as nvcc implicitly includes cuda_runtime.h at the top of the translation unit. 
In this case the --default-stream per-thread compilation flag needs to be used or the CUDA_API_PER_THREAD_DEFAULT_STREAM macro needs to be defined with the -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1 compiler flag.
    如果是nvcc来编译代码则使用的CUDA_API_PER_THREAD_DEFAULT_STREAM无效，除非编译的时候加上-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1 才行
```

​        For code that is compiled using the **--default-stream legacy** compilation flag, the default stream is a special stream called the NULL stream and each device has a single NULL stream used for all host threads. The NULL stream is special as it causes implicit synchronization as described in Implicit Synchronization.(这里是所有的主机线程有一个NULL stream)

​        For code that is compiled without specifying a **--default-stream** compilation flag, -**-default-stream legacy** is assumed as the default.

###### 3.2.5.5.3. Explicit Synchronization

​        There are various ways to explicitly synchronize streams with each other.

​        显式同步stream的API列举：

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

​       数据传输overlap需要硬件的支持，kernel执行overlap需要的硬件的支持，kernel执行和数据传输overlap也需要硬件的支持，这些feature并不是所有的设备都支持，并且相对独立的feature。如果设备不支持数据传输overlap，那么需要做的如下例(数据传输和kernel执行是overlap的，但是两个stream的数据传输是按顺序执行的)

```c++
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
```

​        then the memory copy from host to device issued to stream[1] overlaps with the kernel launch issued to stream[0].
​        On devices that do support concurrent data transfers, the two streams of the code sample of Creation and Destruction do overlap: The memory copy from host to device issued to stream[1] overlaps with the memory copy from device to host issued to stream[0] and even with the kernel launch issued to stream[0] (assuming the device supports overlap of data transfer and kernel execution). However, for devices of compute capability 3.0 or lower, the kernel executions cannot possibly overlap because the second kernel launch is issued to stream[1] after the memory copy from device to host is issued to stream[0], so it is blocked until the first kernel launch issued to stream[0] is complete as per **Implicit Synchronization**(chapter3.2.5.5.4). If the code is rewritten as above, the kernel executions overlap (assuming the device supports concurrent kernel execution) since the second kernel launch is issued to stream[1] before the memory copy from device to host is issued to stream[0]. In that case however, the memory copy from device to host issued to stream[0] only overlaps with the last thread blocks of the kernel launch issued to stream[1] as per Implicit Synchronization, which can represent only a small portion of the total execution time of the kernel.

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

​        上面代码的例子中，如果直接printf("...")，语法上是没有问题，但是由于memory copy都是异步执行，并不能保证Mycallback函数能在拷贝操作结束之后执行  

​          A host function enqueued into a stream must not make CUDA API calls (directly or indirectly), as it might end up waiting on itself if it makes such a call leading to a deadlock.  (不要再call)

###### 3.2.5.5.7. Stream Priorities 

​         The relative priorities of streams can be specified at creation using cudaStreamCreateWithPriority(). The range of allowable priorities, ordered as [ highest priority, lowest priority ] can be obtained using the cudaDeviceGetStreamPriorityRange() function. At runtime, pending work in higher-priority streams takes preference over pending work in low-priority streams. 

Graph类似批量操作，预先定义，然后执行，便于优化

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

​        Work submission using graphs is separated into three distinct stages: definition, instantiation(实例化), and execution.
1、During the definition phase, a program creates a description of the operations in the graph along with the dependencies between them.
2、Instantiation takes a snapshot of the graph template, validates it, and performs much of the setup and initialization of work with the aim of minimizing what needs to be done at launch. The resulting instance is known as an executable graph.
3、 An executable graph may be launched into a stream, similar to any other CUDA work. It may be launched any number of times without repeating the instantiation.

###### 3.2.5.6.1. Graph Structure

​        An operation forms a node in a graph. The dependencies between the operations are the edges. These dependencies constrain the execution sequence of the operations.
​        An operation may be scheduled at any time once the nodes on which it depends are complete. Scheduling is left up to the CUDA system.

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

​        Stream capture can be used on any CUDA stream except cudaStreamLegacy (the "NULL stream"). Note that it can be used on cudaStreamPerThread. If a program is using the legacy stream, it may be possible to redefine stream 0 to be the per-thread stream with no functional change. See **Default Stream**(chapter3.2.5.5.2). 

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

​        When any stream in the same context is being captured, and it was not created with cudaStreamNonBlocking(这是一个flag，并不是函数), any attempted use of the legacy stream is invalid. This is because the legacy stream handle at all times encompasses(包含，涉及，包围) these other streams; enqueueing to the legacy stream would create a dependency on the streams being captured, and querying it or synchronizing it would query or synchronize the streams being captured. 

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

#### 3.2.6. Multi-Device System 

##### 3.2.6.1. Device Enumeration 

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

##### 3.2.6.2. Device Selection

​         A host thread can set the device it operates on at any time by calling cudaSetDevice(). Device memory allocations and kernel launches are made on the currently set device; streams and events are created in association with the currently set device. If no call to cudaSetDevice() is made, the current device is device 0.

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

##### 3.2.6.3. Stream and Event Behavior

​        A kernel launch will fail if it is issued to a stream that is not associated to the current device as illustrated in the following code sample.

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

##### 3.2.6.4. Peer-to-Peer Memory Access

​        Depending on the system properties, specifically the PCIe and/or NVLINK topology, devices are able to address each other's memory (i.e., a kernel executing on one device can dereference a pointer to the memory of the other device). This peer-to-peer memory access feature is supported between two devices if cudaDeviceCanAccessPeer() returns true for these two devices. 

​        Peer-to-peer memory access is only supported in 64-bit applications and must be enabled between two devices by calling cudaDeviceEnablePeerAccess() as illustrated in the following code sample. On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections.(非NVSwitch使能的系统可以8个互联，那NVSwitch使能的系统几个设备可以互联)

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

###### 3.2.6.4.1. IOMMU on Linux 

​        On Linux only, CUDA and the display driver does not support IOMMU-enabled bare-metal PCIe peer to peer memory copy. However, CUDA and the display driver does support IOMMU via VM pass through. As a consequence, users on Linux, when running on a native bare metal system, should disable the IOMMU. The IOMMU should be enabled and the VFIO driver be used as a PCIe pass through for virtual machines.

​        (IOMMU是啥？ bare metal system是裸机的意思？)

​        On Windows the above limitation does not exist. 

​        See also Allocating DMA Buffers on 64-bit Platforms. 

##### 3.2.6.5. Peer-to-Peer Memory Copy 

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

​        Note that if peer-to-peer access is enabled between two devices via cudaDeviceEnablePeerAccess() as described in Peer-to-Peer Memory Access, peer-to-peer memory copy between these two devices no longer needs to be staged through the host and is therefore faster. 

#### 3.2.7. Unified Virtual Address Space 

​         When the application is run as a 64-bit process, a single address space is used for the host and all the devices of compute capability 2.0 and higher. All host memory allocations made via CUDA API calls and all device memory allocations on supported devices are within this virtual address range. As a consequence: 

1、The location of any memory on the host allocated through CUDA, or on any of the devices which use the unified address space, can be determined from the value of the pointer using cudaPointerGetAttributes(). 

2、When copying to or from the memory of any device which uses the unified address space, the cudaMemcpyKind parameter of cudaMemcpy*() can be set to cudaMemcpyDefault to determine locations from the pointers. This also works for host pointers not allocated through CUDA, as long as the current device uses unified addressing. 

3、Allocations via cudaHostAlloc() are automatically portable (see Portable Memory) across all the devices for which the unified address space is used, and pointers returned by cudaHostAlloc() can be used directly from within kernels running on these devices (i.e., there is no need to obtain a device pointer via cudaHostGetDevicePointer() as described in Mapped Memory. (如果是映射存储器，直接在kernel中用)

​         Applications may query if the unified address space is used for a particular device by checking that the unifiedAddressing device property (see Device Enumeration) is equal to 1. 

#### 3.2.8. Interprocess Communication (进程间通信)

​        Any device memory pointer or event handle created by a host thread can be directly referenced by any other thread within the same process. It is not valid outside this process however, and therefore cannot be directly referenced by threads belonging to a different process.

​         To share device memory pointers and events across processes, an application must use the Inter Process Communication API, which is described in detail in the reference manual. The IPC API is only supported for 64-bit processes on Linux and for devices of compute capability 2.0 and higher. Note that the IPC API is not supported for cudaMallocManaged allocations. 

​         Using this API, an application can get the IPC handle for a given device memory pointer using cudaIpcGetMemHandle(), pass it to another process using standard IPC mechanisms (e.g., interprocess shared memory or files), and use cudaIpcOpenMemHandle() to retrieve a device pointer from the IPC handle that is a valid pointer within this other process. Event handles can be shared using similar entry points. 

​        An example of using the IPC API is where a single master process generates a batch of input data, making the data available to multiple slave processes without requiring regeneration or copying.    

​         Applications using CUDA IPC to communicate with each other should be compiled, linked, and run with the same CUDA driver and runtime. 

```
CUDA IPC calls are not supported on Tegra devices.
```

#### 3.2.9. Error Checking 

​        All runtime functions return an error code, but for an asynchronous function (see Asynchronous Concurrent Execution), this error code cannot possibly report any of the asynchronous errors that could occur on the device since the function returns before the device has completed the task; the error code only reports errors that occur on the host prior to executing the task, typically related to parameter validation; if an asynchronous error occurs, it will be reported by some subsequent unrelated runtime function call. 

​        The only way to check for asynchronous errors just after some asynchronous function call is therefore to synchronize just after the call by calling cudaDeviceSynchronize() (or by using any other synchronization mechanisms described in Asynchronous Concurrent Execution) and checking the error code returned by cudaDeviceSynchronize(). 

​        The runtime maintains an error variable for each host thread that is initialized to cudaSuccess and is overwritten by the error code every time an error occurs (be it a parameter validation error or an asynchronous error). cudaPeekAtLastError() returns this variable. cudaGetLastError() returns this variable and resets it to cudaSuccess.

​        Kernel launches do not return any error code, so cudaPeekAtLastError() or cudaGetLastError() must be called just after the kernel launch to retrieve any pre-launch errors. To ensure that any error returned by cudaPeekAtLastError() or cudaGetLastError() does not originate from calls prior to the kernel launch, one has to make sure that the runtime error variable is set to cudaSuccess just before the kernel launch, for example, by calling cudaGetLastError() just before the kernel launch. Kernel launches are asynchronous, so to check for asynchronous errors, the application must synchronize in-between the kernel launch and the call to cudaPeekAtLastError() or cudaGetLastError().

​        Note that cudaErrorNotReady that may be returned by cudaStreamQuery() and cudaEventQuery() is not considered an error and is therefore not reported by cudaPeekAtLastError() or cudaGetLastError(). 

#### 3.2.10. Call Stack 

​        On devices of compute capability 2.x and higher, the size of the call stack can be queried using cudaDeviceGetLimit() and set using cudaDeviceSetLimit(). 

​        When the call stack overflows, the kernel call fails with a stack overflow error if the application is run via a CUDA debugger (cuda-gdb, Nsight) or an unspecified launch error, otherwise. 

#### 3.2.11. Texture and Surface Memory 

​        CUDA supports a subset of the texturing hardware that the GPU uses for graphics to access texture and surface memory. Reading data from texture or surface memory instead of global memory can have several performance benefits as described in Device Memory Accesses. 

​        There are two different APIs to access texture and surface memory: 

1、The texture reference API that is supported on all devices, 

2、The texture object API that is only supported on devices of compute capability 3.x and higher. 

​        The texture reference API has limitations that the texture object API does not have. They are mentioned in Texture Reference API. 

##### 3.2.11.1. Texture Memory 

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

###### 3.2.11.1.1. Texture Object API

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

###### 3.2.11.1.2. Texture Reference API

​         Some of the attributes of a texture reference are immutable and must be known at compile time; they are specified when declaring the texture reference. A texture reference is declared at file scope as a variable of type texture:

```c++
texture<DataType, Type, ReadMode> texRef;
```

where:
1、DataType specifies the type of the texel;
2、Type specifies the type of the texture reference and is equal to
cudaTextureType1D, cudaTextureType2D, or cudaTextureType3D, for a
one-dimensional, two-dimensional, or three-dimensional texture, respectively,
or cudaTextureType1DLayered or cudaTextureType2DLayered for a onedimensional or two-dimensional layered texture respectively; Type is an optional
argument which defaults to cudaTextureType1D;
3、ReadMode specifies the read mode; it is an optional argument which defaults to
cudaReadModeElementType.

​         A texture reference can only be declared as a static global variable and cannot be passed as an argument to a function.
​        The other attributes of a texture reference are mutable and can be changed at runtime through the host runtime. As explained in the reference manual, the runtime API has a low-level C-style interface and a high-level C++-style interface. The texture type is defined in the high-level API as a structure publicly derived from the textureReference type defined in the low-level API as such:

```c++
struct textureReference {
    int normalized;
    enum cudaTextureFilterMode filterMode;
    enum cudaTextureAddressMode addressMode[3];
    struct cudaChannelFormatDesc channelDesc;
    int sRGB;
    unsigned int maxAnisotropy;
    enum cudaTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
}
```

1、 normalized specifies whether texture coordinates are normalized or not;
2、 filterMode specifies the filtering mode;
3、 addressMode specifies the addressing mode;
4、 channelDesc describes the format of the texel; it must match the DataType argument of the texture reference declaration; channelDesc is of the following type:

```c++
struct cudaChannelFormatDesc {
    int x, y, z, w;
    enum cudaChannelFormatKind f;
};
```

where x, y, z, and w are equal to the number of bits of each component of the returned value and f is:  

​    4.1、cudaChannelFormatKindSigned if these components are of signed integer type,
​    4.2、cudaChannelFormatKindUnsigned if they are of unsigned integer type,
​    4.3、cudaChannelFormatKindFloat if they are of floating point type.

5、See reference manual for sRGB, maxAnisotropy, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, and maxMipmapLevelClamp.

normalized, addressMode, and filterMode may be directly modified in host code.

Before a kernel can use a texture reference to read from texture memory, the texture reference must be bound to a texture using cudaBindTexture() or cudaBindTexture2D() for linear memory, or cudaBindTextureToArray() for CUDA arrays. cudaUnbindTexture() is used to unbind a texture reference. Once a texture reference has been unbound, it can be safely rebound to another array, even if kernels that use the previously bound texture have not completed. It is recommended to allocate two-dimensional textures in linear memory using cudaMallocPitch() and use the pitch returned by cudaMallocPitch() as input parameter to cudaBindTexture2D(). The following code samples bind a 2D texture reference to linear memory pointed to by devPtr:             

1、Using the low-level API:

```c++
texture<float, cudaTextureType2D,
cudaReadModeElementType> texRef;
textureReference* texRefPtr;
cudaGetTextureReference(&texRefPtr, &texRef);
cudaChannelFormatDesc channelDesc =
cudaCreateChannelDesc<float>();
size_t offset;
cudaBindTexture2D(&offset, texRefPtr, devPtr, &channelDesc,
width, height, pitch);
```

2、 Using the high-level API:

```c++
texture<float, cudaTextureType2D,
cudaReadModeElementType> texRef;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
size_t offset;
cudaBindTexture2D(&offset, texRef, devPtr, channelDesc, width, height, pitch);
```

The following code samples bind a 2D texture reference to a CUDA array cuArray:

1、 Using the low-level API:

```c++
texture<float, cudaTextureType2D,
cudaReadModeElementType> texRef;
textureReference* texRefPtr;
cudaGetTextureReference(&texRefPtr, &texRef);
cudaChannelFormatDesc channelDesc;
cudaGetChannelDesc(&channelDesc, cuArray);
cudaBindTextureToArray(texRef, cuArray, &channelDesc);
```

2、 Using the high-level API:

```c++
texture<float, cudaTextureType2D,
cudaReadModeElementType> texRef;
cudaBindTextureToArray(texRef, cuArray);
```

​         The format specified when binding a texture to a texture reference must match the parameters specified when declaring the texture reference; otherwise, the results of texture fetches are undefined.
​        There is a limit to the number of textures that can be bound to a kernel as specified in Table 15.
​        The following code sample applies some simple transformation kernel to a texture.

```c++
// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
// Simple transformation kernel
__global__ void transformKernel(float* output, int width, int height, float theta)
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
	output[y * width + x] = tex2D(texRef, tu, tv);
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
	cudaMemcpyToArray(cuArray, 0, 0, h_data, size,
	cudaMemcpyHostToDevice);
	// Set texture reference parameters
	texRef.addressMode[0] = cudaAddressModeWrap;
	texRef.addressMode[1] = cudaAddressModeWrap;
	texRef.filterMode = cudaFilterModeLinear;
	texRef.normalized = true;
	// Bind the array to the texture reference
	cudaBindTextureToArray(texRef, cuArray, channelDesc);
	// Allocate result of transformation in device memory
	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	// Invoke kernel
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
				 (height + dimBlock.y - 1) / dimBlock.y);
	transformKernel<<<dimGrid, dimBlock>>>(output, width, height, angle);
	// Free device memory
	cudaFreeArray(cuArray);
	cudaFree(output);
	return 0;
}
```

###### 3.2.11.1.3 16-Bit Floating-Point Textures 

The 16-bit floating-point or half format supported by CUDA arrays is the same as the IEEE 754-2008 binary2 format. 

CUDA C++ does not support a matching data type, but provides intrinsic functions to convert to and from the 32-bit floating-point format via the unsigned short type: __float2half_rn(float) and __half2float(unsigned short). These functions are only supported in device code. Equivalent functions for the host code can be found in the OpenEXR library, for example. 

16-bit floating-point components are promoted to 32 bit float during texture fetching before any filtering is performed.

 A channel description for the 16-bit floating-point format can be created by calling one of the cudaCreateChannelDescHalf*() functions.

######  3.2.11.1.4 Layered Textures 

A one-dimensional or two-dimensional layered texture (also known as texture array in Direct3D and array texture in OpenGL) is a texture made up of a sequence of layers, all of which are regular textures of same dimensionality, size, and data type. 

A one-dimensional layered texture is addressed using an integer index and a floatingpoint texture coordinate; the index denotes a layer within the sequence and the coordinate addresses a texel within that layer. A two-dimensional layered texture is addressed using an integer index and two floating-point texture coordinates; the index denotes a layer within the sequence and the coordinates address a texel within that layer.

 A layered texture can only be a CUDA array by calling cudaMalloc3DArray() with the cudaArrayLayered flag (and a height of zero for one-dimensional layered texture).

Layered textures are fetched using the device functions described in tex1DLayered(), tex1DLayered(), tex2DLayered(), and tex2DLayered(). Texture filtering (see Texture Fetching) is done only within a layer, not across layers. 

Layered textures are only supported on devices of compute capability 2.0 and higher. 

###### 3.2.11.1.5. Cubemap Textures 

A cubemap texture is a special type of two-dimensional layered texture that has six layers representing the faces of a cube: 

1、The width of a layer is equal to its height. 

2、The cubemap is addressed using three texture coordinates x, y, and z that are interpreted as a direction vector emanating from the center of the cube and pointing to one face of the cube and a texel within the layer corresponding to that face. More specifically, the face is selected by the coordinate with largest magnitude m and the corresponding layer is addressed using coordinates (s/m+1)/2 and (t/m+1)/2 where s and t are defined in Table 2. 

A layered texture can only be a CUDA array by calling cudaMalloc3DArray() with the cudaArrayCubemap flag.
Cubemap textures are fetched using the device function described in texCubemap() and texCubemap().
Cubemap textures are only supported on devices of compute capability 2.0 and higher.

###### 3.2.11.1.6. Cubemap Layered Textures

​         A cubemap layered texture is a layered texture whose layers are cubemaps of same dimension. 

​         A cubemap layered texture is addressed using an integer index and three floatingpoint texture coordinates; the index denotes a cubemap within the sequence and the coordinates address a texel within that cubemap.

​        A layered texture can only be a CUDA array by calling cudaMalloc3DArray() with the cudaArrayLayered and cudaArrayCubemap flags. 

​        Cubemap layered textures are fetched using the device function described in texCubemapLayered() and texCubemapLayered(). Texture filtering (see Texture Fetching) is done only within a layer, not across layers.

​      Cubemap layered textures are only supported on devices of compute capability 2.0 and higher. 

###### 3.2.11.1.7. Texture Gather 

​         Texture gather is a special texture fetch that is available for two-dimensional textures only. It is performed by the tex2Dgather() function, which has the same parameters as tex2D(), plus an additional comp parameter equal to 0, 1, 2, or 3 (see tex2Dgather() and tex2Dgather()). It returns four 32-bit numbers that correspond to the value of the component comp of each of the four texels that would have been used for bilinear filtering during a regular texture fetch. For example, if these texels are of values (253, 20, 31, 255), (250, 25, 29, 254), (249, 16, 37, 253), (251, 22, 30, 250), and comp is 2, tex2Dgather() returns (31, 29, 37, 30). 

​         Note that texture coordinates are computed with only 8 bits of fractional precision. tex2Dgather() may therefore return unexpected results for cases where tex2D() would use 1.0 for one of its weights (α or β, see Linear Filtering). For example, with an x texture coordinate of 2.49805: xB=x-0.5=1.99805, however the fractional part of xB is stored in an 8-bit fixed-point format. Since 0.99805 is closer to 256.f/256.f than it is to 255.f/256.f, xB has the value 2. A tex2Dgather() in this case would therefore return indices 2 and 3 in x, instead of indices 1 and 2. 

​        Texture gather is only supported for CUDA arrays created with the cudaArrayTextureGather flag and of width and height less than the maximum specified in Table 15 for texture gather, which is smaller than for regular texture fetch.

​        Texture gather is only supported on devices of compute capability 2.0 and higher. 

##### 3.2.11.2. Surface Memory

For devices of compute capability 2.0 and higher, a CUDA array (described in Cubemap Surfaces), created with the cudaArraySurfaceLoadStore flag, can be read and written via a surface object or surface reference using the functions described in Surface Functions.
Table 15 lists the maximum surface width, height, and depth depending on the compute capability of the device.

###### 3.2.11.2.1. Surface Object API

A surface object is created using cudaCreateSurfaceObject() from a resource description of type struct cudaResourceDesc.
The following code sample applies some simple transformation kernel to a texture.

```c++
// Simple copy kernel
__global__ void copyKernel(cudaSurfaceObject_t inputSurfObj,
cudaSurfaceObject_t outputSurfObj,
int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		uchar4 data;
		// Read from input surface
		surf2Dread(&data, inputSurfObj, x * 4, y);
		// Write to output surface
		surf2Dwrite(data, outputSurfObj, x * 4, y);
	}
}

// Host code
int main()
{
	// Allocate CUDA arrays in device memory
	cudaChannelFormatDesc channelDesc =
	cudaCreateChannelDesc(8, 8, 8, 8,
	cudaChannelFormatKindUnsigned);
	cudaArray* cuInputArray;
	cudaMallocArray(&cuInputArray, &channelDesc, width, height,
	cudaArraySurfaceLoadStore);
	cudaArray* cuOutputArray;
	cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
	cudaArraySurfaceLoadStore);
	// Copy to device memory some data located at address h_data
	// in host memory
	cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size,
	cudaMemcpyHostToDevice);
	// Specify surface
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	// Create the surface objects
	resDesc.res.array.array = cuInputArray;
	cudaSurfaceObject_t inputSurfObj = 0;
	cudaCreateSurfaceObject(&inputSurfObj, &resDesc);
	resDesc.res.array.array = cuOutputArray;
	cudaSurfaceObject_t outputSurfObj = 0;
	cudaCreateSurfaceObject(&outputSurfObj, &resDesc);
	// Invoke kernel
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
	             (height + dimBlock.y - 1) / dimBlock.y);
	copyKernel<<<dimGrid, dimBlock>>>(inputSurfObj, outputSurfObj, width, height);
	// Destroy surface objects
	cudaDestroySurfaceObject(inputSurfObj);
	cudaDestroySurfaceObject(outputSurfObj);
	// Free device memory
	cudaFreeArray(cuInputArray);
	cudaFreeArray(cuOutputArray);
	return 0;
}
```

###### 3.2.11.2.2. Surface Reference API 

A surface reference is declared at file scope as a variable of type surface: 

```c++
surface<void, Type> surfRef; 
```

​         where Type specifies the type of the surface reference and is equal to cudaSurfaceType1D, cudaSurfaceType2D, cudaSurfaceType3D, cudaSurfaceTypeCubemap, cudaSurfaceType1DLayered, cudaSurfaceType2DLayered, or cudaSurfaceTypeCubemapLayered; Type is an optional argument which defaults to cudaSurfaceType1D. A surface reference can only be declared as a static global variable and cannot be passed as an argument to a function. 

​         Before a kernel can use a surface reference to access a CUDA array, the surface reference must be bound to the CUDA array using cudaBindSurfaceToArray(). 

​        The following code samples bind a surface reference to a CUDA array cuArray:

 1、Using the low-level API: 

```c++
surface<void, cudaSurfaceType2D> surfRef; 
surfaceReference* surfRefPtr; 
cudaGetSurfaceReference(&surfRefPtr, "surfRef");
cudaChannelFormatDesc channelDesc; 
cudaGetChannelDesc(&channelDesc, cuArray); 
cudaBindSurfaceToArray(surfRef, cuArray, &channelDesc); 
```

2、Using the high-level API:

```c++
surface<void, cudaSurfaceType2D> surfRef; 
cudaBindSurfaceToArray(surfRef, cuArray);
```

​        A CUDA array must be read and written using surface functions of matching dimensionality and type and via a surface reference of matching dimensionality; otherwise, the results of reading and writing the CUDA array are undefined. 

​        Unlike texture memory, surface memory uses byte addressing. This means that the x-coordinate used to access a texture element via texture functions needs to be multiplied by the byte size of the element to access the same element via a surface function. For example, the element at texture coordinate x of a one-dimensional floating-point CUDA array bound to a texture reference texRef and a surface reference surfRef is read using tex1d(texRef, x) via texRef, but surf1Dread(surfRef, 4*x) via surfRef. Similarly, the element at texture coordinate x and y of a twodimensional floating-point CUDA array bound to a texture reference texRef and a surface reference surfRef is accessed using tex2d(texRef, x, y) via texRef, but surf2Dread(surfRef, 4*x, y) via surfRef (the byte offset of the y-coordinate is internally calculated from the underlying line pitch of the CUDA array). 

The following code sample applies some simple transformation kernel to a texture.  

```c++
// 2D surfaces
surface<void, 2> inputSurfRef;
surface<void, 2> outputSurfRef;
// Simple copy kernel
__global__ void copyKernel(int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		uchar4 data;
		// Read from input surface
		surf2Dread(&data, inputSurfRef, x * 4, y);
		// Write to output surface
		surf2Dwrite(data, outputSurfRef, x * 4, y);
	}
}

// Host code
int main()
{
	// Allocate CUDA arrays in device memory
	cudaChannelFormatDesc channelDesc =
	cudaCreateChannelDesc(8, 8, 8, 8,
	cudaChannelFormatKindUnsigned);
	cudaArray* cuInputArray;
	cudaMallocArray(&cuInputArray, &channelDesc, width, height,
	cudaArraySurfaceLoadStore);
	cudaArray* cuOutputArray;
	cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
	cudaArraySurfaceLoadStore);
	// Copy to device memory some data located at address h_data
	// in host memory
	cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size, cudaMemcpyHostToDevice);
	// Bind the arrays to the surface references
	cudaBindSurfaceToArray(inputSurfRef, cuInputArray);
	cudaBindSurfaceToArray(outputSurfRef, cuOutputArray);
	// Invoke kernel
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
	             (height + dimBlock.y - 1) / dimBlock.y);
	copyKernel<<<dimGrid, dimBlock>>>(width, height);
	// Free device memory
	cudaFreeArray(cuInputArray);
	cudaFreeArray(cuOutputArray);
	return 0;
}
```

###### 3.2.11.2.3. Cubemap Surfaces 

Cubemap surfaces are accessed usingsurfCubemapread() and surfCubemapwrite() (surfCubemapread and surfCubemapwrite) as a two-dimensional layered surface, i.e., using an integer index denoting a face and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. Faces are ordered as indicated in Table 2. 

###### 3.2.11.2.4. Cubemap Layered Surfaces 

Cubemap layered surfaces are accessed using surfCubemapLayeredread() and surfCubemapLayeredwrite() (surfCubemapLayeredread() and surfCubemapLayeredwrite()) as a two-dimensional layered surface, i.e., using an integer index denoting a face of one of the cubemaps and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. Faces are ordered as indicated in Table 2, so index ((2 * 6) + 3), for example, accesses the fourth face of the third cubemap. 

##### 3.2.11.3. CUDA Arrays

 CUDA arrays are opaque memory layouts optimized for texture fetching. They are one dimensional, two dimensional, or three-dimensional and composed of elements, each of which has 1, 2 or 4 components that may be signed or unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats. CUDA arrays are only accessible by kernels through texture fetching as described in Texture Memory or surface reading and writing as described in Surface Memory. 

##### 3.2.11.4. Read/Write Coherency 

The texture and surface memory is cached (see Device Memory Accesses) and within the same kernel call, the cache is not kept coherent with respect to global memory writes and surface memory writes, so any texture fetch or surface read to an address that has been written to via a global write or a surface write in the same kernel call returns undefined data. In other words, a thread can safely read some texture or surface memory location only if this memory location has been updated by a previous kernel call or memory copy, but not if it has been previously updated by the same thread or another thread from the same kernel call. 

#### 3.2.12. Graphics Interoperability 

​         Some resources from OpenGL and Direct3D may be mapped into the address space of CUDA, either to enable CUDA to read data written by OpenGL or Direct3D, or to enable CUDA to write data for consumption by OpenGL or Direct3D.

​         A resource must be registered to CUDA before it can be mapped using the functions mentioned in OpenGL Interoperability and Direct3D Interoperability. These functions return a pointer to a CUDA graphics resource of type struct cudaGraphicsResource. Registering a resource is potentially high-overhead and therefore typically called only once per resource. A CUDA graphics resource is unregistered using cudaGraphicsUnregisterResource(). Each CUDA context which intends to use the resource is required to register it separately. 

​        Once a resource is registered to CUDA, it can be mapped and unmapped as many times as necessary using cudaGraphicsMapResources() and cudaGraphicsUnmapResources(). cudaGraphicsResourceSetMapFlags() can be called to specify usage hints (write-only, read-only) that the CUDA driver can use to optimize resource management.

​        A mapped resource can be read from or written to by kernels using the device memory address returned by cudaGraphicsResourceGetMappedPointer() for buffers and cudaGraphicsSubResourceGetMappedArray() for CUDA arrays. 

​        Accessing a resource through OpenGL, Direct3D, or another CUDA context while it is mapped produces undefined results. OpenGL Interoperability and Direct3D Interoperability give specifics for each graphics API and some code samples. SLI Interoperability gives specifics for when the system is in SLI mode. 

##### 3.2.12.1. OpenGL Interoperability 

​         The OpenGL resources that may be mapped into the address space of CUDA are OpenGL buffer, texture, and renderbuffer objects.

​          A buffer object is registered using cudaGraphicsGLRegisterBuffer(). In CUDA, it appears as a device pointer and can therefore be read and written by kernels or via cudaMemcpy() calls.

​          A texture or renderbuffer object is registered using cudaGraphicsGLRegisterImage(). In CUDA, it appears as a CUDA array. Kernels can read from the array by binding it to a texture or surface reference. They can also write to it via the surface write functions if the resource has been registered with the cudaGraphicsRegisterFlagsSurfaceLoadStore flag. The array can also be read and written via cudaMemcpy2D() calls. cudaGraphicsGLRegisterImage() supports all texture formats with 1, 2, or 4 components and an internal type of float (e.g., GL_RGBA_FLOAT32), normalized integer (e.g., GL_RGBA8, GL_INTENSITY16), and unnormalized integer (e.g., GL_RGBA8UI) (please note that since unnormalized integer formats require OpenGL 3.0, they can only be written by shaders, not the fixed function pipeline). 

​        The OpenGL context whose resources are being shared has to be current to the host thread making any OpenGL interoperability API calls.

​        Please note: When an OpenGL texture is made bindless (say for example by requesting an image or texture handle using the glGetTextureHandle*/glGetImageHandle* APIs) it cannot be registered with CUDA. The application needs to register the texture for interop before requesting an image or texture handle. 

​        The following code sample uses a kernel to dynamically modify a 2D width x height
grid of vertices stored in a vertex buffer object:  

```c++
GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;
int main()
{
	// Initialize OpenGL and GLUT for device 0
	// and make the OpenGL context current
	...
	glutDisplayFunc(display);
	// Explicitly set device 0
	cudaSetDevice(0);
	// Create buffer object and register it with CUDA
	glGenBuffers(1, &positionsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	unsigned int size = width * height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA,
	positionsVBO,
	cudaGraphicsMapFlagsWriteDiscard);
	// Launch rendering loop
	glutMainLoop();
	...
}
void display()
{
	// Map buffer object for writing from CUDA
	float4* positions;
	cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&positions,
	&num_bytes,
	positionsVBO_CUDA));
	// Execute kernel
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
	createVertices<<<dimGrid, dimBlock>>>(positions, time,
	width, height);
	// Unmap buffer object
	cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
	// Render from buffer object
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);
	// Swap buffers
	glutSwapBuffers();
	glutPostRedisplay();
}

void deleteVBO()
{
	cudaGraphicsUnregisterResource(positionsVBO_CUDA);
	glDeleteBuffers(1, &positionsVBO);
}

__global__ void createVertices(float4* positions, float time,
unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	// Calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u * freq + time)
	* cosf(v * freq + time) * 0.5f;
	// Write positions
	positions[y * width + x] = make_float4(u, w, v, 1.0f);
}
```

On Windows and for Quadro GPUs, cudaWGLGetDevice() can be used to retrieve the CUDA device associated to the handle returned by wglEnumGpusNV(). Quadro GPUs offer higher performance OpenGL interoperability than GeForce and Tesla GPUs in a multi-GPU configuration where OpenGL rendering is performed on the Quadro GPU and CUDA computations are performed on other GPUs in the system. 

##### 3.2.12.2. Direct3D Interoperability 

Direct3D interoperability is supported for Direct3D 9Ex, Direct3D 10, and Direct3D 11.

 A CUDA context may interoperate only with Direct3D devices that fulfill the following criteria: Direct3D 9Ex devices must be created with DeviceType set to D3DDEVTYPE_HAL and BehaviorFlags with the D3DCREATE_HARDWARE_VERTEXPROCESSING flag; Direct3D 10 and Direct3D 11 devices must be created with DriverType set to D3D_DRIVER_TYPE_HARDWARE.

 The Direct3D resources that may be mapped into the address space of CUDA are Direct3D buffers, textures, and surfaces. These resources are registered using cudaGraphicsD3D9RegisterResource(), cudaGraphicsD3D10RegisterResource(), and cudaGraphicsD3D11RegisterResource(). 

The following code sample uses a kernel to dynamically modify a 2D width x height grid of vertices stored in a vertex buffer object. 

###### 3.2.12.2.1. Direct3D 9 Version  

```c++
IDirect3D9* D3D;
IDirect3DDevice9* device;
struct CUSTOMVERTEX {
	FLOAT x, y, z;
	DWORD color;
};
IDirect3DVertexBuffer9* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;
int main()
{
	int dev;
	// Initialize Direct3D
	D3D = Direct3DCreate9Ex(D3D_SDK_VERSION);
	// Get a CUDA-enabled adapter
	unsigned int adapter = 0;
	for (; adapter < g_pD3D->GetAdapterCount(); adapter++) {
		D3DADAPTER_IDENTIFIER9 adapterId;
		g_pD3D->GetAdapterIdentifier(adapter, 0, &adapterId);
		if (cudaD3D9GetDevice(&dev, adapterId.DeviceName)
		== cudaSuccess)
		break;
	}
	// Create device
	...
	D3D->CreateDeviceEx(adapter, D3DDEVTYPE_HAL, hWnd,
						D3DCREATE_HARDWARE_VERTEXPROCESSING,
						&params, NULL, &device);
	// Use the same device
	cudaSetDevice(dev);
	// Create vertex buffer and register it with CUDA
	unsigned int size = width * height * sizeof(CUSTOMVERTEX);
	device->CreateVertexBuffer(size, 0, D3DFVF_CUSTOMVERTEX, D3DPOOL_DEFAULT, &positionsVB, 0);
	cudaGraphicsD3D9RegisterResource(&positionsVB_CUDA,
									 positionsVB,
									 cudaGraphicsRegisterFlagsNone);
	cudaGraphicsResourceSetMapFlags(positionsVB_CUDA, cudaGraphicsMapFlagsWriteDiscard);
	// Launch rendering loop
	while (...) {
		...
		Render();
		...
	}
	...
}

void Render()
{
	// Map vertex buffer for writing from CUDA
	float4* positions;
	cudaGraphicsMapResources(1, &positionsVB_CUDA, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&positions,
	&num_bytes,
	positionsVB_CUDA));
	// Execute kernel
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
	createVertices<<<dimGrid, dimBlock>>>(positions, time, width, height);
	// Unmap vertex buffer
	cudaGraphicsUnmapResources(1, &positionsVB_CUDA, 0);
	// Draw and present
	...
}
void releaseVB()
{
	cudaGraphicsUnregisterResource(positionsVB_CUDA);
	positionsVB->Release();
}
__global__ void createVertices(float4* positions, float time,
unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	// Calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
	// Calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u * freq + time)
	* cosf(v * freq + time) * 0.5f;
	// Write positions
	positions[y * width + x] =
	make_float4(u, w, v, __int_as_float(0xff00ff00));
}

```

###### 3.2.12.2.2. Direct3D 10 Version  

###### 3.2.12.2.3. Direct3D 11 Version  

##### 3.2.12.3. SLI Interoperability 

​        In a system with multiple GPUs, all CUDA-enabled GPUs are accessible via the CUDA driver and runtime as separate devices. There are however special considerations as described below when the system is in SLI mode.

​        First, an allocation in one CUDA device on one GPU will consume memory on other GPUs that are part of the SLI configuration of the Direct3D or OpenGL device. Because of this, allocations may fail earlier than otherwise expected. 

​        Second, applications should create multiple CUDA contexts, one for each GPU in the SLI configuration. While this is not a strict requirement, it avoids unnecessary data transfers between devices. The application can use the cudaD3D[9|10|11]GetDevices() for Direct3D and cudaGLGetDevices() for OpenGL set of calls to identify the CUDA device handle(s) for the device(s) that are performing the rendering in the current and next frame. Given this information the application will typically choose the appropriate device and map Direct3D or OpenGL resources to the CUDA device returned by cudaD3D[9|10|11]GetDevices() or cudaGLGetDevices() when the deviceList parameter is set to cudaD3D[9|10|11]DeviceListCurrentFrame or cudaGLDeviceListCurrentFrame.

​       Please note that resource returned from cudaGraphicsD9D[9|10| 11]RegisterResource and cudaGraphicsGLRegister[Buffer|Image] must be only used on device the registration happened. Therefore on SLI configurations when data for different frames is computed on different CUDA devices it is necessary to register the resources for each separatly. 

See Direct3D Interoperability and OpenGL Interoperability for details on how the CUDA runtime interoperate with Direct3D and OpenGL, respectively.

 3.3. External Resource Interoperability 

External resource interoperability allows CUDA to import certain resources that are explicitly exported by other APIs. These objects are typically exported by other APIs using handles native to the Operating System, like file descriptors on Linux or NT handles on Windows. They could also be exported using other unified interfaces such as the NVIDIA Software Communication Interface. There are two types of resources that can be imported: memory objects and synchronization objects. 

Memory objects can be imported into CUDA using cudaImportExternalMemory(). An imported memory object can be accessed from within kernels using device pointers mapped onto the memory object via cudaExternalMemoryGetMappedBuffer()or CUDA mipmapped arrays mapped via cudaExternalMemoryGetMappedMipmappedArray(). Depending on the type of memory object, it may be possible for more than one mapping to be setup on a single memory object. The mappings must match the mappings setup in the exporting API. Any mismatched mappings result in undefined behavior. Imported memory objects must be freed using cudaDestroyExternalMemory(). Freeing a memory object does not free any mappings to that object. Therefore, any device pointers mapped onto that object must be explicitly freed using cudaFree() and any CUDA mipmapped arrays mapped onto that object must be explicitly freed using cudaFreeMipmappedArray(). It is illegal to access mappings to an object after it has been destroyed. \

​         Synchronization objects can be imported into CUDA using cudaImportExternalSemaphore(). An imported synchronization object can then be signaled using cudaSignalExternalSemaphoresAsync() and waited on using cudaWaitExternalSemaphoresAsync(). It is illegal to issue a wait before the corresponding signal has been issued. Also, depending on the type of the imported synchronization object, there may be additional constraints imposed on how they can be signaled and waited on, as described in subsequent sections. Imported semaphore objects must be freed using cudaDestroyExternalSemaphore(). All outstanding signals and waits must have completed before the semaphore object is destroyed. 

#### 3.3.1. Vulcan Interoperability

##### 3.3.1.1. Matching device UUIDs 

​        When importing memory and synchronization objects exported by Vulkan, they must be imported and mapped on the same device as they were created on. The CUDA device that corresponds to the Vulkan physical device on which the objects were created can be determined by comparing the UUID of a CUDA device with that of the Vulkan physical device, as shown in the following code sample. Note that the Vulkan physical device should not be part of a device group that contains more than one Vulkan physical device. The device group as returned by vkEnumeratePhysicalDeviceGroups that contains the given Vulkan physical device must have a physical device count of 1. 

```c++
int getCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice) {
	VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
	vkPhysicalDeviceIDProperties.sType =
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
	vkPhysicalDeviceIDProperties.pNext = NULL;
	VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
	vkPhysicalDeviceProperties2.sType =
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;
	vkGetPhysicalDeviceProperties2(vkPhysicalDevice,
	&vkPhysicalDeviceProperties2);
	int cudaDeviceCount;
	cudaGetDeviceCount(&cudaDeviceCount);
	for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, cudaDevice);
		if (!memcmp(&deviceProp.uuid, vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE)) {
			return cudaDevice;
		}
	}
	return cudaInvalidDeviceId;
}
```

##### 3.3.1.2. Importing memory objects 

​        On Linux and Windows 10, both dedicated and non-dedicated memory objects exported by Vulkan can be imported into CUDA. On Windows 7, only dedicated memory objects can be imported. When importing a Vulkan dedicated memory object, the flag cudaExternalMemoryDedicated must be set.

​          A Vulkan memory object exported using VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT can be imported into CUDA using the file descriptor associated with that object as shown below. Note that CUDA assumes ownership of the file descriptor once it is imported. Using the file descriptor after a successful import results in undefined behavior. 

```c++
cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(int fd, unsigned long long size, bool isDedicated) {
	cudaExternalMemory_t extMem = NULL;
	cudaExternalMemoryHandleDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	desc.handle.fd = fd;
	desc.size = size;
	if (isDedicated) {
		desc.flags |= cudaExternalMemoryDedicated;
	}
	cudaImportExternalMemory(&extMem, &desc);
	// Input parameter 'fd' should not be used beyond this point as CUDA has
	assumed ownership of it
	return extMem;
}
```

​        A Vulkan memory object exported using VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT can be imported into CUDA using the NT handle associated with that object as shown below. Note that CUDA does not assume ownership of the NT handle and it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed. 

```c++
cudaExternalMemory_t importVulkanMemoryObjectFromNTHandle(HANDLE handle,
unsigned long long size, bool isDedicated) {
	cudaExternalMemory_t extMem = NULL;
	cudaExternalMemoryHandleDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	desc.handle.win32.handle = handle;
	desc.size = size;
	if (isDedicated) {
		desc.flags |= cudaExternalMemoryDedicated;
	}
	cudaImportExternalMemory(&extMem, &desc);
	// Input parameter 'handle' should be closed if it's not needed anymore
	CloseHandle(handle);
	return extMem;
}
```

​        A Vulkan memory object exported using VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT can also be imported using a named handle if one exists as shown below.  

```c++
cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(LPCWSTR name, unsigned long long size, bool isDedicated) {
	cudaExternalMemory_t extMem = NULL;
	cudaExternalMemoryHandleDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	desc.handle.win32.name = (void *)name;
	desc.size = size;
	if (isDedicated) {
		desc.flags |= cudaExternalMemoryDedicated;
	}
	cudaImportExternalMemory(&extMem, &desc);
	return extMem;
}
```

​        A Vulkan memory object exported using VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT can be imported into CUDA using the globally shared D3DKMT handle associated with that object as shown below. Since a globally shared D3DKMT handle does not hold a reference to the underlying memory it is automatically destroyed when all other references to the resource are destroyed.  

```c++
cudaExternalMemory_t importVulkanMemoryObjectFromKMTHandle(HANDLE handle, unsigned long long size, bool isDedicated) {
	cudaExternalMemory_t extMem = NULL;
	cudaExternalMemoryHandleDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
	desc.handle.win32.handle = (void *)handle;
	desc.size = size;
	if (isDedicated) {
		desc.flags |= cudaExternalMemoryDedicated;
	}
	cudaImportExternalMemory(&extMem, &desc);
	return extMem;
}
```

##### 3.3.1.3. Mapping buffers onto imported memory objects

​        A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping must match that specified when creating the mapping using the corresponding Vulkan API. All mapped device pointers must be freed using cudaFree().

```c++
void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
	void *ptr = NULL;
	cudaExternalMemoryBufferDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.offset = offset;
	desc.size = size;
	cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
	// Note: ‘ptr’ must eventually be freed using cudaFree()
	return ptr;
}
```

3.3.1.4. Mapping mipmapped arrays onto imported memory objects

​        A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions, format and number of mip levels must match that specified when creating the mapping using the corresponding Vulkan API. Additionally, if the mipmapped array is bound as a color target in Vulkan, the flag cudaArrayColorAttachment must be set. All mapped mipmapped arrays must be freed using cudaFreeMipmappedArray(). The following code sample shows how to convert Vulkan parameters into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects. 

```c++
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, 
														 unsigned long long offset, 
														 cudaChannelFormatDesc *formatDesc,
														 cudaExtent *extent, 
														 unsigned int flags, 
														 unsigned int numLevels) {
	cudaMipmappedArray_t mipmap = NULL;
	cudaExternalMemoryMipmappedArrayDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.offset = offset;
	desc.formatDesc = *formatDesc;
	desc.extent = *extent;
	desc.flags = flags;
	desc.numLevels = numLevels;
	// Note: ‘mipmap’ must eventually be freed using cudaFreeMipmappedArray()
	cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
	return mipmap;
}
cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format)
{
	cudaChannelFormatDesc d;
	memset(&d, 0, sizeof(d));
	switch (format) {
		case VK_FORMAT_R8_UINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindUnsigned; break;
		case VK_FORMAT_R8_SINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindSigned; break;
		case VK_FORMAT_R8G8_UINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindUnsigned; break;
		case VK_FORMAT_R8G8_SINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindSigned; break;
		case VK_FORMAT_R8G8B8A8_UINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8;
		d.f = cudaChannelFormatKindUnsigned; break;
		case VK_FORMAT_R8G8B8A8_SINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8;
		d.f = cudaChannelFormatKindSigned; break;
		case VK_FORMAT_R16_UINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindUnsigned; break;
		case VK_FORMAT_R16_SINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindSigned; break;
		case VK_FORMAT_R16G16_UINT: d.x = 16; d.y = 16; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindUnsigned; break;
		case VK_FORMAT_R16G16_SINT: d.x = 16; d.y = 16; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindSigned; break;
		case VK_FORMAT_R16G16B16A16_UINT: d.x = 16; d.y = 16; d.z = 16; d.w = 16;
		d.f = cudaChannelFormatKindUnsigned; break;
		case VK_FORMAT_R16G16B16A16_SINT: d.x = 16; d.y = 16; d.z = 16; d.w = 16;
		d.f = cudaChannelFormatKindSigned; break;
		case VK_FORMAT_R32_UINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindUnsigned; break;
		case VK_FORMAT_R32_SINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindSigned; break;
		case VK_FORMAT_R32_SFLOAT: d.x = 32; d.y = 0; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindFloat; break;
		case VK_FORMAT_R32G32_UINT: d.x = 32; d.y = 32; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindUnsigned; break;
		case VK_FORMAT_R32G32_SINT: d.x = 32; d.y = 32; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindSigned; break;
		case VK_FORMAT_R32G32_SFLOAT: d.x = 32; d.y = 32; d.z = 0; d.w = 0;
		d.f = cudaChannelFormatKindFloat; break;
		case VK_FORMAT_R32G32B32A32_UINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32;
		d.f = cudaChannelFormatKindUnsigned; break;
		case VK_FORMAT_R32G32B32A32_SINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32;
		d.f = cudaChannelFormatKindSigned; break;
		case VK_FORMAT_R32G32B32A32_SFLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32;
		d.f = cudaChannelFormatKindFloat; break;
		default: assert(0);
	}
	return d;
}

cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers,
										VkImageViewType vkImageViewType) {
	cudaExtent e = { 0, 0, 0 };
	switch (vkImageViewType) {
		case VK_IMAGE_VIEW_TYPE_1D: 
			e.width = vkExt.width; 
			e.height = 0;
			e.depth = 0; 
			break;
		case VK_IMAGE_VIEW_TYPE_2D: 
			e.width = vkExt.width; 
			e.height = vkExt.height; 
			e.depth = 0; 
			break;
		case VK_IMAGE_VIEW_TYPE_3D:
			e.width = vkExt.width; 
			e.height = vkExt.height; 
			e.depth = vkExt.depth; 
			break;
		case VK_IMAGE_VIEW_TYPE_CUBE: 
			e.width = vkExt.width; 
			e.height = vkExt.height; 
			e.depth = arrayLayers; 
			break;
		case VK_IMAGE_VIEW_TYPE_1D_ARRAY: 
			e.width = vkExt.width; 
			e.height = 0;
			e.depth = arrayLayers; 
			break;
		case VK_IMAGE_VIEW_TYPE_2D_ARRAY: 
			e.width = vkExt.width; 
			e.height = vkExt.height; 
			e.depth = arrayLayers; 
			break;
		case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: 
			e.width = vkExt.width; 
			e.height = vkExt.height; 
			e.depth = arrayLayers; 
			break;
		default: assert(0);
	}
	return e;
}
unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType vkImageViewType, 
													  VkImageUsageFlags vkImageUsageFlags, 
													  bool allowSurfaceLoadStore) {
	unsigned int flags = 0;
	switch (vkImageViewType) {
		case VK_IMAGE_VIEW_TYPE_CUBE: 
			flags |= cudaArrayCubemap;
			break;
		case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: 
			flags |= cudaArrayCubemap |
			cudaArrayLayered; break;
		case VK_IMAGE_VIEW_TYPE_1D_ARRAY: 
			flags |= cudaArrayLayered;
			break;
		case VK_IMAGE_VIEW_TYPE_2D_ARRAY: 
			flags |= cudaArrayLayered;
			break;
		default: 
			break;
	}
	if (vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
		flags |= cudaArrayColorAttachment;
	}
	if (allowSurfaceLoadStore) {
		flags |= cudaArraySurfaceLoadStore;
	}
	return flags;
}
```

##### 3.3.1.5. Importing synchronization objects

​        A Vulkan semaphore object exported using VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BITcan be imported into CUDA using the file descriptor associated with that object as shown below. Note that CUDA assumes ownership of the file descriptor once it is imported. Using the file descriptor after a successful import results in undefined behavior.

```c++
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromFileDescriptor(int fd) {
	cudaExternalSemaphore_t extSem = NULL;
	cudaExternalSemaphoreHandleDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
	desc.handle.fd = fd;
	cudaImportExternalSemaphore(&extSem, &desc);
	// Input parameter 'fd' should not be used beyond this point as CUDA has assumed ownership of it
	return extSem;
}
```

​        A Vulkan semaphore object exported using VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT can be imported into CUDA using the NT handle associated with that object as shown below. Note that CUDA does not assume ownership of the NT handle and it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying semaphore can be freed. 

```c++
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNTHandle(HANDLE handle) {
	cudaExternalSemaphore_t extSem = NULL;
	cudaExternalSemaphoreHandleDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	desc.handle.win32.handle = handle;
	cudaImportExternalSemaphore(&extSem, &desc);
	// Input parameter 'handle' should be closed if it's not needed anymore
	CloseHandle(handle);
	return extSem;
}
```

A Vulkan semaphore object exported using VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT can also be imported using a named handle if one exists as shown below.  

```c++
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNamedNTHandle(LPCWSTR
name) {
	cudaExternalSemaphore_t extSem = NULL;
	cudaExternalSemaphoreHandleDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	desc.handle.win32.name = (void *)name;
	cudaImportExternalSemaphore(&extSem, &desc);
	return extSem;
}
```

​         A Vulkan semaphore object exported using VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT can be imported into CUDA using the globally shared D3DKMT handle associated with that object as shown below. Since a globally shared D3DKMT handle does not hold a reference to the underlying semaphore it is automatically destroyed when all other references to the resource are destroyed.  

```c++
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromKMTHandle(HANDLE handle)
{
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    desc.handle.win32.handle = (void *)handle;
    cudaImportExternalSemaphore(&extSem, &desc);
    return extSem;
}

```

3.3.1.6. Signaling/waiting on imported synchronization objects An imported Vulkan semaphore object can be signaled as shown below. Signaling such a semaphore object sets it to the signaled state. The corresponding wait that waits on this signal must be issued in Vulkan. Additionally, the wait that waits on this signal must be issued after this signal has been issued.

```c++
void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t
stream) {
    cudaExternalSemaphoreSignalParams params = {};
    memset(&params, 0, sizeof(params));
    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}
```

​        An imported Vulkan semaphore object can be waited on as shown below. Waiting on such a semaphore object waits until it reaches the signaled state and then resets it back to the unsignaled state. The corresponding signal that this wait is waiting on must be issued in Vulkan. Additionally, the signal must be issued before this wait can be issued. 

```c++
void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams params = {};
    memset(&params, 0, sizeof(params));
    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}
```

​         Traditional OpenGL-CUDA interop as outlined in section 3.2.12.1 works by CUDA directly consuming handles created in OpenGL. However, since OpenGL can also consume memory and synchronization objects created in Vulkan, there exists an alternative approach to doing OpenGL-CUDA interop. Essentially, memory and synchronization objects exported by Vulkan could be imported into both, OpenGL and CUDA, and then used to coordinate memory accesses between OpenGL and CUDA. Please refer to the following OpenGL extensions for further details on how to import memory and synchronization objects exported by Vulkan: 

GL_EXT_memory_object
GL_EXT_memory_object_fd
GL_EXT_memory_object_win32
GL_EXT_semaphore
GL_EXT_semaphore_fd
GL_EXT_semaphore_win32  

#### 3.3.3. Direct3D 12 Interoperability 

##### 3.3.3.1. Matching device LUIDs 

​        When importing memory and synchronization objects exported by Direct3D 12, they must be imported and mapped on the same device as they were created on. The CUDA device that corresponds to the Direct3D 12 device on which the objects were created can be determined by comparing the LUID of a CUDA device with that of the Direct3D 12 device, as shown in the following code sample. Note that the Direct3D 12 device must not be created on a linked node adapter. I.e. the node count as returned by ID3D12Device::GetNodeCount must be 1. 

```c++
int getCudaDeviceForD3D12Device(ID3D12Device *d3d12Device) {
    LUID d3d12Luid = d3d12Device->GetAdapterLuid();
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);
    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        char *cudaLuid = deviceProp.luid;
        if (!memcmp(&d3d12Luid.LowPart, cudaLuid, sizeof(d3d12Luid.LowPart)) &&
       		!memcmp(&d3d12Luid.HighPart, cudaLuid + sizeof(d3d12Luid.LowPart),
        			sizeof(d3d12Luid.HighPart))) {
        	return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}
```



3.3.3.2. Importing memory objects

​        A shareable Direct3D 12 heap memory object, created by setting the flag D3D12_HEAP_FLAG_SHARED in the call to ID3D12Device::CreateHeap, can be imported into CUDA using the NT handle associated with that object as shown below. Note that it is the application’s responsibility to close the NT handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed. 

```c++
cudaExternalMemory_t importD3D12HeapFromNTHandle(HANDLE handle, unsigned long
long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    cudaImportExternalMemory(&extMem, &desc);
    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);
    return extMem;
}
```

​        A shareable Direct3D 12 heap memory object can also be imported using a named handle if one exists as shown below.  

```c++
cudaExternalMemory_t importD3D12HeapFromNamedNTHandle(LPCWSTR name, unsigned
long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    desc.handle.win32.name = (void *)name;
    desc.size = size;
    cudaImportExternalMemory(&extMem, &desc);
    return extMem;
}

```

A shareable Direct3D 12 committed resource, created by setting the flag D3D12_HEAP_FLAG_SHARED in the call to D3D12Device::CreateCommittedResource, can be imported into CUDA using the NT handle associated with that object as shown below. When importing a Direct3D 12 committed resource, the flag
cudaExternalMemoryDedicated must be set. Note that it is the application’s responsibility to close the NT handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed.  

```c++
cudaExternalMemory_t importD3D12CommittedResourceFromNTHandle(HANDLE handle,
unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;
    cudaImportExternalMemory(&extMem, &desc);
    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);
    return extMem;
}
```

A shareable Direct3D 12 committed resource can also be imported using a named handle if one exists as shown below.  

```c++
cudaExternalMemory_t importD3D12CommittedResourceFromNamedNTHandle(LPCWSTR name,
unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    desc.handle.win32.name = (void *)name;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;
    cudaImportExternalMemory(&extMem, &desc);
    return extMem;
}
```

3.3.3.3. Mapping buffers onto imported memory objects
        A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping must match that specified when creating the mapping using the corresponding Direct3D 12 API. All mapped device pointers must be freed using cudaFree().  

```c++
void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long
long offset, unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.size = size;
    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
    // Note: ‘ptr’ must eventually be freed using cudaFree()
    return ptr;
}
```

3.3.3.4. Mapping mipmapped arrays onto imported memory objects
A CUDA mipmapped array can be mapped onto an imported memory object asshown below. The offset, dimensions, format and number of mip levels must matchthat specified when creating the mapping using the corresponding Direct3D 12 API. Additionally, if the mipmapped array can be bound as a render target in Direct3D 12, the flag cudaArrayColorAttachment must be set. All mapped mipmapped arrays must be freed using cudaFreeMipmappedArray(). The following code sample shows how  to convert Vulkan parameters into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects  

```c++
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, 	
														 cudaChannelFormatDesc *formatDesc, cudaExtent *extent, 
														 unsigned int flags, unsigned int numLevels) {
	cudaMipmappedArray_t mipmap = NULL;
	cudaExternalMemoryMipmappedArrayDesc desc = {};
	memset(&desc, 0, sizeof(desc));
	desc.offset = offset;
	desc.formatDesc = *formatDesc;
	desc.extent = *extent;
	desc.flags = flags;
	desc.numLevels = numLevels;
	// Note: ‘mipmap’ must eventually be freed using cudaFreeMipmappedArray()
	cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
	return mipmap;
}
cudaChannelFormatDesc getCudaChannelFormatDescForDxgiFormat(DXGI_FORMAT dxgiFormat)
{
	cudaChannelFormatDesc d;
	memset(&d, 0, sizeof(d));
	switch (dxgiFormat) {
		case DXGI_FORMAT_R8_UINT: 
			d.x = 8; 
			d.y = 0; 
			d.z = 0; 
			d.w = 0;
			d.f = cudaChannelFormatKindUnsigned; 
			break;
		case DXGI_FORMAT_R8_SINT: 
			d.x = 8; 
			d.y = 0; 
			d.z = 0; 
			d.w = 0;
			d.f = cudaChannelFormatKindSigned; 
			break;
		case DXGI_FORMAT_R8G8_UINT: 
			d.x = 8; d.y = 8; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindUnsigned; 
			break;
		case DXGI_FORMAT_R8G8_SINT: 
			d.x = 8; d.y = 8; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindSigned; 
			break;
		case DXGI_FORMAT_R8G8B8A8_UINT: 
			d.x = 8; d.y = 8; d.z = 8; d.w = 8;
			d.f = cudaChannelFormatKindUnsigned; 
			break;
		case DXGI_FORMAT_R8G8B8A8_SINT: 
			d.x = 8; d.y = 8; d.z = 8; d.w = 8;
			d.f = cudaChannelFormatKindSigned; 
			break;
		case DXGI_FORMAT_R16_UINT: 
			d.x = 16; d.y = 0; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindUnsigned; 
			break;
		case DXGI_FORMAT_R16_SINT: 
			d.x = 16; d.y = 0; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindSigned; 
			break;
		case DXGI_FORMAT_R16G16_UINT: 
			d.x = 16; d.y = 16; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindUnsigned; 
			break;
		case DXGI_FORMAT_R16G16_SINT: 
			d.x = 16; d.y = 16; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindSigned; 
			break;
		case DXGI_FORMAT_R16G16B16A16_UINT: 
			d.x = 16; d.y = 16; d.z = 16; d.w = 16;
			d.f = cudaChannelFormatKindUnsigned; 
			break;
		case DXGI_FORMAT_R16G16B16A16_SINT: 
			d.x = 16; d.y = 16; d.z = 16; d.w = 16;
			d.f = cudaChannelFormatKindSigned; 
			break;
		case DXGI_FORMAT_R32_UINT: 
			d.x = 32; d.y = 0; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindUnsigned; 
			break;
		case DXGI_FORMAT_R32_SINT: 
			d.x = 32; d.y = 0; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindSigned;
			break;
		case DXGI_FORMAT_R32_FLOAT: 
			d.x = 32; d.y = 0; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindFloat; 
			break;
		case DXGI_FORMAT_R32G32_UINT:
			d.x = 32; d.y = 32; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindUnsigned; 
			break;
		case DXGI_FORMAT_R32G32_SINT:
			d.x = 32; d.y = 32; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindSigned;
			break;
		case DXGI_FORMAT_R32G32_FLOAT:
			d.x = 32; d.y = 32; d.z = 0; d.w = 0;
			d.f = cudaChannelFormatKindFloat;
			break;
		case DXGI_FORMAT_R32G32B32A32_UINT: 
			d.x = 32; d.y = 32; d.z = 32; d.w = 32;
			d.f = cudaChannelFormatKindUnsigned;
			break;
		case DXGI_FORMAT_R32G32B32A32_SINT: 
			d.x = 32; d.y = 32; d.z = 32; d.w = 32;
			d.f = cudaChannelFormatKindSigned;
			break;
		case DXGI_FORMAT_R32G32B32A32_FLOAT: 
			d.x = 32; d.y = 32; d.z = 32; d.w = 32;
			d.f = cudaChannelFormatKindFloat;
			break;
		default: 
			assert(0);
	}
	return d;
}

cudaExtent getCudaExtentForD3D12Extent(UINT64 width, UINT height, UINT16
									   depthOrArraySize, D3D12_SRV_DIMENSION d3d12SRVDimension) {
	cudaExtent e = { 0, 0, 0 };
	switch (d3d12SRVDimension) {
		case D3D12_SRV_DIMENSION_TEXTURE1D: 
		e.width = width; e.height = 0;
		e.depth = 0; break;
		case D3D12_SRV_DIMENSION_TEXTURE2D: 
		e.width = width; e.height =
		height; e.depth = 0; break;
		case D3D12_SRV_DIMENSION_TEXTURE3D: 
		e.width = width; e.height =
		height; e.depth = depthOrArraySize; break;
		case D3D12_SRV_DIMENSION_TEXTURECUBE: 
		e.width = width; e.height =
		height; e.depth = depthOrArraySize; break;
		case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: 
		e.width = width; e.height = 0;
		e.depth = depthOrArraySize; break;
		case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
		e.width = width; e.height =
		height; e.depth = depthOrArraySize; break;
		case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: 
		e.width = width; e.height =
		height; e.depth = depthOrArraySize; break;
		default: assert(0);
	}
	return e;
}
unsigned int getCudaMipmappedArrayFlagsForD3D12Resource(D3D12_SRV_DIMENSION d3d12SRVDimension, 
														D3D12_RESOURCE_FLAGS d3d12ResourceFlags, 
														bool allowSurfaceLoadStore) {
	unsigned int flags = 0;
	switch (d3d12SRVDimension) {
		case D3D12_SRV_DIMENSION_TEXTURECUBE: 
			flags |= cudaArrayCubemap;
			break;
		case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: 
			flags |= cudaArrayCubemap | cudaArrayLayered; 
			break;
		case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: 
			flags |= cudaArrayLayered;
			break;
		case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: 
			flags |= cudaArrayLayered;
			break;
		default: 
			break;
	}
	if (d3d12ResourceFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
		flags |= cudaArrayColorAttachment;
	}
	if (allowSurfaceLoadStore) {
		flags |= cudaArraySurfaceLoadStore;
	}
	return flags;
}
```

##### 3.3.3.5. Importing synchronization objects

A shareable Direct3D 12 fence object, created by setting the flag D3D12_FENCE_FLAG_SHARED in the call to ID3D12Device::CreateFence, can be imported into CUDA using the NT handle associated with that object as shown below. 

Note that it is the application’s responsibility to close the handle when it is not required  anymore. The NT handle holds a reference to the resource, so it must be explicitly freed
before the underlying semaphore can be freed.  

```c++
cudaExternalSemaphore_t importD3D12FenceFromNTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.handle = handle;
    cudaImportExternalSemaphore(&extSem, &desc);
    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);
    return extSem;
}
```

​        A shareable Direct3D 12 fence object can also be imported using a named handle if one exists as shown below.  

```c++
cudaExternalSemaphore_t importD3D12FenceFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.name = (void *)name;
    cudaImportExternalSemaphore(&extSem, &desc);
    return extSem;
}
```

##### 3.3.3.6. Signaling/waiting on imported synchronization objects

An imported Direct3D 12 fence object can be signaled as shown below. Signaling such
a fence object sets its value to the one specified. The corresponding wait that waits on
this signal must be issued in Direct3D 12. Additionally, the wait that waits on this signal
must be issued after this signal has been issued.  

```c++
void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long
value, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};
    memset(&params, 0, sizeof(params));
    params.params.fence.value = value;
    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}
```

An imported Direct3D 12 fence object can be waited on as shown below. Waiting on such
a fence object waits until its value becomes greater than or equal to the specified value.  

The corresponding signal that this wait is waiting on must be issued in Direct3D 12.
Additionally, the signal must be issued before this wait can be issued.  

```c++
void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long
value, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};
    memset(&params, 0, sizeof(params));
    params.params.fence.value = value;
    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}
```

#### 3.3.4. Direct3D 11 Interoperability  略

#### 3.3.5. NVIDIA Software Communication Interface Interoperability (NVSCI)  

NvSciBuf and NvSciSync are interfaces developed for serving the following purposes -

Allow applications to allocate and exchange buffers in memory - NvSciBuf 

Allow applications to manage synchronization objects at operation boundaries - NvSciSync

 More details on these interfaces are available at -

NvSciBuf - https://docs.nvidia.com/drive/active/5.1.6.0L/nvvib_docs/index.html#page/DRIVE_OS_Linux_SDK_Development_Guide%2FGraphics%2Fnvsci_nvscibuf.html%23 

NvSciSync - https://docs.nvidia.com/drive/active/5.1.6.0L/nvvib_docs/index.html#page/DRIVE_OS_Linux_SDK_Development_Guide%2FGraphics%2Fnvsci_nvscisync.html%23wwpID0E0PM0HA

 3.3.5.1. Importing memory objects 

​        For allocating an NvSciBuf object compatible with a given CUDA device, the corresponding GPU id must be set with NvSciBufGeneralAttrKey_GpuId in the NvSciBuf attribute list as shown below. For more details on how to allocate and maintain NvSciBuf objects refer to https://docs.nvidia.com/drive/active/5.1.6.0L/nvvib_docs/DRIVE_OS_Linux_SDK_Development_Guide/baggage/group__nvscibuf__obj__api.html#ga3a1be8a02e29ce4c92e2ed27fa9ea828. 

```c++
NvSciBufObj createNvSciBufObject() {
	// Raw Buffer Attributes for CUDA
	NvSciBufType bufType = NvSciBufType_RawBuffer;
	uint64_t rawsize = SIZE;
	uint64_t align = 0;
	bool cpuaccess_flag = true;
	NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
	uint64_t gpuId[] = {};
	cuDeviceGetUuid(&uuid, dev));
	gpuid[0] = uuid.bytes;
	// Fill in values
	NvSciBufAttrKeyValuePair rawbuffattrs[] = {
		{ NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
		{ NvSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
		{ NvSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
		{ NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
		sizeof(cpuaccess_flag) },
		{ NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
		{ NvSciBufGeneralAttrKey_GpuId, &gpuid, sizeof(gpuId) },
	};
	// Create list by setting attributes
	err = NvSciBufAttrListSetAttrs(attrListBuffer, rawbuffattrs,
								   sizeof(rawbuffattrs)/sizeof(NvSciBufAttrKeyValuePair));
	NvSciBufAttrListCreate(NvSciBufModule, &attrListBuffer);
	// Reconcile And Allocate
	NvSciBufAttrListReconcile(&attrListBuffer, 1, &attrListReconciledBuffer,
	&attrListConflictBuffer)
	NvSciBufObjAlloc(attrListReconciledBuffer, &bufferObjRaw);
	return bufferObjRaw;
}
```

​          The allocated NvSciBuf memory object can be imported in CUDA using the NvSciBufObj handle as shown below. Application should query the allocated NvSciBufObj for attributes required for filling CUDA External Memory Descriptor. Note that the attribute list and NvSciBuf objects should be maintained by the application. If the NvSciBuf object imported into CUDA is also mapped by other drivers, then the application must use NvSciSync objects (Refer 3.2.13.5.4 Importing synchronization objects) as appropriate barriers to maintain coherence between CUDA and the other drivers. 

```c++
cudaExternalMemory_t importNvSciBufObject (NvSciBufObj bufferObjRaw) {
	/*************** Query NvSciBuf Object **************/
	NvSciBufAttrKeyValuePair bufattrs[] = {
		{NvSciBufRawBufferAttrKey_Size, NULL, 0},
	};
	NvSciBufAttrListGetAttrs(retList, bufattrs, sizeof(bufattrs)/sizeof(NvSciBufAttrKeyValuePair)));
	ret_size = *(static_cast<const uint64_t*>(bufattrs[0].value));
	/*************** NvSciBuf Registration With CUDA **************/
	// Fill up CUDA_EXTERNAL_MEMORY_HANDLE_DESC
	cudaExternalMemoryHandleDesc memHandleDesc;
	memset(&memHandleDesc, 0, sizeof(memHandleDesc));
	memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
	memHandleDesc.handle.nvSciBufObject = bufferObjRaw;
	memHandleDesc.size = ret_size;
	cudaImportExternalMemory(&extMemBuffer, &memHandleDesc);
	return extMemBuffer;
}
```

##### 3.3.5.2. Mapping buffers onto imported memory objects 

​        A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping can be filled as per the attributes of the allocated NvSciBufObj. All mapped device pointers must be freed using cudaFree().  

```c++
void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long
long offset, unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.size = size;
    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
    // Note: ‘ptr’ must eventually be freed using cudaFree()
    return ptr;
}
```

##### 3.3.5.3. Mapping mipmapped arrays onto imported memory objects 

​        A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions and format can be filled as per the attributes of the allocated NvSciBufObj. The number of mip levels must be 1. All mapped mipmapped arrays must be freed using cudaFreeMipmappedArray(). The following code sample shows how to convert NvSciBuf attributes into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects. 

```c++
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem,                                                   unsigned long long offset,
                                              cudaChannelFormatDesc *formatDesc,                                                       cudaExtent *extent, 
                                              unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;
    // Note: ‘mipmap’ must eventually be freed using cudaFreeMipmappedArray()
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
    return mipmap;
}
```

##### 3.3.5.4. Importing synchronization objects

​        NvSciSync attributes that are compatible with a given CUDA device can be generated
using cudaDeviceGetNvSciSyncAttributes(). The returned attribute list can be used to
create a NvSciSyncObj that is guaranteed compatibility with a given CUDA device.  

```c++
NvSciSyncObj createNvSciSyncObject() {
    NvSciSyncObj nvSciSyncObj
    int cudaDev0 = 0;
    int cudaDev1 = 1;
    NvSciSyncAttrList signalerAttrList = NULL;
    NvSciSyncAttrList waiterAttrList = NULL;
    NvSciSyncAttrList reconciledList = NULL;
    NvSciSyncAttrList newConflictList = NULL;
    NvSciSyncAttrListCreate(module, &signalerAttrList);
    NvSciSyncAttrListCreate(module, &waiterAttrList);
    NvSciSyncAttrList unreconciledList[2] = {NULL, NULL};
    unreconciledList[0] = signalerAttrList;
    unreconciledList[1] = waiterAttrList;
    cudaDeviceGetNvSciSyncAttributes(signalerAttrList, cudaDev0,
    								 CUDA_NVSCISYNC_ATTR_SIGNAL);
    cudaDeviceGetNvSciSyncAttributes(waiterAttrList, cudaDev1,
    								 CUDA_NVSCISYNC_ATTR_WAIT);
    NvSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList,
    						   &newConflictList);
    NvSciSyncObjAlloc(reconciledList, &nvSciSyncObj);
    return nvSciSyncObj;
}
```

An NvSciSync object (created as above) can be imported into CUDA using the NvSciSyncObj handle as shown below. Note that ownership of the NvSciSyncObj handle continues to lie with the application even after it is imported.  

```c++
cudaExternalSemaphore_t importNvSciSyncObject(void* nvSciSyncObj) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    desc.handle.nvSciSyncObj = nvSciSyncObj;
    cudaImportExternalSemaphore(&extSem, &desc);
    // Deleting/Freeing the nvSciSyncObj beyond this point will lead to
    undefined behavior in CUDA
    return extSem;
}
```

3.3.5.5. Signaling/waiting on imported synchronization objects
An imported NvSciSyncObj object can be signaled as outlined below. Signaling NvSciSync backed semaphore object initializes the fence parameter passed as input. This fence parameter is waited upon by a wait operation that corresponds to the aforementioned signal. Additionally, the wait that waits on this signal must be issued after this signal has been issued. If the flags are set to cudaExternalSemaphoreSignalSkipNvSciBufMemSync then memory synchronization operations (over all the imported NvSciBuf in this process) that are executed as a part of the signal operation by default are skipped.  

```c++
void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t
stream, void *fence) {
    cudaExternalSemaphoreSignalParams signalParams = {};
    memset(&signalParams, 0, sizeof(signalParams));
    signalParams.params.nvSciSync.fence = (void*)fence;
    signalParams.flags = 0; //OR cudaExternalSemaphoreSignalSkipNvSciBufMemSync
    cudaSignalExternalSemaphoresAsync(&extSem, &signalParams, 1, stream);
}
```

​        An imported NvSciSyncObj object can be waited upon as outlined below. Waiting on NvSciSync backed semaphore object waits until the input fence parameter is signaled by the corresponding signaler. Additionally, the signal mustbe issued before the wait can be issued. If the flags are set to cudaExternalSemaphoreWaitSkipNvSciBufMemSync then memory synchronization operations (over all the imported NvSciBuf in this process) that are executed as a part of the wait operation by default are skipped. 

```c++
void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream,
void *fence) {
    cudaExternalSemaphoreWaitParams waitParams = {};
    memset(&waitParams, 0, sizeof(waitParams));
    waitParams.params.nvSciSync.fence = (void*)fence;
    waitParams.flags = 0; //OR cudaExternalSemaphoreWaitSkipNvSciBufMemSync
    cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, stream);
}
```

3.4. Versioning and Compatibility 

​        There are two version numbers that developers should care about when developing a CUDA application: The compute capability that describes the general specifications and features of the compute device (see Compute Capability) and the version of the CUDA driver API that describes the features supported by the driver API and runtime. 

​        The version of the driver API is defined in the driver header file as CUDA_VERSION. It allows developers to check whether their application requires a newer device driver than the one currently installed. This is important, because the driver API is backward compatible, meaning that applications, plug-ins, and libraries (including the CUDA runtime) compiled against a particular version of the driver API will continue to work on subsequent device driver releases as illustrated in Figure 13. The driver API is not forward compatible, which means that applications, plug-ins, and libraries (including the CUDA runtime) compiled against a particular version of the driver API will not work on previous versions of the device driver. 

​        It is important to note that there are limitations on the mixing and matching of versions that is supported: 

1、Since only one version of the CUDA Driver can be installed at a time on a system, the installed driver must be of the same or higher version than the maximum Driver API version against which any application, plug-ins, or libraries that must run on that system were built. 

2、All plug-ins and libraries used by an application must use the same version of the CUDA Runtime unless they statically link to the Runtime, in which case multiple versions of the runtime can coexist in the same process space. Note that if nvcc is used to link the application, the static version of the CUDA Runtime library will be used by default, and all CUDA Toolkit libraries are statically linked against the CUDA Runtime. 

3、All plug-ins and libraries used by an application must use the same version of any libraries that use the runtime (such as cuFFT, cuBLAS, ...) unless statically linking to those libraries. 

​        For Tesla GPU products, CUDA 10 introduced a new forward-compatible upgrade path for the user-mode components of the CUDA Driver. This feature is described in CUDA Compatibility. The requirements on the CUDA Driver version described here apply to the version of the user-mode components. 

3.5. Compute Modes

​        On Tesla solutions running Windows Server 2008 and later or Linux, one can set any device in a system in one of the three following modes using NVIDIA's System Management Interface (nvidia-smi), which is a tool distributed as part of the driver: 

1、Default compute mode: Multiple host threads can use the device (by calling cudaSetDevice() on this device, when using the runtime API, or by making current a context associated to the device, when using the driver API) at the same time. 

2、Exclusive-process compute mode: Only one CUDA context may be created on the device across all processes in the system. The context may be current to as many threads as desired within the process that created that context. 

3、Prohibited compute mode: No CUDA context can be created on the device. 

​        This means, in particular, that a host thread using the runtime API without explicitly calling cudaSetDevice() might be associated with a device other than device 0 if device 0 turns out to be in prohibited mode or in exclusive-process mode and used by another process. cudaSetValidDevices() can be used to set a device from a prioritized list of devices. 

​        Note also that, for devices featuring the Pascal architecture onwards (compute capability with major revision number 6 and higher), there exists support for Compute Preemption. This allows compute tasks to be preempted at instructionlevel granularity, rather than thread block granularity as in prior Maxwell and Kepler GPU architecture, with the benefit that applications with long-running kernels can be prevented from either monopolizing the system or timing out. However, there will be context switch overheads associated with Compute Preemption, which is automatically enabled on those devices for which support exists. The individual attribute query function cudaDeviceGetAttribute() with the attribute cudaDevAttrComputePreemptionSupported can be used to determine if the device in use supports Compute Preemption. Users wishing to avoid context switch overheads associated with different processes can ensure that only one process is active on the GPU by selecting exclusive-process mode.

​         Applications may query the compute mode of a device by checking the computeMode device property (see Device Enumeration).

 3.6. Mode Switches 

​        GPUs that have a display output dedicate some DRAM memory to the so-called primary surface, which is used to refresh the display device whose output is viewed by the user. When users initiate a mode switch of the display by changing the resolution or bit depth of the display (using NVIDIA control panel or the Display control panel on Windows), the amount of memory needed for the primary surface changes. For example, if the user changes the display resolution from 1280x1024x32-bit to 1600x1200x32-bit, the system must dedicate 7.68 MB to the primary surface rather than 5.24 MB. (Full-screen graphics applications running with anti-aliasing enabled may require much more display memory for the primary surface.) On Windows, other events that may initiate display mode switches include launching a full-screen DirectX application, hitting Alt +Tab to task switch away from a full-screen DirectX application, or hitting Ctrl+Alt+Del to lock the computer.

​         If a mode switch increases the amount of memory needed for the primary surface, the system may have to cannibalize memory allocations dedicated to CUDA applications. Therefore, a mode switch results in any call to the CUDA runtime to fail and return an invalid context error.

​        3.7. Tesla Compute Cluster Mode for Windows 

​        Using NVIDIA's System Management Interface (nvidia-smi), the Windows device driver can be put in TCC (Tesla Compute Cluster) mode for devices of the Tesla and Quadro Series of compute capability 2.0 and higher. 

​        This mode has the following primary benefits: 

1、It makes it possible to use these GPUs in cluster nodes with non-NVIDIA integrated graphics; 

2、It makes these GPUs available via Remote Desktop, both directly and via cluster management systems that rely on Remote Desktop; 

3、It makes these GPUs available to applications running as a Windows service (i.e., in Session 0). 

However, the TCC mode removes support for any graphics functionality 































