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