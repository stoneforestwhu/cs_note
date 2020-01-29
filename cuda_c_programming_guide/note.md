note:

分配设备存储器：

cudaMalloc()

cudaFree()

cudaMallocPitch(void **devPtr, size_t *pitch, size_t widthByte, size_t height)

cudaMalloc3D(cudaPitchedPtr* devPitchedPtr, cudaExtent extent)  // 注意这两个参数

分配页锁定主机存储器：

cudaHostAlloc()         

cudaFreeHost()  

创建stream的API:

cudaStreamCreate()

cudaStreamCreateWithFlags()

cudaStreamCreateWithPriority()

cudaStreamNonBlocking是一个flag,并不是函数

销毁stream的API：

cudaStreamDestroy()

创建graph的API:



创建event的API:

cudaEventCreate()

cudaEventCreate()

销毁event的API:

cudaEventDestroy()

cudaEventDestroy()

不论是stream还是graph的概念，感觉都是相当高阶的，caffe里都没用到过，这些玩意儿有啥用，cudnn中会有？？？