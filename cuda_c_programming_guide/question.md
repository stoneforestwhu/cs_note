question:

1、Page-Locked Host Memory introduces page-locked host memory that is required to overlap(这里的overlap到底是什么意思) kernel execution with data transfers between host and device memory. 见chapter3.2引言

2、什么是 single unified address space, 尤其是那个table1中的 40bit 47bit都是啥意思？ 见 chapter3.2.2

3、writecombining memory is not snooped during transfers across the PCI Express bus, which can improve transfer performance by up to 40% (snooped 是啥意思？)

​       另外一个， write-combining memory 释放了L1和L2的资源，也就意味着读操作的时候必须去memory中读，这个开销很大，但是写操作没影响，是否意味着写操作不通过L1和L24、

4、intra-device copy是设备内拷贝数据的意思吗？ 见3.2.5.3 Overlap of Data Transfer and Kernel Execution