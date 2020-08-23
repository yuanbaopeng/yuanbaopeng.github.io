---
layout: post
title:  "CUDA-samples学习二"
date:   2020-08-23 08:38:00 +0800
categories: CUDA
---

# clock

用于演示在核函数中使用`clock()`进行性能测量的方法。

核函数调用实例动态共享内存申请的方式，在`<<<...>>>`中的第三个参数是申请共享内存的大小。

```c++
timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 *NUM_THREADS>>>(dinput, doutput, dtimer);
```

在核函数使用了该共享内存，共享内存是block内共享，即同一个block内的线程是共享一个内存，不同block之前互相看不到对方的共享内存。

共享内存分为静态申请和动态申请两种方式：

静态申请是在核函数内使用`__shared__`关键字直接进行数组定义，并指定数组大小`__shared__ float shared[256]`

动态申请是定义核函数时不知道内存的大小，使用`extern __shared__`进行申明，在调用核函数时，通过`<<<...>>>`的第三个参数进行大小指定

```c++
__global__ static void timedReduction(const float *input, float *output, clock_t *timer)
{
    // __shared__ float shared[2 * blockDim.x];
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock();

    // Copy input.
    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    // Perform reduction to find minimum.
    for (int d = blockDim.x; d > 0; d /= 2)
    {
        __syncthreads();

        if (tid < d)
        {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0)
            {
                shared[tid] = f1;
            }
        }
    }

    // Write result.
    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid+gridDim.x] = clock();
}
```

# clock_nvrtc

这个例子和上一个例子展示的功能是一样，不过使用了另外一种编译运行核函数的方式RTC（Runtime Compilation）运行时编译，可以在程序运行后加载cuda文件，进行编译和运行的技术。

正常编译cuda文件是需要使用nvcc进行编译，动态编译只需使用g++编译cpp文件，cuda文件在程序运行时被编译和运行。

核心主逻辑如下

```c++
    kernel_file = sdkFindFilePath("clock_kernel.cu", argv[0]);
    compileFileToPTX(kernel_file, argc, argv, &ptx, &ptxSize, 0);

    CUmodule module = loadPTX(ptx, argc, argv);
    CUfunction kernel_addr;

    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "timedReduction"));

    dim3 cudaBlockSize(NUM_THREADS,1,1);
    dim3 cudaGridSize(NUM_BLOCKS, 1, 1);

    CUdeviceptr dinput, doutput, dtimer;
    checkCudaErrors(cuMemAlloc(&dinput, sizeof(float) * NUM_THREADS * 2));
    checkCudaErrors(cuMemAlloc(&doutput, sizeof(float) * NUM_BLOCKS));
    checkCudaErrors(cuMemAlloc(&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
    checkCudaErrors(cuMemcpyHtoD(dinput, input, sizeof(float) * NUM_THREADS * 2));

    void *arr[] = { (void *)&dinput, (void *)&doutput, (void *)&dtimer };

    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                            cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
                                            cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, /* block dim */
                                             sizeof(float) * 2 *NUM_THREADS,0, /* shared mem, stream */
                                            &arr[0], /* arguments */
                                            0));

    checkCudaErrors(cuCtxSynchronize());
    checkCudaErrors(cuMemcpyDtoH(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
    checkCudaErrors(cuMemFree(dinput));
    checkCudaErrors(cuMemFree(doutput));
    checkCudaErrors(cuMemFree(dtimer));
```

核函数申明方式有所不同，开头增加了`extern "C"`，其余和上一个例子一样，去掉了`static`。

# cppIntegration

展示cuda文件如何与已有cpp文件集成，只需在cuda文件中定义c形式接口，就可以在cpp文件中调用

例子中使用了CUDA vector types，一种C++扩展，诸如int1，int2，int3，int4，float1，float2，float3，float4等，实际上是结构体，最多有四个成员x, y, z, w，并且保证了字节对齐。

参考资料：[vector-types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types)


# cppOverload

演示了核函数的重载实现，三个同名的核函数，签名不一样。