---
layout: post
title:  "CUDA-samples学习三"
date:   2020-09-12 15:46:00 +0800
categories: CUDA
---

# cudaNvSci

Software Communication Interfaces(Sci)

涉及显存通信及交互的机制，暂时不是很清楚

# cudaOpenMP

例子用于展示利用OpenMP在多个GPU上进行工作，OpenMP通过简单的宏指令实现多线程并发，这样在每个线程在不通的GPU上启动核函数，实现多GPU的并发操作。

核心逻辑

```c++
    //omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA device
    omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there are CUDA devices
    #pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        checkCudaErrors(cudaSetDevice(cpu_thread_id % num_gpus));   // "% num_gpus" allows more CPU threads than GPU devices
        checkCudaErrors(cudaGetDevice(&gpu_id));
        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);

        int *d_a = 0;   // pointer to memory on the device associated with this CPU thread
        int *sub_a = a + cpu_thread_id * n / num_cpu_threads;   // pointer to this CPU thread's portion of data
        unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
        dim3 gpu_threads(128);  // 128 threads per block
        dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));

        checkCudaErrors(cudaMalloc((void **)&d_a, nbytes_per_kernel));
        checkCudaErrors(cudaMemset(d_a, 0, nbytes_per_kernel));
        checkCudaErrors(cudaMemcpy(d_a, sub_a, nbytes_per_kernel, cudaMemcpyHostToDevice));
        kernelAddConstant<<<gpu_blocks, gpu_threads>>>(d_a, b);

        checkCudaErrors(cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_a));

    }
```

例子演示了OpenMP的使用方式及API

```c++
#include <omp.h>

int num_cpus = omp_get_num_procs();

omp_set_num_threads(num_gpus)
#pragma omp parallel
{
    unsigned int cpu_thread_id = omp_get_thread_num();
    unsigned int num_cpu_threads = omp_get_num_threads();
}
```

# cudaTensorCoreGemm

需要cuda9.0及sm7.0以上支持

# fp16ScalarProduct

半精度使用的例子，fp16的数据类型是half。

cuda特有数据类型half2，其实这是矢量类型，包含两个half类型元素的vector，cuda中有很多类似的自定义类型，int2，int3，float2，float3等

例子中展示了一种sizeof不常见的用法`size*sizeof*vec[i]`，其实这个和`size*sizeof(*vec[i])`同义。之所以可以像前面那么用，sizeof是一个关键字，并不是函数，所以可以像这么使用`sizeof int`，例子中前面有*可以区分就连在了一起。

核函数

```c++ 
__global__ void scalarProductKernel_intrinsics(
        half2 const * const a,
        half2 const * const b,
        float * const results,
        size_t const size
        )
{
    const int stride = gridDim.x*blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    shArray[threadIdx.x] = __float2half2_rn(0.f);
    half2 value = __float2half2_rn(0.f);

    for (int i = threadIdx.x + blockDim.x + blockIdx.x; i < size; i+=stride)
    {
        value = __hfma2(a[i], b[i], value);
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_intrinsics(shArray);

    if (threadIdx.x == 0)
    {
        half2 result = shArray[0];
        float f_result = __low2float(result) + __high2float(result);
        results[blockIdx.x] = f_result;
    }
}

__global__ void scalarProductKernel_native(
        half2 const * const a,
        half2 const * const b,
        float * const results,
        size_t const size
        )
{
    const int stride = gridDim.x*blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    half2 value(0.f, 0.f);
    shArray[threadIdx.x] = value;

    for (int i = threadIdx.x + blockDim.x + blockIdx.x; i < size; i+=stride)
    {
        value = a[i] * b[i] + value;
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_native(shArray);

    if (threadIdx.x == 0)
    {
        half2 result = shArray[0];
        float f_result = (float)result.y + (float)result.x;
        results[blockIdx.x] = f_result;
    }
}
```

输入是两个元素half2类型的向量，可以看成由二维平面点组成的序列，一个二维平面点就是一个half2。计算逻辑是将上下两组点对应位置做内积然后求和。

使用了两种计算方式，一个是原生计算，另一个是采用half2的指令进行计算。感觉是half对操作符进行了重载，直接用原生操作符就可以计算。





