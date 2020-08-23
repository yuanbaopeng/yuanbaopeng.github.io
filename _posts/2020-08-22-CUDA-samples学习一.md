---
layout: post
title:  "CUDA-samples学习一"
date:   2020-08-22 21:41:00 +0800
categories: CUDA
---

# asyncAPI

用于说明cuda的异步api的使用，即在cpu上调用cuda异步api是立即就返回的，此时可以在cpu继续干其他的事情。

sample中是通过查询cuda event是否发生确定gpu上的任务是否已完成。

启动核心逻辑

```c++
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }
```

核函数比较简单，对数组的每个元素增加固定的value

```c++
__global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}
```

# cdpSimplePrint

cdp是CUDA dynamic parallelism的缩写，CUDA的动态并行，其实是说CUDA在3.5后支持在核函数中再次调用核函数的能力，称为dynamic parallelism。

sample中通过在核函数中打印当前的thread id和block id来说明。

核函数进行了递归调用，其中用到了`__syncthreads()`，这一个线程同步机制，会让同一个block内的所有线程都到达此处后再接着运行

```c++
__global__ void cdp_kernel(int max_depth, int depth, int thread, int parent_uid)
{
    // We create a unique ID per block. Thread 0 does that and shares the value with the other threads.
    __shared__ int s_uid;

    if (threadIdx.x == 0)
    {
        s_uid = atomicAdd(&g_uids, 1);
    }

    __syncthreads();

    // We print the ID of the block and information about its parent.
    print_info(depth, thread, s_uid, parent_uid);

    // We launch new blocks if we haven't reached the max_depth yet.
    if (++depth >= max_depth)
    {
        return;
    }

    cdp_kernel<<<gridDim.x, blockDim.x>>>(max_depth, depth, threadIdx.x, s_uid);
}
```

# cdpSimpleQuicksort

这个sample同上一个sample是类似的，用cdp的能力实现了快排，我这里将`check_results`函数里的资源申请提取到了`main`函数里，资源申请和释放放在同一个函数里，方便管理。结果发现这个sample使用了两种资源申请方式，也是很特别了。

```c++
    h_data =(unsigned int *)malloc(num_items*sizeof(unsigned int));
    unsigned int *results_h = new unsigned[num_items];
    ...
    delete[] results_h;
    free(h_data);
```

核函数是一个普通的快排实现，和cpu版的区别不大，直接使用指针较多，另外由于cuda嵌套调用深度的限制和少量数据排序用嵌套不合适的原因，采用了选择排序进行实现

```c++
__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right, int depth)
{
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)
    {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *lptr = data+left;
    unsigned int *rptr = data+right;
    unsigned int  pivot = data[(left+right)/2];

    // Do the partitioning.
    while (lptr <= rptr)
    {
        // Find the next left- and right-hand values to swap
        unsigned int lval = *lptr;
        unsigned int rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot)
        {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot)
        {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < nright)
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if (nleft < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}
```

注意到这里在核函数内部调用`cdp_simple_quicksort`时，使用了独立的cuda stream，stream是调用核函数`<<<...>>>`中的第四个参数



