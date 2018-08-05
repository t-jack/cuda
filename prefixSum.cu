#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1048576
#define THREAD_NUM 1024

typedef int ARRAY_TYPE;


__global__ void prefixSum_gpu_1(ARRAY_TYPE *a, ARRAY_TYPE *t)
{
    int i;
    ARRAY_TYPE temp;
    __shared__ ARRAY_TYPE a_shared[THREAD_NUM];

    a_shared[threadIdx.x] = a[threadIdx.x + blockIdx.x * THREAD_NUM];
    __syncthreads();

    for(i=1;i<1024;i*=2){
        if(threadIdx.x >= i){
            temp = a_shared[threadIdx.x - i];
        }
        __syncthreads();
        if(threadIdx.x >= i){
            a_shared[threadIdx.x] += temp;
        }
        __syncthreads();
    }

    if(threadIdx.x == THREAD_NUM-1){
        t[blockIdx.x] = a_shared[threadIdx.x];
        // printf("%d\n", t[blockIdx.x]);
    }

}


__global__ void prefixSum_gpu_2(ARRAY_TYPE *t, ARRAY_TYPE *p)
{

    int i;
    ARRAY_TYPE temp;
    __shared__ ARRAY_TYPE t_shared[THREAD_NUM];

    t_shared[threadIdx.x] = t[threadIdx.x];
    __syncthreads();

    for(i=1;i<1024;i*=2){
        if(threadIdx.x >= i){
            temp = t_shared[threadIdx.x - i];
        }
        __syncthreads();
        if(threadIdx.x >= i){
            t_shared[threadIdx.x] += temp;
        }
        __syncthreads();
    }

    p[threadIdx.x] = t_shared[threadIdx.x];
}

__global__ void prefixSum_gpu_3(ARRAY_TYPE *p, ARRAY_TYPE *a, ARRAY_TYPE *b)
{
    int i;
    ARRAY_TYPE psum, temp;
    __shared__ ARRAY_TYPE a_shared[THREAD_NUM];

    if(blockIdx.x > 0){
        psum = p[blockIdx.x-1];
    }else{
        psum = 0;
    }

    a_shared[threadIdx.x] = a[threadIdx.x + blockIdx.x * THREAD_NUM];
    __syncthreads();

    for(i=1;i<1024;i*=2){
        if(threadIdx.x >= i){
            temp = a_shared[threadIdx.x - i];
        }
        __syncthreads();
        if(threadIdx.x >= i){
            a_shared[threadIdx.x] += temp;
        }
        __syncthreads();
    }

    b[threadIdx.x + blockIdx.x * THREAD_NUM] = a_shared[threadIdx.x] + psum;

}


__host__ void prefixSum_cpu(ARRAY_TYPE *a, ARRAY_TYPE *b)
{
    int i;
    b[0] = a[0];
    for(i=1;i<N;i++){
        b[i] = b[i-1] + a[i];
    }
}



int main(int argc, char *argv[])
{
    int i;
    struct timeval s, e;
    cudaEvent_t dev_start, dev_stop;
    float kernel1, kernel2, kernel3, h2dTime, d2hTime;

    cudaEventCreate(&dev_start);
    cudaEventCreate(&dev_stop);
    
    ARRAY_TYPE *a_host, *b_host;
    ARRAY_TYPE *a_dev, *b_dev, *t_dev, *p_dev;
    ARRAY_TYPE check;

    a_host = (ARRAY_TYPE*)malloc(sizeof(ARRAY_TYPE)*N);
    b_host = (ARRAY_TYPE*)malloc(sizeof(ARRAY_TYPE)*N);

    for(i=0;i<N;i++){
        a_host[i] = rand()%10;
    }

    gettimeofday(&s, NULL);
    prefixSum_cpu(a_host, b_host);
    gettimeofday(&e, NULL);
    printf("time = %lf [ms]\n", (e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec)*1.0E-6 * 1000);

    printf("%d\n", b_host[N-1]);

    cudaMalloc(&a_dev, sizeof(ARRAY_TYPE)*N);
    cudaMalloc(&b_dev, sizeof(ARRAY_TYPE)*N);
    cudaMalloc(&t_dev, sizeof(ARRAY_TYPE)*(N/THREAD_NUM));
    cudaMalloc(&p_dev, sizeof(ARRAY_TYPE)*THREAD_NUM);

    cudaEventRecord(dev_start, 0);
    cudaMemcpy(a_dev, a_host, sizeof(ARRAY_TYPE)*N, cudaMemcpyHostToDevice);
    cudaEventRecord(dev_stop, 0);
    cudaEventSynchronize(dev_stop);
    cudaEventElapsedTime(&h2dTime, dev_start, dev_stop);

    cudaEventRecord(dev_start, 0);
    prefixSum_gpu_1<<<N/THREAD_NUM, THREAD_NUM>>>(a_dev, t_dev);
    cudaEventRecord(dev_stop, 0);
    cudaEventSynchronize(dev_stop);
    cudaEventElapsedTime(&kernel1, dev_start, dev_stop);

    cudaEventRecord(dev_start, 0);
    prefixSum_gpu_2<<<1, THREAD_NUM>>>(t_dev, p_dev);
    cudaEventRecord(dev_stop, 0);
    cudaEventSynchronize(dev_stop);
    cudaEventElapsedTime(&kernel2, dev_start, dev_stop);

    cudaEventRecord(dev_start, 0);
    prefixSum_gpu_3<<<N/THREAD_NUM, THREAD_NUM>>>(p_dev, a_dev, b_dev);
    cudaEventRecord(dev_stop, 0);
    cudaEventSynchronize(dev_stop);
    cudaEventElapsedTime(&kernel3, dev_start, dev_stop);

    cudaEventRecord(dev_start, 0);
    cudaMemcpy(b_host, b_dev, sizeof(ARRAY_TYPE)*N, cudaMemcpyDeviceToHost);
    cudaEventRecord(dev_stop, 0);
    cudaEventSynchronize(dev_stop);
    cudaEventElapsedTime(&d2hTime, dev_start, dev_stop);

    printf("%d\n", b_host[N-1]);

    check = 0;
    for(i=0;i<N;i++){
        check += a_host[i];
        if(check != b_host[i]){
            printf("error at %d\n", i);
            break;
        }
    }

    printf("HostToDevice : %f [ms]\n", h2dTime);
    printf("kernel 1     : %f [ms]\n", kernel1);
    printf("kernel 2     : %f [ms]\n", kernel2);
    printf("kernel 3     : %f [ms]\n", kernel3);
    printf("deviceToHost : %f [ms]\n", d2hTime);
    printf("gpuTotal     : %f [ms]\n", h2dTime + kernel1 + kernel2 + kernel3 + d2hTime);

    free(a_host);
    free(b_host);

    return 0;
}
