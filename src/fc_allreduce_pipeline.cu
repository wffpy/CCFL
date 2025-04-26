#include "fc_allreduce_pipeline.cuh"
#include <cublas_v2.h>
#include <cstdio>

// 用cublas模拟tensor core fc kernel
void launch_fc_kernel(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    // C = A x B
    // A: MxK, B: KxN, C: MxN
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                B, N, A, K, &beta, C, N);
    cublasDestroy(handle);
}

// NCCL all-reduce
void launch_allreduce(float* data, int count, ncclComm_t comm, cudaStream_t stream) {
    ncclAllReduce((const void*)data, (void*)data, count, ncclFloat, ncclSum, comm, stream);
}

// 方案一：host端分块pipeline
void launch_fc_allreduce_pipeline(float* A, float* B, float* C, int M, int N, int K, float* comm_buf, int comm_count, ncclComm_t comm, cudaStream_t compute_stream, cudaStream_t comm_stream) {
    int block_rows = 256; // 分块行数
    int num_blocks = (M + block_rows - 1) / block_rows;
    for (int blk = 0; blk < num_blocks; ++blk) {
        int row_start = blk * block_rows;
        int row_end = (row_start + block_rows > M) ? M : (row_start + block_rows);
        int rows = row_end - row_start;
        float* A_blk = A + row_start * K;
        float* C_blk = C + row_start * N;
        float* comm_blk = comm_buf + row_start * N;
        // 1. 计算本块
        launch_fc_kernel(A_blk, B, C_blk, rows, N, K, compute_stream);
        // 2. 拷贝到通信buffer
        cudaMemcpyAsync(comm_blk, C_blk, rows * N * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream);
        // 3. 事件同步
        cudaEvent_t ready;
        cudaEventCreate(&ready);
        cudaEventRecord(ready, compute_stream);
        cudaStreamWaitEvent(comm_stream, ready, 0);
        // 4. all-reduce
        launch_allreduce(comm_blk, rows * N, comm, comm_stream);
        cudaEventDestroy(ready);
    }
}

// 方案二：kernel内部pipeline（实验性，P2P通信模拟）
__global__ void fc_allreduce_pipeline_kernel_impl(float* A, float* B, float* C, int M, int N, int K, float* comm_buf, int block_rows, int num_blocks, int peer_device) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int blk = 0; blk < num_blocks; ++blk) {
        int row_start = blk * block_rows;
        int row_end = (row_start + block_rows > M) ? M : (row_start + block_rows);
        int rows = row_end - row_start;
        // 1. 计算本块（仅主线程演示）
        if (tid == 0) {
            // 简化：只做memcpy模拟GEMM
            for (int i = 0; i < rows * N; ++i) {
                C[row_start * N + i] = A[row_start * K + (i % K)] + B[(i % N) * K];
            }
        }
        __syncthreads();
        // 2. 通信（P2P模拟）
        if (tid == 0 && peer_device >= 0) {
            // TODO(wangfangfei): 替换为真实的P2P通信
            // cudaMemcpyPeerAsync(comm_buf + row_start * N, peer_device, C + row_start * N, blockIdx.x, rows * N * sizeof(float));
        }
        __syncthreads();
    }
}

void launch_fc_allreduce_pipeline_kernel(float* A, float* B, float* C, int M, int N, int K, float* comm_buf, int comm_count, int peer_device, cudaStream_t stream) {
    int block_rows = 256;
    int num_blocks = (M + block_rows - 1) / block_rows;
    fc_allreduce_pipeline_kernel_impl<<<1, 32, 0, stream>>>(A, B, C, M, N, K, comm_buf, block_rows, num_blocks, peer_device);
}

// 通用接口
void launch_fc_allreduce_pipeline_v2(float* A, float* B, float* C, int M, int N, int K, float* comm_buf, int comm_count, ncclComm_t comm, int peer_device, cudaStream_t compute_stream, cudaStream_t comm_stream, PipelineMode mode) {
    if (mode == HostPipeline) {
        launch_fc_allreduce_pipeline(A, B, C, M, N, K, comm_buf, comm_count, comm, compute_stream, comm_stream);
    } else if (mode == KernelPipeline) {
        launch_fc_allreduce_pipeline_kernel(A, B, C, M, N, K, comm_buf, comm_count, peer_device, compute_stream);
    }
} 