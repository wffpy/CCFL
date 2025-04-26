#include "fc_allreduce_pipeline.cuh"
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(-1); } } while(0)
#define CHECK_NCCL(call) \
    do { ncclResult_t res = call; if (res != ncclSuccess) { \
        printf("NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); exit(-1); } } while(0)

int main() {
    int M = 1024, N = 1024, K = 1024;
    int comm_count = M * N;
    float *A, *B, *C, *comm_buf;
    CHECK_CUDA(cudaMalloc(&A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&comm_buf, comm_count * sizeof(float)));

    // NCCL init (单卡demo)
    ncclComm_t comm;
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));
    CHECK_NCCL(ncclCommInitRank(&comm, 1, id, 0));

    cudaStream_t compute_stream, comm_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&comm_stream));

    // 选择pipeline模式
    PipelineMode mode = HostPipeline; // HostPipeline 或 KernelPipeline
    int peer_device = -1; // 多卡时可设为目标device

    printf("Running pipeline mode: %s\n", mode == HostPipeline ? "HostPipeline" : "KernelPipeline");
    launch_fc_allreduce_pipeline_v2(A, B, C, M, N, K, comm_buf, comm_count, comm, peer_device, compute_stream, comm_stream, mode);

    // 等待完成
    CHECK_CUDA(cudaStreamSynchronize(compute_stream));
    CHECK_CUDA(cudaStreamSynchronize(comm_stream));

    printf("Pipeline finished!\n");

    ncclCommDestroy(comm);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(comm_buf);
    cudaStreamDestroy(compute_stream); cudaStreamDestroy(comm_stream);
    return 0;
} 