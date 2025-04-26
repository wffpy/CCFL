#pragma once
#include <cuda_runtime.h>
#include <nccl.h>

// pipeline模式选择
enum PipelineMode {
    HostPipeline = 0, // host端分块调度
    KernelPipeline = 1 // kernel内部分块调度（实验性）
};

// 启动全连接层（fc）kernel，使用tensor core
void launch_fc_kernel(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream);

// 启动all-reduce通信
void launch_allreduce(float* data, int count, ncclComm_t comm, cudaStream_t stream);

// 方案一：host端分块pipeline
void launch_fc_allreduce_pipeline(float* A, float* B, float* C, int M, int N, int K, float* comm_buf, int comm_count, ncclComm_t comm, cudaStream_t compute_stream, cudaStream_t comm_stream);

// 方案二：kernel内部pipeline（实验性，通信用P2P模拟）
void launch_fc_allreduce_pipeline_kernel(float* A, float* B, float* C, int M, int N, int K, float* comm_buf, int comm_count, int peer_device, cudaStream_t stream);

// 通用接口：根据mode选择pipeline方案
void launch_fc_allreduce_pipeline_v2(float* A, float* B, float* C, int M, int N, int K, float* comm_buf, int comm_count, ncclComm_t comm, int peer_device, cudaStream_t compute_stream, cudaStream_t comm_stream, PipelineMode mode); 