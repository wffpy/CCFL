# CCFL
Computation And Communication Fusion Library

本项目演示如何用 CUDA stream 实现全连接层（tensor core）与 all-reduce 通信的流水线并行，加速分布式训练。

## 主要特性
- Tensor Core 加速的全连接层（fc kernel）
- NCCL all-reduce 通信
- 计算与通信流水线并行（pipeline）

## 目录结构
```
CCF/
├── CMakeLists.txt
├── README.md
├── include/
│   └── fc_allreduce_pipeline.cuh
├── src/
│   ├── main.cu
│   └── fc_allreduce_pipeline.cu
├── test/
│   └── test_pipeline.cu
└── scripts/
    └── run.sh
```

## 构建
```bash
mkdir build && cd build
cmake ..
make
```

## 运行
```bash
./ccf_test
```

## 依赖
- CUDA 11.0+
- NCCL 2.x
- CMake 3.10+ 