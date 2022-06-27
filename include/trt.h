#pragma once

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"

#define PROFILE
#define FP16

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

template <typename T>
static void destroy_trt_pointer(T* t){
    if(t) t->destroy();
}

static void Getinfo(void)
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

class Logger : public nvinfer1::ILogger{
public:
    void log(Severity severity, const char* msg) noexcept override{
        if(severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR){
            std::cerr << "trt_infer: " << msg << std::endl;
        }
    }
};

struct Profiler : public nvinfer1::IProfiler{
    using Record = std::pair<std::string, float >;
    std::vector<Record> mProfiler;

    void reportLayerTime(const char* layerName, float ms) noexcept override{
        auto record = std::find_if(mProfiler.begin(), mProfiler.end(), [&](const Record& r){return r.first == layerName;});
        if(record == mProfiler.end()){
            mProfiler.emplace_back(std::make_pair(layerName, ms));
        } else{
            record->second += ms;
        }
    }

    void printLayerTime(const int iterations){
        float totalTime = 0.f;
        for(auto &profiler : mProfiler){
            std::cout << profiler.first << " " << profiler.second / iterations << "ms\n";
            totalTime += profiler.second;
        }
        std::cout << "Time all layers: " << totalTime / iterations << "ms\n";
    }
};

class TRT{
public:
    TRT(const std::string& modelFile, const std::string& engineFile, const int& batchSize, cudaStream_t stream = 0);

    ~TRT();

    bool doInfer(void** buffers);

private:
    int batch_{0};
//    cudaEvent_t start_, stop_;
    Logger logger_;
    Profiler profiler_;
    //inference model
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    //build model
    nvinfer1::IBuilder* builder_{nullptr};
    nvinfer1::INetworkDefinition* network_{nullptr};
    nvinfer1::IBuilderConfig* builder_config_{nullptr};
    nvonnxparser::IParser* parser_{nullptr};
    cudaStream_t stream_{0};

};
