//
// Created by sunc on 2022/6/11.
//

#pragma once

#include "trt.h"
#include <memory>


class Restormer{
public:
    Restormer(cudaStream_t stream, const std::string& modelFile, const std::string& engineFile,
                                    const int N, const int C, const int H, const int W);

    ~Restormer();

    bool doInfer(float* input);
private:
    float* input_{nullptr};
    float* output_{nullptr};
    float* output_host_{nullptr};
    std::shared_ptr<TRT> trt_;
    cudaStream_t stream_;
    int batch_{0};
    int net_height_{0};
    int net_width_{0};
    int net_channel_{0};
    int stride_{0};
};