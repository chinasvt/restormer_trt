//
// Created by sunc on 2022/6/11.
//

#include "restormer.h"

Restormer::Restormer(cudaStream_t stream, const std::string &modelFile, const std::string &engineFile, const int N,
                     const int C, const int H, const int W)
                     :stream_{stream},
                      batch_{N},
                      net_channel_{C},
                      net_height_{H},
                      net_width_{W}{
        trt_ = std::make_shared<TRT>(TRT(modelFile, engineFile, batch_, stream_));
        stride_ = net_channel_ * net_height_ * net_width_;
        checkCudaErrors(cudaMalloc((void**)&input_, batch_ * stride_ * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&output_, batch_ * stride_ * sizeof(float)));
        output_host_ = (float*)malloc(batch_ * stride_ * sizeof(float));
}

Restormer::~Restormer() {
    trt_.reset();
    checkCudaErrors(cudaFree(input_));
    checkCudaErrors(cudaFree(output_));
    free(output_host_);
}

bool Restormer::doInfer(float* input) {
    checkCudaErrors(cudaMemcpy(input_, input, batch_ * stride_ * sizeof(float), cudaMemcpyHostToDevice));
    void* buffers[] = {input_, output_};
    trt_->doInfer(buffers);
}
