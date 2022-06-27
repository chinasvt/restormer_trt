#include <fstream>
#include "trt.h"

TRT::TRT(const std::string& modelFile, const std::string& engineFile, const int& batchSize, cudaStream_t stream)
        :stream_{stream},
         batch_{batchSize}{
    std::fstream trtCache(engineFile, std::ifstream::in);

    if(!trtCache.is_open()){
        std::cout << "engine file is not esxit, start building it!\n";
        //convert onnx to trt engine
        builder_ = nvinfer1::createInferBuilder(logger_);
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network_ = builder_->createNetworkV2(explicitBatch);
        parser_ = nvonnxparser::createParser(*network_, logger_);

        if(!parser_->parseFromFile(modelFile.c_str(), static_cast<int >(nvinfer1::ILogger::Severity::kWARNING))){
            std::cerr << ": failed to parse onnx model file, please check the onnx version and trt support op!"
                      << std::endl;
            exit(-1);
        }

        builder_config_ = builder_->createBuilderConfig();
#ifdef FP16
        builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
#endif
        builder_->setMaxBatchSize(batch_);
        builder_config_->setMaxWorkspaceSize(size_t(8) << 30);

        engine_ = builder_->buildEngineWithConfig(*network_, *builder_config_);
        if(!engine_){
            std::cerr << ": engine init null!" << std::endl;
            exit(-1);
        }

        auto trtModelStream = engine_->serialize();
        std::fstream trtOut(engineFile, std::fstream::out);
        if(!trtOut.is_open()){
            std::cout << "Can't store trt cache.\n";
            exit(-1);
        }
        trtOut.write((char*)trtModelStream->data(), trtModelStream->size());
        trtOut.close();
        destroy_trt_pointer(trtModelStream);
    } else{
        std::cout << "load TRT engine file\n";
        char* data;
        unsigned int length;
        trtCache.seekg(0, trtCache.end);
        length = trtCache.tellg();
        trtCache.seekg(0, trtCache.beg);
        data = (char*)malloc(length);
        if(!data){
            std::cout << "Can't malloc data.\n";
            exit(-1);
        }
        trtCache.read(data, length);

        runtime_ = nvinfer1::createInferRuntime(logger_);
        if(!runtime_){
            std::cerr << ": runtime null!" << std::endl;
            exit(-1);
        }

        engine_ = runtime_->deserializeCudaEngine(data, length, 0);
        if(!engine_){
            std::cerr << ": engine null!" << std::endl;
            exit(-1);
        }

        free(data);
        trtCache.close();
    }
    context_ = engine_->createExecutionContext();
#ifdef PROFILE
    context_->setProfiler(&profiler_);
#endif
   
}

TRT::~TRT() {
    //runtime
    destroy_trt_pointer(context_);
    destroy_trt_pointer(engine_); //builder also need
    destroy_trt_pointer(runtime_);
    //build
    destroy_trt_pointer(builder_config_);
    destroy_trt_pointer(parser_);
    destroy_trt_pointer(network_);
    destroy_trt_pointer(builder_);
//    checkCudaErrors(cudaEventDestroy(start_));
//    checkCudaErrors(cudaEventDestroy(stop_));
}

bool TRT::doInfer(void** buffers) {
    std::cout << "start inference...\n";
#ifdef PROFILE
    const int iterations = 100;
    if(context_){
        printf("111111111111\n");
    }
    for(int i = 0; i < iterations; i++){
        context_->executeV2(buffers);
    }
    profiler_.printLayerTime(iterations);
#endif
    if(!context_->enqueueV2(buffers, stream_, nullptr)){
        std::cerr << "inference failed!\n";
        return false;
    }
    return true;
}

