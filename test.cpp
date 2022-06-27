//
// Created by sunc on 2022/6/11.
//
#include <fstream>
#include <iostream>
#include <cassert>
#include "trt.h"


#define DEVICE 0  // GPU id
#define BATCH_SIZE 1


// stuff we know about the network and the input/output blobs
static const int INPUT_H = 368;  // H, W must be able to  be divided by 32.
static const int INPUT_W = 552;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W * 3;
const char* INPUT_BLOB_NAME = "input.1";
const char* OUTPUT_BLOB_NAME = "29528";
#define iterations 1000

static Logger gLogger;
//static Profiler gProfiler;

void load_data(const char* filename, void** data, unsigned int& length){
    std::ifstream ifs;
    ifs.open(filename, std::ifstream::in);
    if(!ifs.is_open()){
        std::cout << "read file failed!!!!\n";
        exit(-1);
    }
    ifs.seekg(0, ifs.end);
    unsigned int len = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    char* buffer = new char[len];
    ifs.read(buffer, len);
    *data = (void*)buffer;
    length = len;
    ifs.close();
}

void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize) {
    const nvinfer1::ICudaEngine& engine = context.getEngine();
    //context.setProfiler(&gProfiler);
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    checkCudaErrors(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    checkCudaErrors(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //for(int i = 0; i < iterations; i++){
    //	context.execute(batchSize, buffers);
    //}
    //gProfiler.printLayerTime(iterations);
    context.enqueue(batchSize, buffers, stream, nullptr);
    checkCudaErrors(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    checkCudaErrors(cudaFree(buffers[inputIndex]));
    checkCudaErrors(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {

    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("../models/restormer.trt", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // prepare input data ---------------------------
    unsigned int length = 0;
    const std::string filename = "../data/input.bin";
    void* data = NULL;
    std::shared_ptr<char> buffer((char*)data, std::default_delete<char[]>());
    load_data(filename.data(), &data, length);
    buffer.reset((char*)data);
    float* input = (float*)buffer.get();
    size_t buffer_count = length / sizeof(float);
    assert(INPUT_H * INPUT_W * 3 == buffer_count && "input count must be same!");
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    //doInference(*context, data, prob, BATCH_SIZE);
    doInference(*context, input, prob, BATCH_SIZE);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    std::ofstream out;
    out.open("../data/result.txt", std::ios::out);
    for(int i = 0; i < OUTPUT_SIZE * BATCH_SIZE; i++){
        out << prob[i];
	out << "\n";
    }
    out.close();
    return 0;
}
