#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include "draw.hpp"
// #include <cuda_provider_factory.h>

using namespace cv;
using namespace std;


void HWCToCHW(const cv::Mat& src, float *dst) {
    std::vector<Mat> vec;
    cv::split(src, vec);
    int hw = src.rows * src.cols;
    memcpy(dst + hw * 0, vec[0].data, hw * sizeof(float));
    memcpy(dst + hw * 1, vec[1].data, hw * sizeof(float));
    memcpy(dst + hw * 2, vec[2].data, hw * sizeof(float));
}

void preprocess(const cv::Mat& img, float *blob) {

    // Rect top_left_roi(107, 0, img.rows, img.rows);
    // Mat img_crop = img(top_left_roi);
    Mat resize_img;
    resize(img, resize_img, Size(512, 512), INTER_AREA);
    Mat blob0 = resize_img.clone();
    blob0.convertTo(blob0, CV_32FC3);
    Mat blob1 = (blob0 - 127.5) / 127.5;
    
    HWCToCHW(blob1, blob);
}

int main(){

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  //设置图优化类型
    // OrtCUDAProviderOptions cuda_options;
    // session_options.AppendExecutionProvider_CUDA(cuda_options);

    const char* model_path = "./modnet.onnx";
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    const char * input_name = session.GetInputNameAllocated(0, allocator).get();
    const char * output_name = session.GetOutputNameAllocated(0, allocator).get();
    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    vector<const char*> input_node_names = { "input" };
    vector<const char*> output_node_names = { "output" };

    std::vector<int64_t> input_node_dims = {1, 3, 512, 512};
    size_t input_tensor_size = 1 * 3 * 512 * 512;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    


    float blob[1 * 3 * 512 * 512];

    //video
    string path="test_v_0.mp4";
    VideoCapture capture(path);

    int frame_width=capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height=capture.get(CAP_PROP_FRAME_HEIGHT);
    int count=capture.get(CAP_PROP_FRAME_COUNT);
    double fps=capture.get(CAP_PROP_FPS);
    VideoWriter writer("video_save.mp4",capture.get(CAP_PROP_FOURCC),fps,Size(frame_width,frame_height));
    Mat img;

    //背景图
    Mat bg = imread("./bg.jpg", 1);
    // FPS开始时间
    auto start = std::chrono::high_resolution_clock::now();
    int img_num=0;
    while(capture.read(img)){

        preprocess(img, blob);
        vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
        std::vector<int64_t> inputShape = input_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        // cout << "Input Dims: " << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3] << endl;

        auto output_tensors = session.Run(Ort::RunOptions(nullptr), input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_node_names.size());
        std::vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        // cout << "Output Dims: " << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << endl;
        
        auto* rawOutput = output_tensors[0].GetTensorData<float>();
        Mat output(512, 512, CV_32FC1, (float*)rawOutput);
        draw_matte(img,output,"background",bg);
        cout<<"saveing video"<<endl;
        writer.write(img);
        img_num+=1;
    }
    // FPS结束时间
    auto end = std::chrono::high_resolution_clock::now();
    //计算微妙
    // auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/ 1000.f;

    //计算毫秒
    std::chrono::duration<double, std::milli> tm = end - start;	// 毫秒
    std::cout << "Duration: " << tm.count() << " ms" << std::endl;
    cout<<"img_num:"<<img_num<<"fps:"<<img_num*1000/tm.count()<<endl;
    // release writer
    writer.release();
    // release capture
    capture.release();
    return 0;
}

