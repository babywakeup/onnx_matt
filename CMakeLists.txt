cmake_minimum_required(VERSION 3.10)  
# 启用调试标志  
add_compile_options(-g)  

project(YourProjectName)  


# 添加OpenCV库  
find_package(OpenCV REQUIRED)  
include_directories(${OpenCV_INCLUDE_DIRS})  
  
# # 添加CUDA库  
# find_package(CUDA REQUIRED)  
# include_directories(${CUDA_INCLUDE_DIRS})  
# link_directories(${CUDA_LIBRARY_DIRS})  
  

set(ONNXRUNTIME_INCLUDE_DIRS /onnxruntime-linux-x64-1.15.1/include)
set(ONNXRUNTIME_LIBRARY_DIRS /usr/lib /app/.local/lib)
# 添加ONNX Runtime库  
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})  
link_directories(${ONNXRUNTIME_LIBRARY_DIRS})  

#添加自己的头文件
include_directories(./include)
set(onnxruntime_providers_shared /usr/local/lib/python3.8/dist-packages/onnxruntime/capi/libonnxruntime_providers_shared.so)
# 添加你的源文件  
add_executable(YourProjectName main.cpp draw.cpp) # 或者你需要的其他源文件  
  
# 链接你的项目到OpenCV、CUDA和ONNX Runtime库  
target_link_libraries(YourProjectName onnxruntime ${OpenCV_LIBS} ${onnxruntime_providers_shared})#${CUDA_LIBS}
