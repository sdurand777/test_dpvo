
# optional libraries
find_package(Pangolin REQUIRED)

message(STATUS "CMakeLists de viewerdpvo")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
#pybind11_add_module(dpviewerx viewer.cpp viewer_cuda.cu)
pybind11_add_module(viewerdpvo viewer.cpp viewer_cuda.cu)
include_directories(${Pangolin_INCLUDE_DIRS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(dpviewerx PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
# target_link_libraries(dpviewerx PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${Pangolin_LIBRARIES})
target_compile_definitions(viewerdpvo PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(viewerdpvo PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${Pangolin_LIBRARIES})

