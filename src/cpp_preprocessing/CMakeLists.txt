cmake_minimum_required(VERSION 3.19)
project(image_to_graph_dataset)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED )
find_package(xtensor REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${image_to_graph_dataset_SOURCE_DIR})
# environment variable as SLIC requires opencv_contrib
if (INCLUDE_SLIC)
        ADD_DEFINITIONS( -D_INCLUDE_SLIC=\"INCLUDE_SLIC\" )
endif()
# Only for optimisation (Processor specific optimisations), if problems are encoutered this can be removed
add_compile_options(-march=native)
add_executable(image_to_graph_dataset
        image_to_graph_dataset.cpp 
        data_classes.hpp 
        determine_superpixels.hpp
        image_list_to_graphs.hpp
        DisjointSetForest.hpp)

# Only for optimisation (Link Time Optimisation), if problems are encoutered this can be removed
set_property(TARGET image_to_graph_dataset PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
target_link_libraries(image_to_graph_dataset ${OpenCV_LIBS} )
target_link_libraries(image_to_graph_dataset xtensor)
target_include_directories(image_to_graph_dataset PRIVATE ${SEGMENT})
target_include_directories(image_to_graph_dataset PRIVATE ${xtensor_INCLUDE_DIRS})

# If running on platforms other than Linux this should be replaced with the corresponding math library
target_link_libraries(image_to_graph_dataset m)
# If running on platforms other than Linux this should be replaced with the corresponding threading library
target_link_libraries(image_to_graph_dataset pthread)