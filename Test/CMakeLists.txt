cmake_minimum_required(VERSION 3.1)
project( OCVTest )
find_package( OpenCV REQUIRED )
find_package( MPI REQUIRED )
add_executable( OCVTest main.cpp)
add_definitions( -DOMPI_SKIP_MPICXX )
#Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#OpenCV dir
SET(OPENCV_DIR /usr/local/OpenCV/lib/cmake/opencv4)
include_directories(OPENMPI_DIR /usr/include/x86_64-linux-gnu/openmpi )
include_directories(MPI_DIR /usr/include/x86_64-linux-gnu/mpi )
target_link_libraries( OCVTest ${OpenCV_LIBS} ${MPI_C_LIBRARIES} )


