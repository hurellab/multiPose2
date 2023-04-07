cmake_minimum_required(VERSION 3.16)
project(poseEst)
set(CMAKE_BUILD_TYPE Release)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
option(LIBIGL_GLFW             "Build target igl::glfw"            ON)
option(LIBIGL_COPYLEFT_TETGEN  "Build target igl_copyleft::tetgen" ON)
include(libigl)

# Add your project files
file(GLOB SRC_FILES *.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl_copyleft::tetgen)

# copy files required for TGF generation
set(DATA_SCRIPTS
  yolo.tgf
  )

foreach(_script ${DATA_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
    )
endforeach()