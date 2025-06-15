# 如何制作库文件
cmake_minimum_required(VERSION 3.5)
project(Test)
# 收集源文件
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 添加头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 设置可执行文件的输出路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# 设置库文件搜索路径
link_directories(${CMAKE_SOURCE_DIR}/lib2)

# 在cmake中链接静态库的命令
# link_libraries. 此命令只能链接静态库
link_libraries(test_static)








# 通过生成的库文件生成一个可执行文件
add_executable(app ${SRC})
