#include<iostream>
#include<cstdlib> //for std::aligned_alloc
#include<immintrin.h> //for avx intrinsics
#include<cstring> 

int main()
{
    //分配对齐内存 对齐32字节 总大小也要是对齐值的倍数(32 字节 * 8floats)

    const size_t align = 32;
    const size_t size = sizeof(float) * 8;
    
    // 使用更安全的内存分配方式
    float* data = nullptr;
    if (posix_memalign((void**)&data, align, size) != 0) {
        std::cerr << "Memory allocation failed" << std::endl;
        return 1;
    }
    std::cout << "Memory allocated at: " << static_cast<void*>(data) << std::endl;

    //初始化数据
    // for (int i = 0; i < 8; ++i) {
    //     data[i] = i + 1.0f;
    // }
    // std::cout << "Data initialized." << std::endl;

    //用avx
    std::cout << "About to load with AVX..." << std::endl;
    __m256 vec = _mm256_load_ps(data);//只能加载对齐内存
    std::cout << "Loaded with AVX." << std::endl;
    alignas(32) float result[8];
    _mm256_store_ps(result, vec);



    //打印结果
    std::cout << "Loaded AVX data: ";
    for(float f : result)
        std::cout << f << " ";
    std::cout << std::endl;

    //释放内存
    free(data);
    return 0;
}

