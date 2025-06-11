#include<iostream>
#include<cstdlib> //for std::aligned_alloc
#include<immintrin.h> //for avx intrinsics
#include<cstring> 

int main()
{
    //分配对齐内存 对齐32字节 总大小也要是对齐值的倍数(32 字节 * 8floats)

    size_t align = 32;
    size_t size = sizeof(float)*8;
    float* data = (float*)std::aligned_alloc(align,size);

    //初始化数据 也可以不初始化
    for (int i = 0;i<8;++i){
        data[i] = i+1.0f;
    }

    //用avx
    __m256 vec = _mm256_load_ps(data);//只能加载对齐内存
    float result[8];
    _mm256_store_ps(result,vec);

    //打印结果
    std::cout<<"Loaded AVX data";
    for(float f: result)
        std::cout<<f<<" ";
    std::cout<<std::endl;

    //释放内存()
    free(data);
    return 0;
}

