#include<memory>
#include<iostream>

int main()
{
    std::allocator<int> alloc;
    int *p = alloc.allocate(3);//分配内存 不初始化

    alloc.construct(p,42);
    alloc.construct(p+1,77);
    alloc.construct(p+2,99);

    std::cout<<p[0]<<" "<<p[1]<<" "<<p[2]<<" "<<std::endl;
    
    //析构&释放
    alloc.destroy(p);
    alloc.destroy(p+1);
    alloc.destroy(p+2);
    alloc.deallocate(p,3);
    return 0;

}