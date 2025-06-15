#pragma once
#include<memory>
#include<utility>
#include<iostream>

/*
A make_smart 带日志的封装 + 完美转发
*/
template<typename T,typename...Args>
std::shared_ptr<T> make_smart(Args&&... args)
{
    std::cout<<"make_smart creating shared_ptr\n";
    //完美转发到 std::make_shared
    return std::make_shared<T>(std::forward<Args>(args)...);
}

/**
 * B  wrapper 任意类型的完美转发包装器
 */
template<typename T>
class Wrapper{
public:
    //接收并且完美转发一切到 T的构造函数
    template<typename ... Args>
    explicit Wrapper(Args... args)
        : obj_{std::forward<Args>(args)...}
    {}
    //提供接口访问
    T* operator->() {return &obj_;}
    const T* operator->() const{return &obj_;}

    T& get() {return obj_;}
    const T& get() const{ return obj_;}

private:
    T obj_;
};
