#include "student.hpp"
#include "utils.hpp"
#include <vector>

int main()
{
    std::cout << "=== 1. 用 make_shared 直接构造 ===\n";
    auto p1 = std::make_shared<Student>("Alice", 20);
    p1->say();

    std::cout << "\n=== 2. 用 make_smart (封装 + 日志) ===\n";
    auto p2 = make_smart<Student>("Bob", 22);
    p2->say();

    std::cout << "\n=== 3. Wrapper 演示（完美转发） ===\n";
    Wrapper<Student> w("Charlie", 24);   // 自动转发到 Student(string,int)
    w->say();

    std::cout << "\n=== 4. Wrapper 再配合 shared_ptr ===\n";
    auto pw = std::make_shared<Wrapper<Student>>("Dave", 26);
    pw->get().say();        // 访问里面的 Student

    std::cout << "\n=== 5. vector 里直接放 Wrapper<Student> ===\n";
    std::vector<Wrapper<Student>> vec;
    vec.emplace_back("Eve", 28);         // 仍然完美转发
    vec.back()->say();

    std::cout << "\n=== 程序结束，自动析构、自动释放内存 ===\n";
}