#pragma once
#include <iostream>
#include <string>

class Student
{
public:
    Student(std::string name, int age)
        : name_{std::move(name)}, age_{age}
    {
        std::cout << "student ctor: " << name_ << ", " << age_ << '\n';
    }
    // 拷贝/移动演示
    Student(const Student &s)
    {
        std::cout << "copy-ctor\n";
        copy_from(s);
    }
    Student(Student &&s) noexcept
    {
        std::cout << "move-ctor\n";
        move_from(std::move(s));
    }

    void say() const {
        std::cout << "Hello, I'm " << name_ << ", age " << age_ << std::endl;
    }

    const std::string& getName() const { return name_; }
    int getAge() const { return age_; }

private:
    std::string name_;
    int age_;

    void copy_from(const Student &s)
    {
        name_ = s.name_;
        age_ = s.age_;
    }
    void move_from(Student &&s)
    {
        name_ = std::move(s.name_);
        age_ = s.age_;
    }
};