### 1、子类调用父类构造函数

​        一般情况下，父类如果有有参构造函数，子类需要在**初始化列表**中显式调用父类的有参构造函数(在函数体中调用没有用)。如果父类有无参的构造函数(可能是编译器合成的默认构造函数，也有可能是程序员自己实现的无参构造函数)，那么子类的构造函数会隐式调用这个无参构造函数。

### 2、友元的访问范围

​        友元函数或者类可以访问类的私有成员变量

### 3、类成员变量初始化

#### 静态成员

c++类的静态成员变量，必须在类内声明，在类外定义和初始化，如下例：

```c++
class A
{
public:
	static int a;   // 只是声明，并没有定义，也就没有分配存储空间
};
int A::a=0;    //  定义和初始化
```

#### 一般变量成员

一般变量可以在初始化列表里或者构造函数里初始化，**不能直接初始化**或者类外初始化

#### 常量成员

常量**必须在初始化列表**里初始化

#### 静态常量成员

静态常量必须只能在定义的时候初始化（定义时直接初始化）

```c++
#include <iostream> 
#include <string> 
using namespace std; 
class Test 
{ 
private: 
    int a; 
    static int b; 
    const int c; 
    static const int d=4;   //  直接初始化
public: 
    Test():c(3)     //  注意c只能在初始化列表中初始化
    { 
        a=1; 
    } 
}; 
int Test::b=2;    //  类外初始化

void main() 
{ 
    Test t; 
}
```

reference  https://blog.csdn.net/sinat_39370511/article/details/91985428

### 4、操作符重载定义的位置

​        定义成成员函数或者友元函数的有一个很重要的区别在于操作符可能没有形参，如果定义成友元函数编译器无法判断出使用哪一个函数

+, - 声明在类内，定义在类外

(), [], -> 声明定义都在类内(感觉定义在类外也可以)  只能定义成成员函数，不能定义成友元函数

= 赋值操作符必须定义为成员函数

++，-- 只能定义成成员函数 前置++没有形参，后置++只有一个不参与运算的int作为形参(哑变量)

\>\> 、<<  为了习惯定义成友元函数

+一般作为友元函数但是 += 一般作为成员函数

><<  为了习惯定义成友元函数

reference  https://blog.csdn.net/creator123123/article/details/81572273

​                   https://zhuanlan.zhihu.com/p/39377425

### 5、智能指针
### 6、new操作符
参考 https://www.cnblogs.com/balingybj/p/4695108.html
placement new仅在一个已经分配好的内存指针上调用构造函数，基本形式如下：
void* operator new (std::size_t size, void* ptr) noexcept;
new (1) class_constructor(...)

