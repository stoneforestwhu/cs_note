### 1、子类调用父类构造函数

​        一般情况下，父类如果有有参构造函数，子类需要在**初始化列表**中显式调用父类的有参构造函数(在函数体中调用没有用)。如果父类有无参的构造函数(可能是编译器合成的默认构造函数，也有可能是程序员自己实现的无参构造函数)，那么子类的构造函数会隐式调用这个无参构造函数。

### 2、友元的访问范围

​        友元函数或者类可以访问类的私有成员变量
