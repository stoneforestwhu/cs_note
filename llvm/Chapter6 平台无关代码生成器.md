## Chapter6 平台无关代码生成器

### 概述

平台无关的代码生成器为从IR到机器指令的转换提供了一个抽象层。IR被转换成SelectionDAG。

### 6.1 LLVM IR指令的生命周期

IR的指令被转换成SelectionDAG的节点来表示，整个过程如下：

1、由LLVM IR创建SelectionDAG

2、SelectionDAG节点合法化

3、DAG合并优化

4、针对目标指令的指令选择

5、调度并发射机器指令

6、寄存器分配--SSA解构、寄存器赋值、寄存器溢出

7、发射机器码

#### 6.6.1 C代码到LLVM IR

#### 6.1.2 IR优化

#### 6.1.3 LLVM IR转化为SelectionDAG

#### 6.1.4 合法化SelectionDAG

SelectionDAG节点未必会被目标架构全部支持，因此需要对DAG节点做出一点修改以适应目标平台

SelectionDAG节点合法化包括**数据类型**和**操作**两个方面。

#### 6.1.5 从目标无关平台DAG转换为机器DAG

机器指令由一个通用的基于表的.td文件描述，之后这些文件通过tablegen工具转为.inc文件，再inc文件中用枚举类型描述了目标平台的寄存器、指令集等信息，并且可以直接被C++代码调用。指令选择的过程可以由SelectCode自动选择器完成，或者通过编写自定义的SelectionDAGISel::Select函数自己定制。在这一步中创建的DAG节点是MachineSDNode节点，它是SDNode的子类，持有构建真实机器指令的必要信息，但仍是DAG节点形式。

#### 6.1.6 指令调度

对DAG进行拓扑排序可以把DAG形式的机器指令变为线性指令集，然后再在顺序上进行优化。优化的过程称为指令调度。

解决指令依赖、寄存器压力、流水线阻塞等问题

#### 6.1.7 寄存器分配

LLVM采用了贪心法来进行寄存器分配，活动周期越长的变量先分配寄存器。生存周期段的变量则填补可用寄存器的时间间隙，减少溢出权重。

#### 6.1.8 代码发射

LLVM中的代码发射有两种方式：

1、JIT 直接把代码发射到内存

2、MC框架 这个不太懂???

### 6.2 使用GraphViz可视化LLVM IR控制流图

略，Ubuntu系统中工具，不过很有用。可以使用GraphViz查看IR转换的DAG.

SelectionDAG节点和边的描述参考 https://zhuanlan.zhihu.com/p/52724656

### 6.3 使用TableGen描述目标平台

### 6.4 定义指令集

6.3和6.4可以并为同一节，6.3表示使用*.td描述目标平台的寄存器，6.4节表示使用\*.td描述目标平台的指令

### 6.5 添加机器码描述

​         LLVM IR 通过 class Function保存函数信息，函数有class BasicBlock对象组成，BasicBlock由class Instruction对象组成。把IR抽象块的内容转换为指定机器的区块，即是把Function对象转换成指定机器的MachineFunction、MachineBasicBlock、MachineInstr对象。

​        LLVM IR  -->  machine code ==> 构建MachineInstr对象。

​        机器指令的表示由 操作码和多个操作数组成。

​        MachineInstr构造函数：

```c++
MachineInstr::MachineInstr(MachineFunction &MF, const MCInstrDesc &tid, const DebugLoc dl, bool NoImp)
```

​         给指令添加操作数：

```c++
void MachineInstr::addOperand(MachineFunction &MF, const MachineOperand &Op)
```

​         给指令添加n内存操作数：

```c++
void MachineInstr::addMemOperand(MachineFunction &MF, MachineMemOperand *MO)
```

​        MachineInstr类有一个MCInstrDesc类型的成员MCID来描述指令，一个内存引用成员(mmo_iterator MemRefs)，一个std::vector<MachineOperand>草最熟的向量成员(这个应该改成MachineOperand*指令了)

### 6.6 实现MachineInstrBuilder类

创建指令的方法如下：

1、创建指令 mov DestReg, 42

```c++
MachineInstr *MI = BuildMI(X86::MOV32ri, 1, DestReg).addImm(42);
//  根据LLVM6.0的源码，MachineInstrBuilder::addImm()的返回值还是MachineInstrBuilder,所以用法应该不太一样
```

2、创建指令 mov DestReg, 42，并放置再BasicBlock的最后

```c++
MachineBasicBlock &MBB =    //  此处应该缺少MachineBasicBlock的constructor
BuildMI(MBB, X86::MOV32ri, 1, DestReg).addImm(42);
```

3、创建指令 mov DestReg, 42，并把它放在指定的迭代器之前

```c++
MachineBasicBlock::iterator MBBI =  //  此处应该缺少MBBI的初始化
BuildMI(MBB, MBBI, X86::MOV32ri, 1, DestReg).addImm(42)
```

4、创建一个自循环分支指令

```c++
BuildMI(MBB, X86::JNE, 1).addMBB(&MBB);
```

### 6.7 实现MachineBasicBlock类

​        大多数LLVM IR的一个BasicBlock可以映射到一个machineBasicBlock对象，有些BasicBlock会映射到多个MachineBasicBlock对象。

​        class MachineBasic提供了getBasicBlock()方法返回它映射到的IR BasicBlock

### 6.8 实现MachineFunction类

​        MachineFunction对应到LLVM IR的FunctionBlock类，MachineFunction类包含一系列的MachineBasicBlock类。MahineFunction类的信息会映射到LLVM IR函数，作为指令选择器的输入。

​        MachineFunction类主要是保存MachineBasicBlock对象的列表，为检索机器函数和修改基本块成员中的对象定义提供了多个方法。MachineFunction类还为函数中的基本块维护了一个控制流图(control flow graph)

### 6.9 编写指令选择器

​        LLVM使用SelectionDAG表示LLVM IR。

​         SelectionDAG图和它的节点SDNode被设计来既可以存储平台无关的信息也可以存储特定平台的信息。

### 6.10 合法化SelectionDAG

1、合法化不支持的数据类型：

​       第一种是标量。标量可以被promoted（将较小的类型转成较大的类型，比如平台只支持i32，那么i1/i8/i16都要提升到i32）, expanded（将较大的类型拆分成多个小的类型，如果目标只支持i32，加法的i64操作数就是非法的。这种情况下，类型合法化通过integer expansion将一个i64操作数分解成2个i32操作数，并产生相应的节点）；第二种是矢量。LLVM IR中有目标平台无法支持的矢量，LLVM也会有两种转换方案， 加宽（widening） ，即将大vector拆分成多个可以被平台支持的小vector，不足一个vector的部分补齐成一个vector；以及 标量化（scalarizing） ，即在不支持SIMD指令的平台上，将矢量拆成多个标量进行运算。(来自 https://zhuanlan.zhihu.com/p/52724656)

2、合法化不支持的指令

未提。

### 6.11 优化SelectionDAG

DAGCombine

### 6.12 基于DAG的指令选择

​        指令选择阶段需要把平台无关的指令转换为特定平台的指令。TableGen类帮助选择特定平台的指令。这个阶段基本上就是匹配平台无关的输入节点，输出特定平台支持的节点。

​        CodeGenAndEmitDAG()࠭函数调用DoInstructionSelection()函数，遍历DAG节点并对每个节点调用Select()函数，如下： 

```c++
SDNode *ResNode = Select(Node);  
```

​       Select()函数是需要由特定平台实现的抽象方法。x86目标平台实现了X86DAGToDAGISel::Select()函数。这个函数拦截一部分节点进行手动匹配，而大部分工作则委托给X86DAGToDAGISel::SelectCode()函数完成。

​       X86DAGToDAGISel::SelectCode函数由TableGen自动生成。它包含一个匹配表，随后调用SelectionDAGISel::SelectCodeCommon()泛型函数，并把匹配表传给它。

### 6.13 基于SelectionDAG的指令调度

​        调度器负责安排DAG中指令的执行顺序。此过程中考虑的启发式优化过程有：

​        1、考虑寄存器压力以优化指令顺序执行顺序

​         2、最小化指令执行的延迟时间

​         。。。

​         经过调度之后，DAG节点转化为MachineInstrs列表并且SelectionDAG节点被解构。

​        调度器基类是ScheduleDAG。

​        调度算法实现了SelectionDAG类中的指令调度，包括拓扑飘絮、深度优先搜索、操作函数、移动节点、指令列表迭代等算法。





























































































































































User和Instruction类的关系



User是Value的子类

Instruction是User的子类，但是Instruction是多继承，父类还有一个ilist_node_parent<Instruction, BasicBlock>



X86TargetLowering  接口将目标平台(x86)的关键信息和平台无关的算法连接起来。

LLVM Calling Convention Representation































