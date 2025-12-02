# Stiff-GIPC 代码架构详解

本文档详细解析了 Stiff-GIPC 项目的代码架构。Stiff-GIPC 是一个基于 GPU 的高性能物理仿真框架，专门针对刚性仿射体 (ABD) 和可变形体 (FEM) 的强耦合仿真进行了优化，并实现了增量势能接触 (IPC) 算法以保证无穿透的稳定模拟。

---

## 1. 项目概览与目录结构

项目采用 C++/CUDA 混合编程，利用 GPU 进行大规模并行计算。核心逻辑位于 `StiffGIPC` 目录下。

### 1.1 目录树核心解析

```
StiffGIPC/
├── gl_main.cu                 # [入口] OpenGL 渲染、窗口管理、场景加载与仿真主循环驱动
├── GIPC.cu / .cuh             # [核心] 物理引擎主类，管理时间步进、非线性优化流程
├── abd_system/                # [子系统] 仿射刚体 (Affine Body Dynamics) 系统
│   ├── abd_system.cu          # ABD 系统的能量、梯度、Hessian 计算
│   └── abd_sim_data.cu        # ABD 状态数据 (位置 q, 速度 v 等) 的 GPU 显存管理
├── linear_system/             # [求解器] 线性方程组求解模块
│   ├── global_linear_system.cu# 全局系统组装 (Assembly)，管理多个子系统
│   └── solver/pcg_solver.cu   # 预处理共轭梯度 (PCG) 求解器
├── gipc/                      # [通用] 基础定义与工具
│   └── utils/                 # 计时器、JSON加载、数学工具等
├── cuda_tools/                # [底层] CUDA 核函数封装、原子操作、内存管理辅助
└── muda/                      # [依赖] 外部库，提供更高级的 CUDA 图计算和内存视图抽象
```

---

## 2. 核心类与数据结构

### 2.1 `GIPC` 类 (GIPC.cuh)
这是整个物理引擎的控制器。它不直接存储大规模网格数据，而是持有对数据的引用和计算流程的控制权。

*   **主要职责**:
    *   **初始化**: `init()`, `build_gipc_system()` 负责分配 GPU 显存，初始化 BVH，构建碰撞对。
    *   **时间步求解**: `IPC_Solver()` 是仿真步进的核心函数。
    *   **碰撞检测**: 管理 `lbvh` (Linear BVH) 进行宽阶段碰撞检测，构建 `_collisonPairs`。
    *   **势能计算**: 计算 IPC 势垒函数 (Barrier Energy) 及其导数。

### 2.2 `device_TetraData` (device_fem_data.cuh)
这是主要的数据容器，所有物理量都存储在 GPU 显存 (Device Memory) 上。为了性能，它采用 **SoA (Structure of Arrays)** 布局。

*   **关键成员**:
    *   `vertexes`, `velocities`: 顶点的当前位置和速度。
    *   `rest_vertexes`: 顶点的静止位置（用于计算形变）。
    *   `masses`: 节点质量。
    *   `tetrahedras`: 四面体拓扑索引。
    *   `DmInverses`: 预计算的形变梯度逆矩阵（用于 FEM）。

### 2.3 `ABDSystem` (abd_system/abd_system.h)
专门处理仿射刚体子系统的类。与传统 FEM 不同，ABD 将物体简化为 12 自由度的仿射变换（线性变换 + 平移）。

*   **关键成员**:
    *   `q`: 广义坐标向量 (12维)。
    *   `abd_gradient`, `abd_body_hessian`: 刚体系统的梯度和 Hessian 块。
    *   `init_system()`: 初始化刚体属性。

### 2.4 `GlobalLinearSystem` (linear_system/global_linear_system.h)
负责将物理问题转化为数学问题 $Ax=b$ 并求解。

*   **主要职责**:
    *   **Assembly (组装)**: 将 ABD 子系统和 FEM 子系统的 Hessian 矩阵块，以及 IPC 约束的 Hessian 组装成一个全局稀疏矩阵。
    *   **Solver**: 调用 PCG 求解器。

---

## 3. 核心仿真流程 (Simulation Loop)

仿真主循环在 `GIPC::IPC_Solver` 中实现。这是一个典型的 **隐式时间积分 (Implicit Time Integration)** 结合 **非线性优化** 的流程。

### 3.1 流程图

1.  **预测 (Prediction)**
    *   计算无约束的预测位置 $\tilde{x} = x_n + h v_n + h^2 M^{-1} f_{ext}$。
    *   对应代码: `computeXTilta`。

2.  **碰撞检测预处理**
    *   更新 BVH 结构。
    *   构建潜在碰撞对 (Collision Pairs)。
    *   对应代码: `buildBVH`, `buildCP`。

3.  **牛顿-拉夫逊迭代 (Newton-Raphson Iteration)**
    *   求解目标函数：$E(x) = \frac{1}{2}(x - \tilde{x})^T M (x - \tilde{x}) + h^2 E_{elastic}(x) + \kappa E_{barrier}(x)$。
    *   **Step 3.1: 计算梯度与 Hessian**
        *   计算弹性力、惯性力、摩擦力和接触势垒力的贡献。
        *   对应代码: `computeSoftConstraintGradientAndHessian`, `setup_abd_system_gradient_hessian`。
    *   **Step 3.2: 线性求解**
        *   求解 $H \Delta x = -g$ 得到搜索方向 $\Delta x$。
        *   使用 PCG 求解器。
    *   **Step 3.3: 连续碰撞检测 (CCD) 与线搜索 (Line Search)**
        *   计算最大步长 $\alpha_{CCD}$ 保证无穿透。
        *   回溯线搜索 (Backtracking Line Search) 确保总能量下降。
        *   对应代码: `solve_subIP`, `self_largestFeasibleStepSize`.

4.  **状态更新**
    *   $x_{n+1} = x_k$
    *   $v_{n+1} = (x_{n+1} - x_n) / h$
    *   对应代码: `updateVelocities`.

---

## 4. 关键技术实现细节

### 4.1 GPU 并行计算
项目大量使用 CUDA Kernel 进行并行计算。
*   **原子操作**: 在从节点累加力/Hessian 到全局矩阵时，使用了 `atomicAdd` 避免竞争。
*   **归约算法 (Reduction)**: 在计算总能量、最大误差时，使用了并行归约算法 (如 `thrust::reduce` 或自定义实现)。

### 4.2 空间哈希与 BVH
为了加速碰撞检测，项目使用了两级加速结构：
1.  **LBVH (Linear Bounding Volume Hierarchy)**: 用于快速剔除不相交的物体对。
2.  **Spatial Hashing**: 在某些局部检测中辅助使用，通过 `_calcTetMChash` 计算莫顿码 (Morton Code) 对图元进行排序，提高显存访问的局部性。

### 4.3 混合求解 (Hybrid Solver)
代码通过 `abd_fem_count_info` 精确管理 ABD 和 FEM 的自由度索引。
*   **映射机制**: 建立全局索引到子系统索引的映射，使得线性求解器可以无缝处理两种不同类型的自由度。

---

## 5. 修改与扩展指南

### 如何添加新场景
1.  在 `StiffGIPC/Assets` 中准备网格模型 (.obj, .msh)。
2.  在 `gl_main.cu` 中编写新的 `set_caseX()` 函数，使用 `SimpleSceneImporter` 加载模型并指定类型 (ABD 或 FEM)。
3.  修改 `initScene` 函数调用你的新场景。

### 如何修改物理参数
*   **全局参数**: 修改 `Assets/scene/parameterSetting.txt`。
*   **代码硬编码**: 在 `DefaultSettings()` (gl_main.cu) 或 `GIPC::init` 中修改默认值。

### 调试建议
*   **可视化**: `gl_main.cu` 提供了基础的 OpenGL 渲染，可以按 `f` 键切换 BVH 显示，按 `9` 键保存当前帧。
*   **日志**: 利用 `printf` 在 CUDA Kernel 中打印关键变量（注意 GPU 打印缓冲区的限制）。

