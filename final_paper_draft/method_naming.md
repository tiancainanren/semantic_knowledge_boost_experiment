# 最终方法命名

## 1. 总体框架名

**SAGE++**: **S**emantic **A**daptive **G**raph **E**nhancement++

这个名字作为论文里的总方法名，用来覆盖从 TAC/KEC 出发的整条语义增强路线。

## 2. 核心组件命名

- **SAGE-A**: confidence-aware adaptive fusion
- **SAGE-AG**: adaptive fusion + graph diffusion refinement
- **SAGE-AGP**: adaptive fusion + graph diffusion + prototype calibration
- **SAGE-DKP**: dual-knowledge preservation

其中：
- `SAGE-AG` 适合作为通用主方法；
- `SAGE-AGP` 适合作为细粒度增强版；
- `SAGE-DKP` 作为第二轮增强模块，主要用于细粒度或语义分歧较大的数据集。

## 3. 论文中推荐的表述方式

建议在正文里这样写：

1. **方法框架名**：`SAGE++`
2. **主结果方法**：`SAGE++ (best variant in a fixed SAGE search space)`
3. **核心通用版本**：`SAGE-AG`
4. **细粒度版本**：`SAGE-AGP`
5. **第二轮增强版本**：`SAGE-DKP`

这样做的好处是：
- 保留一个统一的论文主名字；
- 又能在消融中清楚区分不同模块；
- 也能解释为什么 `StanfordDogs` 最终受益于 dual-knowledge preservation，而其他数据集未必需要它。