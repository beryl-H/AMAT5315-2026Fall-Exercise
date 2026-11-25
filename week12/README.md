# AMAT5315 Week 12 · 自旋玻璃与学习分支

本作业依据 16 页的修订学习单，覆盖自旋玻璃的 MILP、精确分支定界、JAX 模仿训练和 Rust CPU 迁移实验。输入、参数和验收位置见 [作业规格](SPEC.md)、[求解器设计](branchopt.design.toml)和[训练器设计](train_branch.design.toml)。

## 1. LP 松弛、基态和取整

`pred 0.6.0` 求得四自旋方形和给定 seed 的 100 自旋图的基态分别为 **−2** 和 **−138**。方形的一组基态位为 `(0,0,1,0)`，对应自旋 `(+1,+1,−1,+1)`。沿闭环，要求三个正耦合边反向和一个负耦合边同向会产生奇数次翻转，至少一条键无法满足。

用二元位 `x_i=(1-s_i)/2` 表示自旋，每条键增加连续变量 `z_ij∈[0,1]`。由四条线性不等式，`(x_i,x_j)=(0,0),(1,1)` 时强制 `z_ij=0`，而 `(0,1),(1,0)` 时强制 `z_ij=1`。目标在线性部分取系数 `−2h_i` 和 `−2J_ij`，再加回常数 `Σh_i+ΣJ_ij`。松弛后四个自旋都可取 0.5，每条键选择自己偏好的 `z`，所以方形 LP 下界为 **−4**。把位全取成 1 并重新计算键的异或，得到可行能量 **+2**，比基态高 **4**。饮食 LP 在 `(89/28,8/7)` 达到 **23.071429**。

22 自旋、52 键、seed 2026 的 16 个图中，每个 LP 下界均为 **-52**，每个都有 **22** 个半整数自旋。平均基态 **-38.25**，平均整数差距 **13.75**，逐图相对差距的平均值为 **36.45%**。最近取整在 **16/16** 个图上未达到基态，平均多出 **34.25**，实例 11 多出 **60**。图见 `artifacts/gap.png`。

## 2. 精确分支定界

用栈执行深度优先搜索，先访问接近松弛解的一支，0.5 时先固定为 1。已验证的整数解提供上界，HiGHS LP 提供子树下界。只在未固定的分数自旋上分支，并同时保留其 0 和 1 两个分支。

| 四面体节点 | 新固定的位 | LP 下界 | 动作 |
| ---: | --- | ---: | --- |
| 1 | 根节点 | −6 | 分裂自旋 0 |
| 2 | `x₀=1` | −4 | 分裂自旋 1 |
| 3 | `x₁=1` | −3 | 得到基态 `(1,1,0,0)` |
| 4 | `x₁=0` | −3 | 下界等于已知最优，剪枝 |
| 5 | `x₀=0` | −2 | 下界高于已知最优，剪去仍有三个半整数位的子树 |

每个候选的两个试探 LP 给出下界增益，Strong Branching 用两项增益的乘积偏好两侧都显著收紧的分裂。试探不更新已知整数解。48 个相同图的每例平均值如下。

| 策略 | 节点 | LP 次数 | 搜索时间（ms） |
| --- | ---: | ---: | ---: |
| most-frac | 50.88 | 50.88 | 23.77 |
| strong | 16.00 | 287.42 | 123.00 |

Strong Branching 平均每例少走约 69% 的节点，但解约 5.65 倍的 LP。96 次求解对照 HiGHS MILP 的最大绝对误差为 **1.62e-12**。图见 `artifacts/bnb.png`，每条曲线按该策略的实例计数独立排序。

## 3. 模仿训练与 Rust 推理

训练集包含 **276** 个节点、36 个图，验证集包含 **84** 个节点、12 个图，两集合按图划分，重叠为零。候选用相同的 12 项局部图及 LP 特征和同一组 `12 → 48 tanh → 1` 权重。只用训练候选估计标准化参数，节点内 softmax 遮住填充位置，并列教师最优候选平分目标概率。初始权重采用 Xavier，批次使用 seed 2026 打乱。120 轮中按讲义只在第 1 轮和每五轮考察保存点，第 **85** 轮的验证交叉熵最低，为 **1.7533**。验证 Top-1 为 **72.6%**，同一并列规则下随机 Top-1 为 **24.5%**。CPU 训练耗时 **1.77 秒**。

六组未用于训练的验证探针在 Rust 和 JAX 之间的最大 logits 与概率差分别为 **2.66e-15** 和 **3.61e-16**，低于 `1e-4`。神经打分只选择合法的分支自旋，剪枝仍由 LP 界和已核验的整数解决定。16 个未见图中，实例 **4** 的基态能量为 **-38**，在线神经分支的节点节省最多。most-frac 首先分裂自旋 **0, 1**，用了 **157** 个节点和 **58.43 ms**。在线网络先选 **4, 21**，用了 **21** 个节点和 **8.77 ms**。自旋编号从 0 开始。损失曲线和排序后的节点数见 `artifacts/neural_bnb.png`。

## 4. CPU 耗时与尺寸迁移

同一份权重用于两个规模，root-neural 只在根节点取特征并给所有自旋打分，其余节点用缓存标准分与当前分数性的 3 倍相加排序。在线 neural 对每个实际分裂节点重新取特征和打分。LP 时间含模型装配和 HiGHS 求解，特征与推理时间含一次邻接表构建、特征提取和 MLP 推理。根缓存的标准化计入推理，其后读取缓存和排序计入其余搜索工作。各时间取单批顺序运行的平均 wall time，搜索计时排除独立 HiGHS MILP 对照。

| 自旋 / 键 | 策略 | 平均节点 | 平均 LP 次数 | 总时间（ms） | LP（ms） | 特征与推理（ms） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 22 / 52 | most-frac | 50.88 | 50.88 | 20.722 | 20.672 | 0.000 |
| 22 / 52 | strong | 15.62 | 287.75 | 121.361 | 121.259 | 0.000 |
| 22 / 52 | neural | 15.25 | 15.25 | 6.395 | 6.209 | 0.170 |
| 22 / 52 | root-neural | 15.50 | 15.50 | 6.295 | 6.249 | 0.029 |
| 30 / 72 | most-frac | 177.75 | 177.75 | 120.600 | 120.349 | 0.000 |
| 30 / 72 | strong | 38.00 | 899.25 | 553.922 | 553.576 | 0.000 |
| 30 / 72 | neural | 39.75 | 39.75 | 24.550 | 23.961 | 0.542 |
| 30 / 72 | root-neural | 50.75 | 50.75 | 29.396 | 29.297 | 0.040 |

两组共 **96 次**求解均匹配 HiGHS，最大绝对误差 **2.01e-12**。三个无前瞻策略的 LP 次数均等于节点数，Strong Branching 的额外 LP 数等于探测候选数的两倍。在 30 自旋图上，在线策略相对于 most-frac 少 77.6% 的节点，相对 Strong Branching 快 22.6 倍。根缓存策略相对于 most-frac 少 71.4% 的节点，相对 Strong Branching 快 18.8 倍。根缓存显著压低了打分开销，在线策略搜索树更小。两组节点及大图耗时分解见 `artifacts/eval.png`。

## 复现

从 `week12/` 运行。实测环境为 WSL Ubuntu 24.04、AMD Ryzen 7 8745H、Rust/Cargo 1.96.0、`highs` 2.2.0、`pred` 0.6.0、Python 3.12、JAX 0.12.1、NumPy 2.5.5 和 Matplotlib 3.11.2。构建需要 C/C++ 编译器、CMake、libclang 和 `uv`。HiGHS 单线程并关闭 presolve，MILP 绝对和相对 gaps 均设为零。

```bash
# 如需安装 pred
cargo install problemreductions-cli@0.6.0 --locked
mkdir -p artifacts/pred
(
  cd artifacts/pred
  pred create SpinGlass --graph 0-1,1-2,2-3,3-0 --couplings=1,1,1,-1 -o square.json
  time pred solve square.json > square.solution.json
  pred create SpinGlass --random --num-vertices 100 --edge-prob 0.04 --seed 2026 -o r100.json
  time pred solve r100.json > r100.solution.json
)

cargo test --locked --release
cargo install --locked --path . --force
branchopt --mode gap --spins 22 --edges 52 --count 16 --seed 2026 --out artifacts/gap
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy most-frac --seed 2026 --out artifacts/bnb_most
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy strong --seed 2026 --out artifacts/bnb_strong
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy strong --export-labels --seed 2026 --out artifacts/bnb_strong

uv run --no-project --python 3.12 --with jax==0.12.1 python train_branch.py \
  --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json
branchopt --mode eval --spins 22 --edges 52 --count 16 --seed 3026 \
  --model artifacts/model.json --out artifacts/eval_test
branchopt --mode eval --spins 30 --edges 72 --count 8 --seed 4026 \
  --model artifacts/model.json --out artifacts/eval_transfer

python3 -B -m unittest discover -s tests -v
python3 -B verify.py pred
python3 -B verify.py bnb
python3 -B verify.py eval
uv run --no-project --python 3.12 --with matplotlib==3.11.2 python plots.py all
```

`pred 0.6.0` 可能在打印结果前报告已知的规则枚举线程警告。本次两次命令均退出为零，能量也按原始图和返回位复算。四张图、各轮 `run.json`、标签、权重、探针及 `pred` 文件在 Git 忽略的 `artifacts/` 下。WSL 原生 Pi 使用课程既有模型协助复核算法规格、图特征和 JAX 掩码方案，Codex 整合源码并独立执行上述数值检查。
