# AMAT5315 Week 12 · 整数规划与神经分支

[作业规格](SPEC.md) · [求解器接口](branchopt.design.toml) · [训练接口](train_branch.design.toml)

## LP 松弛与取整

车间的 LP 最优点是 `(7/3, 8/3)`，收益为 `83/3`。四舍五入得到 `(2, 3)`，需要 29 小时。向下取整得到 `(2, 2)`，收益为 22。真正的整数最优点 `(3, 2)` 收益为 27，和 LP 上界相差 `2/3`。饮食 LP 在 `(89/28, 8/7)` 达到 `323/14`。

种子 2026 生成的 16 个 32 × 24 集合覆盖实例平均整数差距为 10.49%。16 次最近取整都留下了未覆盖的行。逐行加入最便宜的覆盖列后，16 个方案都可行，成本比整数最优高 13.04%–78.26%。`artifacts/gap.png` 由保存的 LP、MILP 和修复方案绘制。

## 分支定界与 Strong Branching

车间先解根节点，得到 `83/3`。先走 `x₂ ≤ 2`，遇到整数可行解 `(3, 2)` 并将 27 记为当前最佳收益。再解 `x₂ ≥ 3` 的 LP，上界是 26.75，直接剪掉这一侧。实际 LP 求解和搜索节点均为 3。

| 策略 | 每例平均节点 | 每例平均 LP 求解 | 每例平均搜索耗时 |
| --- | ---: | ---: | ---: |
| most-frac | 44.04 | 44.04 | 8.19 ms |
| strong | 18.83 | 225.63 | 47.48 ms |

Strong Branching 将平均节点数降低 57.2%，但每个探索节点平均要解约 12 个 LP。48 个实例上的 96 次策略求解均与 HiGHS MILP 最优值一致，最大误差 `7.08 × 10⁻¹²`。标签按 36 个训练和 12 个验证实例预分配编号，实际保存 318 和 110 个有两个以上候选的节点。`artifacts/bnb.png` 汇总两组配对结果。

## JAX 模仿训练

同一个 `12 → 48 tanh → 1` MLP 分别给各候选变量打分。训练使用训练集的均值和标准差，按节点做带掩码的 softmax。并列最佳的 Strong Branching 候选平分目标概率。验证交叉熵在第 22 轮达到最低值 2.1358，之后训练交叉熵继续下降，验证交叉熵逐渐回升。保存的模型取第 22 轮权重。

独立脚本用保存的权重对 110 个验证节点重算，得到 Top-1 **31.8%** 和 Top-3 **65.5%**。在并列最佳候选都算命中的同一规则下，均匀随机选一个的平均命中率为 **18.3%**。若每个节点只承认一个正确候选，`mean(1/K)` 为 10.3%。`artifacts/training.png` 展示 120 轮曲线，模型保存了六组验证原始特征、JAX logits 和概率。

## Rust 推理与尺寸迁移

Rust 在六组探针上的最大 logits 误差为 `8.88 × 10⁻¹⁶`，最大概率误差为 `2.78 × 10⁻¹⁶`。用同一组权重评估两个规模，24 个实例、四种策略共 96 次求解均达到 HiGHS 最优值，最大差值 `8.89 × 10⁻¹¹`，低于讲义的 `10⁻⁴` 门槛。神经打分只选择合法的分支变量。LP 界负责剪枝，整数候选用原约束和目标函数复核。三个无前瞻策略各做一次 LP 求解对应一个搜索节点，Strong Branching 的额外求解与每个候选的两个试探子节点对应。

下表均为每例平均值，耗时单位为 ms。

| 规模 | 策略 | 节点 | LP 求解次数 | LP 耗时 | 特征与打分耗时 | 总搜索耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32 × 24 | most-frac | 46.75 | 46.75 | 8.37 | 0 | 8.42 |
| 32 × 24 | strong | 20.38 | 248.13 | 47.43 | 0 | 47.54 |
| 32 × 24 | neural | 26.25 | 26.25 | 4.50 | 0.256 | 4.79 |
| 32 × 24 | root-neural | 31.63 | 31.63 | 5.73 | 0.034 | 5.80 |
| 48 × 34 | most-frac | 223.75 | 223.75 | 54.58 | 0 | 54.87 |
| 48 × 34 | strong | 73.75 | 1277.75 | 352.32 | 0 | 352.88 |
| 48 × 34 | neural | 133.75 | 133.75 | 34.20 | 1.583 | 35.97 |
| 48 × 34 | root-neural | 177.25 | 177.25 | 44.58 | 0.070 | 44.89 |

在线神经分支相对 most-frac 在两组分别快约 **1.76×** 和 **1.53×**，相对 Strong Branching 分别快约 **9.94×** 和 **9.81×**。缓存根节点特征的策略减少了特征提取和打分成本，但在线打分进一步缩小了树，因此在这两组中总耗时更短。结果来自本机、给定种子和这一批实例。两组的节点及耗时图为 `artifacts/evaluation.png`。

32 × 24 中节省最多的是实例 4，从 129 个节点降至 29。most-frac 的前两次分支变量为 8、2，在线策略选 2、5。48 × 34 中对应实例 7，从 529 个节点降至 285，前两次分支从 21、38 换为 12、28。所有列编号从 0 开始。

## 复现

在 `week12/` 中运行以下命令。环境需要 Rust、C/C++ 编译器、CMake、libclang、`uv` 和 Python 3.12 或更新版本。Rust 依赖由 `Cargo.lock` 固定，HiGHS 关闭预求解并使用一个线程。基准用 ChaCha8 根据命令行的 seed 生成实例。JAX 使用 CPU 和 float64，训练 seed 为 2026。本次基准在 WSL Ubuntu 24.04 的 AMD Ryzen 7 8745H 上运行，使用 Rust 1.96.0、`highs` 2.2.0、JAX 0.12.1、NumPy 2.5.5 和 Matplotlib 3.11.2。

```bash
cargo test --locked --release
cargo install --locked --path .
branchopt --mode gap --vars 32 --rows 24 --count 16 \
  --seed 2026 --out artifacts/gap
python3 -B -m unittest discover -s tests -v

cargo install --locked --path . --force
branchopt --mode bnb --vars 32 --rows 24 --count 48 \
  --policy strong --export-labels --seed 2026 --out artifacts/bnb_strong
branchopt --mode bnb --vars 32 --rows 24 --count 48 \
  --policy most-frac --seed 2026 --out artifacts/bnb_most
python3 verify.py bnb

uv run --no-project --python 3.12 --with jax==0.12.1 python train_branch.py \
  --dataset artifacts/bnb_strong/dataset.json --out artifacts
python3 verify.py model

cargo install --locked --path . --force
branchopt --mode eval --vars 32 --rows 24 --count 16 --seed 3026 \
  --model artifacts/model.json --out artifacts/eval_test
branchopt --mode eval --vars 48 --rows 34 --count 8 --seed 4026 \
  --model artifacts/model.json --out artifacts/eval_transfer
python3 verify.py eval
uv run --no-project --python 3.12 --with matplotlib==3.11.2 python plots.py all
```

所有 `run.json`、强分支标签、模型权重和图都在 `artifacts/`。数字和图由保存的证据重算。表中的运行时间指 CPU 环境下的经过时间，求解器的 LP 时间与特征计算及神经打分时间分别记录。

开发使用原生 WSL Pi 和火山方舟 Agent Plan 的 `glm-5.3-flash`。Pi 参与 LP/MILP、训练器与 Rust 推理草稿，以及分支定界方案复核。协调端完成集成、代码修正、数值复核和图表。
