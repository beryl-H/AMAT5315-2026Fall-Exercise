# AMAT5315 Week 5：自动微分、Born modelling 与 checkpointed RTM

已完成讲义的四部分计算练习。原有 Part 1 经过重新核验；Part 2–4 使用 Rust 声波传播器、真正的 Enzyme timestep JVP/VJP，以及讲义的 Treeverse 调度。讲义输入保持原样，所有 `.npy` 数组留在本地。

## 可复现运行

在 Ubuntu 24.04 的仓库 `week5/` 中运行：

```bash
bash scripts/setup.sh
bash scripts/run_all.sh
```

`setup.sh` 将 Rust 安装到忽略的 `inputs/toolchain/`，不会替换系统 Rust。版本固定为 `nightly-2026-09-05`，包含 `enzyme`、`rustfmt`；`Cargo.lock` 锁定 Rust 依赖。Python 3.12 的依赖由 `uv==0.12.23` 与 `uv.lock` 锁定，项目环境为 `.venv/`，主要版本为 JAX/JAXlib 0.11.2、NumPy 2.5.3、matplotlib 3.11.2、pytest 9.1.1。

在自己的交互终端中先进入 `week5/`，执行 `source scripts/env.sh`，即可使用 `seismic --help`、`cargo` 和项目的 `python`。环境设置仅影响 source 它的 shell。

`download_inputs.py` 从课程的 [week5-inputs.zip](https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip) 下载缺失输入，并检查确切文件字节的 SHA-256；已有输入不一致时停止并保留原文件。

| 本地输入 | SHA-256 |
| --- | --- |
| `inputs/reflector.json` | `22b84948e28e8e13e958f47b2d4bf9a70a3d0c8ee6ffcaab3cdeb8003b9902a6` |
| `inputs/marmousi.json` | `67beabdee96d8e3f6433c2f897b3b24b0c744e7808784148bb06310665d888b0` |

`MARMOUSI-LICENSE` 随源码保留。来源、原始模型 commit、MAT 文件 SHA-256、插值与平滑过程由输入文件的 provenance 及每次运行的 `run.json` 记录。

## 数值结果

所有量都来自实际输出数组，详见 [verification.json](artifacts/verification.json) 与 [verification.txt](evidence/verification.txt)。

| 检查 | 实测结果 | 讲义／独立检验要求 |
| --- | --- | --- |
| LJ：601 个点的 forward / reverse 最大绝对误差 | 均为 `2.84217e-14` | `< 1e-12` |
| LJ：中心有限差分，h=1e-6 | 最大绝对误差 `4.65604e-9` | 对照解析导数 |
| Reflector forward gather L2 | `11.574769503614` | `11.574770`，相对容差 `1e-4` |
| 三炮最大绝对峰值 | `0.60809514 / 0.59271397 / 0.60809514`，trace index 83 | 峰值出现在离震源一格的 receiver |
| 对角线波峰位移：step 108 → 144 | `1.272792206136 km` | `1.296 km ± 0.141421 km` |
| Born 数据的 d·d | `0.03484789021516378` | 约 `0.0348479` |
| m·(Jᵀd) | `0.03484789021516377` | 与 d·d 的相对误差 `< 1e-9` |
| 点积相对误差 | `3.98239e-16`（末位可能随平台舍入变化） | 独立随机权重检查也通过 |
| Reflector 深度 profile 峰值 | z=21，即 `2.1 km` | 与真实中心相差最多一格 |
| Marmousi raw image L2 | `6.703774060378e-4` | `6.7037741e-4`，相对容差 `1e-4` |
| Marmousi 保存的重启状态 | `6`，`20,788,320 bytes` | 初始状态 + 5 个额外槽 |

下表的 work 是每炮由调度器执行的 primal forward steps，包含重放，排除 VJP 内部工作。字节数只计算保存的完整状态。

| 额外槽 δ | Forward steps / shot | Peak saved states | Peak saved bytes | 与 full-history 图像的相对 L2 误差 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 28,680 | 2 | 53,792 | 0，逐位相同 |
| 3 | 1,695 | 4 | 107,584 | 0，逐位相同 |
| 5 | 990 | 6 | 161,376 | 0，逐位相同 |
| 10 | 642 | 11 | 295,856 | 0，逐位相同 |
| 完整历史 | 240 | 241 | 6,481,936 | 基准 |

Marmousi δ=5 的每炮 forward steps 为 7,884，reverse steps 为 1,200。实际从未运行其 full-history；若保存 1,201 个状态，理论上需要 `4,161,128,720 bytes`。上述 saved bytes 不含工作区、adjoint、model、receiver data、操作日志或 Enzyme scratch，不能解释为整个进程的 RSS。

21 份 action 文件均被 `audit_actions.py` 独立重放审计。讲义要求的缺失 reverse、重复 reverse、逆序错误、非法 restore、预算超限五项计数全部为零；另检查 grad 输入已保存、call/workspace 步号、store/fetch 有效性、实际槽计数以及结束时只保留状态 0。

## 如何向老师解释证据

1. **完整重启状态**：`s_n = (u_{n-1}, u_n)`，两个 float64 `[z,x]` 场连续放在长度 `2*nx*nz` 的数组中。波动方程是二阶时间更新，单独保存 `u_n` 不足以重启。给定固定 model、sponge、source 参数和步号 n，源码可重新生成 `q_n`。
2. **局部 Enzyme 调用**：`src/kernel.rs` 的 `step_forward` / `step_reverse` 是编译器对同一个 timestep stencil 生成的 JVP/VJP。state 和 velocity 为活动输入；sigma、source、dt、dx、网格大小为常量。`src/simulation.rs` 组合这些局部导数；没有手写生产用的波场导数。
3. **重放与图像**：`src/schedule.rs` 的 `call` 仅调用 primal `advance` 并更新独立 workspace；不会更新 adjoint 或 image。`grad` 只使用 saved slot 的输入状态，按 `N-1,...,0` 各执行一次。图像逐位一致、动作审计与实际 live-slot 断言共同支持这一点。
4. **两种独立导数证据**：点积检验测试 J 与 Jᵀ 的转置关系，但二者若共同缩放错误也可能通过。`tests/test_derivatives.py` 另外使用随机 velocity 方向的中心有限差分（相对误差 `<1e-7`），以及独立随机 receiver 权重、重复 receiver、多炮和零边界 adjoint，避免只检验 d=Jm 的单个方向。

离散 stencil 为：

```text
u_next = [2*u_current - (1-sigma*dt)*u_previous
          + dt²*(c²*five_point_Laplacian(u_current) + q_n)] / (1+sigma*dt)
```

边界固定为零，receiver 在更新后采样，perturbation 是 velocity `m=δc`。因此 `δ(c²)=2cδc`，不能把它误当成 squared slowness。内积是不加 dx、dt 权重的 Euclidean sum。

Reflector 的空间单位是 100 m、时间单位是 0.1 s：dx=1 对应 100 m，dt=0.2 对应 0.02 s，c=1.8 对应 1.8 km/s。图中横轴与深度使用 km，时间使用 s，深度／时间向下增加。Forward step 144 的场在 2.88 s；reverse step 132 之后得到输入状态 s_132 的 adjoint，记录时间为 2.64 s。第一炮的 `wavefield.npy` 为 float32 展示记录，计算及保存的重启状态仍为 float64。

原始迁移图像是 `JᵀJm`，不是求解 `(JᵀJ)^{-1}Jᵀd`。有限频带、有限孔径、照明强度和离散传播都会改变其幅值与分辨率，因此正确导数也不能保证重建图像等于 m。Reflector 的 side lobes 不代表新反射层；Marmousi 的 8–14 km、上部 3 km 区域可辨认倾斜条带，而深部在统一幅度色标下较弱。图中没有 depth gain、归一化补偿或后处理反演。

## 验证与已处理问题

最终 release build、`cargo fmt --check`、全部 **37** 项 pytest 和完整 evidence verification 通过；shell 脚本经 `bash -n` 检查。Enzyme smoke 给出 `[8,12,12]`。去掉 `-Zautodiff=Enable` 的独立编译负面对照按预期失败。

- 开发过程中先保存讲义规格与失败测试，再实现 forward、derivative、Treeverse 与独立审计。历史失败日志有明确标注，并不代表最终状态失败。
- 最初直接递归实现的恢复基点行为导致各预算都重算 28,680 步。改为保存 Visit/Loop 局部参数的显式调用帧后，讲义分割、动作顺序、保存预算、work 与图像均通过。进一步保留导出 `schedule` 入口的纯 Rust probe 在 opt-level 0 得到 8 步、opt-level 3 得到错误的 15 步，普通 rustc 驱动的 LTO 构建也复现；去掉该入口的早期 probe 两档都正确。由此证实了该探针的优化相关运行差异，具体编译阶段及原始执行器故障的完整因果关系仍未定位。参见 [诊断记录](docs/compiler-optimization.md)。
- 复核修正了极端 Ricker source 的 `inf*0`、中间幅值溢出和过早丢弃可表示的放大尾部，并保留三个先失败后通过的回归用例。修正后重新运行全部数值实验。
- 禁止在带有旧 `wavefield.npy` 的目录中关闭 recording 后重跑，以免 run.json 指向错误的旧帧；程序在写出前明确拒绝该情况。可选择新输出目录。
- Checkpoint 测试自己生成 Born 与 full-history 基准，不依赖未提交的本地 `.npy`。最终全流程曾检出新源码与旧产物比较的 fixture 问题；修正后在同版本数组上保持逐位相等的严格检查。
- JAX 使用 CPU。日志中的 CUDA 提示是可选 GPU 后端缺失提示；本练习数值检查不依赖 GPU。

每幅 PNG 已打开检查坐标、单位、色标、裁切和讲义要求的特征；修正过一处 depth-profile 标题裁切。`wavefield.png` 是根据明确帧索引导出的静态查看结果。

详细文件变更、执行命令、失败和纠正措施见 [execution-log.md](docs/execution-log.md)。Week 6 的云账号注册与支付验证需要本人按老师解释的费用办理；本次没有创建云资源或 GPU 租赁。讲义说明 Week 5 无需另行提交，下次课堂展示证据。

## 核心文件

| 文件 | 用途 |
| --- | --- |
| `SPEC.md`, `lj_ad.py`, `test_lj_ad.py`, `test_modes.py`, `test_graphs.py` | 原有 Part 1 规格、代码与测试 |
| `docs/seismic-spec.md`, `seismic.design.toml` | 讲义数值约定、CLI 与输出格式 |
| `docs/implementation-plan.md` | 完成检查清单 |
| `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `build.rs` | 构建与固定依赖；只对无 std 的 kernel 启用 Enzyme |
| `src/kernel.rs`, `src/lib.rs`, `src/main.rs`, `src/npy.rs` | stencil、模型验证、CLI、NPY I/O |
| `src/simulation.rs`, `src/schedule.rs` | 多炮 JVP/VJP、full history、Treeverse executor |
| `tests/test_seismic.py`, `tests/test_derivatives.py`, `tests/test_checkpoint.py`, `tests/test_action_audit.py` | 独立数值、数据契约与 action mutation 测试 |
| `pyproject.toml`, `uv.lock`, `.gitignore`, `.gitattributes` | Python 固定版本、本地数组／构建排除与原始日志空白保留 |
| `scripts/{env,setup,run_all}.sh`, `scripts/download_inputs.py` | 本地环境、输入校验与全流程 |
| `scripts/{run_ad,compare_modes,plot_graphs,plot_seismic,verify_evidence,audit_actions,write_inventory}.py` | 导数、图像、独立核验与库存生成 |
| `docs/compiler-probe.rs`, `docs/compiler-private-probe.rs`, `docs/compiler-optimization.md` | 可重现的导出／非导出入口诊断及证据边界 |
| `docs/自动微分、伴随成像与检查点.md` | 中文概念学习笔记、单位与证据边界 |
| `MARMOUSI-LICENSE` | 输入模型许可 |

## Evidence inventory

下面逐一列出每个 evidence 文件及产生它的命令／脚本。`.npy` 标为 local，Git 忽略；PNG、JSON、文本日志均小于 5 MB。历史 red-test / mutation / 递归诊断日志需对应当时源码状态，当前实现不会重新产生这些预期失败；Git 历史及 execution log 保留阶段背景。库存由 `python scripts/write_inventory.py` 更新。

<!-- GENERATED EVIDENCE INVENTORY -->

| 运行代号 | 完整命令 |
| --- | --- |
| `adjoint` | `target/release/seismic --data artifacts/born/born_data.npy --every 3 --experiment inputs/reflector.json --mode adjoint --out artifacts/adjoint --storage full` |
| `born` | `target/release/seismic --experiment inputs/reflector.json --mode born --out artifacts/born` |
| `checkpoint-1` | `target/release/seismic --checkpoints 1 --data artifacts/born/born_data.npy --experiment inputs/reflector.json --mode adjoint --out artifacts/checkpoint-1 --storage treeverse` |
| `checkpoint-10` | `target/release/seismic --checkpoints 10 --data artifacts/born/born_data.npy --experiment inputs/reflector.json --mode adjoint --out artifacts/checkpoint-10 --storage treeverse` |
| `checkpoint-3` | `target/release/seismic --checkpoints 3 --data artifacts/born/born_data.npy --experiment inputs/reflector.json --mode adjoint --out artifacts/checkpoint-3 --storage treeverse` |
| `checkpoint-5` | `target/release/seismic --checkpoints 5 --data artifacts/born/born_data.npy --experiment inputs/reflector.json --mode adjoint --out artifacts/checkpoint-5 --storage treeverse` |
| `forward` | `target/release/seismic --every 3 --experiment inputs/reflector.json --mode forward --out artifacts/forward` |
| `marmousi-born` | `target/release/seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born` |
| `marmousi-image` | `target/release/seismic --checkpoints 5 --data artifacts/marmousi-born/born_data.npy --experiment inputs/marmousi.json --mode adjoint --out artifacts/marmousi-image --storage treeverse` |

| 文件 | 产生它的命令／脚本 | 支持的量／状态 | 字节 |
| --- | --- | --- | ---: |
| [artifacts/ad/derivatives.json](artifacts/ad/derivatives.json) | python scripts/run_ad.py | LJ 导数或 JAX primitive graph | 353 |
| [artifacts/ad/grad-graph.png](artifacts/ad/grad-graph.png) | python scripts/plot_graphs.py | LJ 导数或 JAX primitive graph | 189,021 |
| [artifacts/ad/graph.png](artifacts/ad/graph.png) | python scripts/plot_graphs.py | LJ 导数或 JAX primitive graph | 59,496 |
| [artifacts/ad/modes.png](artifacts/ad/modes.png) | python scripts/compare_modes.py | LJ 导数或 JAX primitive graph | 115,939 |
| [artifacts/adjoint/image.npy](artifacts/adjoint/image.npy) | 上表 `adjoint` 命令 | local / ignored：原始数值数组 | 13,576 |
| [artifacts/adjoint/image.png](artifacts/adjoint/image.png) | python scripts/plot_seismic.py | 按真实数组、步号、单位绘制的展示证据 | 162,762 |
| [artifacts/adjoint/result.json](artifacts/adjoint/result.json) | 上表 `adjoint` 命令 | 模式、规模、耗时、actual counters | 900 |
| [artifacts/adjoint/run.json](artifacts/adjoint/run.json) | 上表 `adjoint` 命令 | 参数、源文件 fingerprint、recording 语义 | 3,977 |
| [artifacts/adjoint/wavefield.npy](artifacts/adjoint/wavefield.npy) | 上表 `adjoint` 命令 | local / ignored：原始数值数组 | 538,048 |
| [artifacts/adjoint/wavefield.png](artifacts/adjoint/wavefield.png) | python scripts/plot_seismic.py | 按真实数组、步号、单位绘制的展示证据 | 87,216 |
| [artifacts/born/born_data.npy](artifacts/born/born_data.npy) | 上表 `born` 命令 | local / ignored：原始数值数组 | 80,768 |
| [artifacts/born/result.json](artifacts/born/result.json) | 上表 `born` 命令 | 模式、规模、耗时、actual counters | 281 |
| [artifacts/born/run.json](artifacts/born/run.json) | 上表 `born` 命令 | 参数、源文件 fingerprint、recording 语义 | 1,592 |
| [artifacts/checkpoint-1/actions-0.json](artifacts/checkpoint-1/actions-0.json) | 上表 `checkpoint-1` 命令 | 动作顺序与 live saved slots | 1,342,399 |
| [artifacts/checkpoint-1/actions-1.json](artifacts/checkpoint-1/actions-1.json) | 上表 `checkpoint-1` 命令 | 动作顺序与 live saved slots | 1,342,399 |
| [artifacts/checkpoint-1/actions-2.json](artifacts/checkpoint-1/actions-2.json) | 上表 `checkpoint-1` 命令 | 动作顺序与 live saved slots | 1,342,399 |
| [artifacts/checkpoint-1/image.npy](artifacts/checkpoint-1/image.npy) | 上表 `checkpoint-1` 命令 | local / ignored：原始数值数组 | 13,576 |
| [artifacts/checkpoint-1/result.json](artifacts/checkpoint-1/result.json) | 上表 `checkpoint-1` 命令 | 模式、规模、耗时、actual counters | 1,026 |
| [artifacts/checkpoint-1/run.json](artifacts/checkpoint-1/run.json) | 上表 `checkpoint-1` 命令 | 参数、源文件 fingerprint、recording 语义 | 1,705 |
| [artifacts/checkpoint-10/actions-0.json](artifacts/checkpoint-10/actions-0.json) | 上表 `checkpoint-10` 命令 | 动作顺序与 live saved slots | 74,214 |
| [artifacts/checkpoint-10/actions-1.json](artifacts/checkpoint-10/actions-1.json) | 上表 `checkpoint-10` 命令 | 动作顺序与 live saved slots | 74,214 |
| [artifacts/checkpoint-10/actions-2.json](artifacts/checkpoint-10/actions-2.json) | 上表 `checkpoint-10` 命令 | 动作顺序与 live saved slots | 74,214 |
| [artifacts/checkpoint-10/image.npy](artifacts/checkpoint-10/image.npy) | 上表 `checkpoint-10` 命令 | local / ignored：原始数值数组 | 13,576 |
| [artifacts/checkpoint-10/result.json](artifacts/checkpoint-10/result.json) | 上表 `checkpoint-10` 命令 | 模式、规模、耗时、actual counters | 1,025 |
| [artifacts/checkpoint-10/run.json](artifacts/checkpoint-10/run.json) | 上表 `checkpoint-10` 命令 | 参数、源文件 fingerprint、recording 语义 | 1,707 |
| [artifacts/checkpoint-3/actions-0.json](artifacts/checkpoint-3/actions-0.json) | 上表 `checkpoint-3` 命令 | 动作顺序与 live saved slots | 121,840 |
| [artifacts/checkpoint-3/actions-1.json](artifacts/checkpoint-3/actions-1.json) | 上表 `checkpoint-3` 命令 | 动作顺序与 live saved slots | 121,840 |
| [artifacts/checkpoint-3/actions-2.json](artifacts/checkpoint-3/actions-2.json) | 上表 `checkpoint-3` 命令 | 动作顺序与 live saved slots | 121,840 |
| [artifacts/checkpoint-3/image.npy](artifacts/checkpoint-3/image.npy) | 上表 `checkpoint-3` 命令 | local / ignored：原始数值数组 | 13,576 |
| [artifacts/checkpoint-3/result.json](artifacts/checkpoint-3/result.json) | 上表 `checkpoint-3` 命令 | 模式、规模、耗时、actual counters | 1,023 |
| [artifacts/checkpoint-3/run.json](artifacts/checkpoint-3/run.json) | 上表 `checkpoint-3` 命令 | 参数、源文件 fingerprint、recording 语义 | 1,705 |
| [artifacts/checkpoint-5/actions-0.json](artifacts/checkpoint-5/actions-0.json) | 上表 `checkpoint-5` 命令 | 动作顺序与 live saved slots | 89,821 |
| [artifacts/checkpoint-5/actions-1.json](artifacts/checkpoint-5/actions-1.json) | 上表 `checkpoint-5` 命令 | 动作顺序与 live saved slots | 89,821 |
| [artifacts/checkpoint-5/actions-2.json](artifacts/checkpoint-5/actions-2.json) | 上表 `checkpoint-5` 命令 | 动作顺序与 live saved slots | 89,821 |
| [artifacts/checkpoint-5/image.npy](artifacts/checkpoint-5/image.npy) | 上表 `checkpoint-5` 命令 | local / ignored：原始数值数组 | 13,576 |
| [artifacts/checkpoint-5/result.json](artifacts/checkpoint-5/result.json) | 上表 `checkpoint-5` 命令 | 模式、规模、耗时、actual counters | 1,020 |
| [artifacts/checkpoint-5/run.json](artifacts/checkpoint-5/run.json) | 上表 `checkpoint-5` 命令 | 参数、源文件 fingerprint、recording 语义 | 1,705 |
| [artifacts/checkpoint-actions.png](artifacts/checkpoint-actions.png) | python scripts/plot_seismic.py | 按真实数组、步号、单位绘制的展示证据 | 116,437 |
| [artifacts/checkpoint-work.png](artifacts/checkpoint-work.png) | python scripts/plot_seismic.py | 按真实数组、步号、单位绘制的展示证据 | 133,871 |
| [artifacts/forward/gathers.png](artifacts/forward/gathers.png) | python scripts/plot_seismic.py | 按真实数组、步号、单位绘制的展示证据 | 146,969 |
| [artifacts/forward/result.json](artifacts/forward/result.json) | 上表 `forward` 命令 | 模式、规模、耗时、actual counters | 284 |
| [artifacts/forward/run.json](artifacts/forward/run.json) | 上表 `forward` 命令 | 参数、源文件 fingerprint、recording 语义 | 3,909 |
| [artifacts/forward/traces.npy](artifacts/forward/traces.npy) | 上表 `forward` 命令 | local / ignored：原始数值数组 | 80,768 |
| [artifacts/forward/wave-speed.png](artifacts/forward/wave-speed.png) | python scripts/plot_seismic.py | 按真实数组、步号、单位绘制的展示证据 | 120,029 |
| [artifacts/forward/wavefield.npy](artifacts/forward/wavefield.npy) | 上表 `forward` 命令 | local / ignored：原始数值数组 | 538,048 |
| [artifacts/forward/wavefield.png](artifacts/forward/wavefield.png) | python scripts/plot_seismic.py | 按真实数组、步号、单位绘制的展示证据 | 85,438 |
| [artifacts/inputs.png](artifacts/inputs.png) | python scripts/plot_seismic.py | 按真实数组、步号、单位绘制的展示证据 | 126,970 |
| [artifacts/marmousi-born/born_data.npy](artifacts/marmousi-born/born_data.npy) | 上表 `marmousi-born` 命令 | local / ignored：原始数值数组 | 7,862,528 |
| [artifacts/marmousi-born/result.json](artifacts/marmousi-born/result.json) | 上表 `marmousi-born` 命令 | 模式、规模、耗时、actual counters | 286 |
| [artifacts/marmousi-born/run.json](artifacts/marmousi-born/run.json) | 上表 `marmousi-born` 命令 | 参数、源文件 fingerprint、recording 语义 | 5,709 |
| [artifacts/marmousi-image/actions-0.json](artifacts/marmousi-image/actions-0.json) | 上表 `marmousi-image` 命令 | 动作顺序与 live saved slots | 589,982 |
| [artifacts/marmousi-image/actions-1.json](artifacts/marmousi-image/actions-1.json) | 上表 `marmousi-image` 命令 | 动作顺序与 live saved slots | 589,982 |
| [artifacts/marmousi-image/actions-2.json](artifacts/marmousi-image/actions-2.json) | 上表 `marmousi-image` 命令 | 动作顺序与 live saved slots | 589,982 |
| [artifacts/marmousi-image/actions-3.json](artifacts/marmousi-image/actions-3.json) | 上表 `marmousi-image` 命令 | 动作顺序与 live saved slots | 589,982 |
| [artifacts/marmousi-image/actions-4.json](artifacts/marmousi-image/actions-4.json) | 上表 `marmousi-image` 命令 | 动作顺序与 live saved slots | 589,982 |
| [artifacts/marmousi-image/actions-5.json](artifacts/marmousi-image/actions-5.json) | 上表 `marmousi-image` 命令 | 动作顺序与 live saved slots | 589,982 |
| [artifacts/marmousi-image/actions-6.json](artifacts/marmousi-image/actions-6.json) | 上表 `marmousi-image` 命令 | 动作顺序与 live saved slots | 589,982 |
| [artifacts/marmousi-image/actions-7.json](artifacts/marmousi-image/actions-7.json) | 上表 `marmousi-image` 命令 | 动作顺序与 live saved slots | 589,982 |
| [artifacts/marmousi-image/actions-8.json](artifacts/marmousi-image/actions-8.json) | 上表 `marmousi-image` 命令 | 动作顺序与 live saved slots | 589,982 |
| [artifacts/marmousi-image/image.npy](artifacts/marmousi-image/image.npy) | 上表 `marmousi-image` 命令 | local / ignored：原始数值数组 | 1,732,488 |
| [artifacts/marmousi-image/result.json](artifacts/marmousi-image/result.json) | 上表 `marmousi-image` 命令 | 模式、规模、耗时、actual counters | 2,011 |
| [artifacts/marmousi-image/run.json](artifacts/marmousi-image/run.json) | 上表 `marmousi-image` 命令 | 参数、源文件 fingerprint、recording 语义 | 5,824 |
| [artifacts/marmousi.png](artifacts/marmousi.png) | python scripts/plot_seismic.py | 按真实数组、步号、单位绘制的展示证据 | 610,531 |
| [artifacts/verification.json](artifacts/verification.json) | python scripts/verify_evidence.py | 机器可读的独立核验结果 | 25,268 |
| [evidence/adjoint-run.txt](evidence/adjoint-run.txt) | 上表 `adjoint` 命令 | 逐炮 L2 输出 | 106 |
| [evidence/baseline-part1.txt](evidence/baseline-part1.txt) | python -m pytest -q test_lj_ad.py test_modes.py test_graphs.py | 原有 13 项测试基线 | 99 |
| [evidence/born-run.txt](evidence/born-run.txt) | 上表 `born` 命令 | 逐炮 L2 输出 | 97 |
| [evidence/build-checkpoint-barrier.txt](evidence/build-checkpoint-barrier.txt) | cargo build --release --locked | 该文件名对应阶段的构建输出（最终见 build-final.txt） | 139 |
| [evidence/build-checkpoint-fix.txt](evidence/build-checkpoint-fix.txt) | cargo build --release --locked | 该文件名对应阶段的构建输出（最终见 build-final.txt） | 139 |
| [evidence/build-checkpoint-frames.txt](evidence/build-checkpoint-frames.txt) | cargo build --release --locked | 该文件名对应阶段的构建输出（最终见 build-final.txt） | 139 |
| [evidence/build-checkpoint.txt](evidence/build-checkpoint.txt) | cargo build --release --locked | 该文件名对应阶段的构建输出（最终见 build-final.txt） | 139 |
| [evidence/build-derivatives.txt](evidence/build-derivatives.txt) | cargo build --release --locked | 该文件名对应阶段的构建输出（最终见 build-final.txt） | 139 |
| [evidence/build-final.txt](evidence/build-final.txt) | cargo build --release --locked | 该文件名对应阶段的构建输出（最终见 build-final.txt） | 62 |
| [evidence/build-forward.txt](evidence/build-forward.txt) | cargo build --release --locked | 该文件名对应阶段的构建输出（最终见 build-final.txt） | 1,589 |
| [evidence/build-ricker-fix.txt](evidence/build-ricker-fix.txt) | cargo build --release --locked | 该文件名对应阶段的构建输出（最终见 build-final.txt） | 139 |
| [evidence/checkpoint-1-run.txt](evidence/checkpoint-1-run.txt) | 上表 `checkpoint-1` 命令 | 逐炮 L2 输出 | 106 |
| [evidence/checkpoint-10-run.txt](evidence/checkpoint-10-run.txt) | 上表 `checkpoint-10` 命令 | 逐炮 L2 输出 | 106 |
| [evidence/checkpoint-3-run.txt](evidence/checkpoint-3-run.txt) | 上表 `checkpoint-3` 命令 | 逐炮 L2 输出 | 106 |
| [evidence/checkpoint-5-run.txt](evidence/checkpoint-5-run.txt) | 上表 `checkpoint-5` 命令 | 逐炮 L2 输出 | 106 |
| [evidence/checkpoint-barrier-failure.txt](evidence/checkpoint-barrier-failure.txt) | python -m pytest -q tests/test_checkpoint.py | 历史失败：scalar barrier 尝试未纠正 work | 5,032 |
| [evidence/checkpoint-debug-trace.txt](evidence/checkpoint-debug-trace.txt) | seismic --experiment inputs/debug-schedule/model.json --mode adjoint --data inputs/debug-schedule/weights.npy --storage treeverse --checkpoints 2 | 历史诊断：加入口参数打印后得到正确恢复基点；不是最终程序 | 200 |
| [evidence/checkpoint-recompute-failure.txt](evidence/checkpoint-recompute-failure.txt) | python -m pytest -q tests/test_checkpoint.py | 历史失败：直接递归重算 work 不符 | 5,028 |
| [evidence/compiler-optimization-probe.txt](evidence/compiler-optimization-probe.txt) | rustc --edition=2024 -C opt-level=0/3 -C panic=abort docs/compiler-private-probe.rs; inputs/compiler-probe-o0/o3 6 | 独立 private-entry probe；两档均正确 | 104 |
| [evidence/compiler-public-probe.txt](evidence/compiler-public-probe.txt) | rustc --edition=2024 -C opt-level=0/3 [-C lto=fat] docs/compiler-probe.rs; inputs/compiler-probe-public-o0/o3/lto 6 | 保留导出 schedule 入口后：O0 为 8 步，O3 为 15 步 | 218 |
| [evidence/enzyme-disabled-control.txt](evidence/enzyme-disabled-control.txt) | rustc --edition=2024 --crate-type=staticlib -C panic=abort src/kernel.rs -o inputs/enzyme-disabled-control.a | 负面对照：未加 -Zautodiff，预期编译失败 | 188 |
| [evidence/enzyme-smoke.txt](evidence/enzyme-smoke.txt) | target/release/seismic --enzyme-smoke | cube: primal / JVP / VJP = 8 / 12 / 12 | 16 |
| [evidence/failing-action-audit-tests.txt](evidence/failing-action-audit-tests.txt) | python -m pytest -q tests/test_action_audit.py | 历史 red：尚无 action auditor | 2,860 |
| [evidence/failing-checkpoint-tests.txt](evidence/failing-checkpoint-tests.txt) | python -m pytest -q tests/test_checkpoint.py | 历史 red：尚无 Treeverse | 10,801 |
| [evidence/failing-derivative-tests.txt](evidence/failing-derivative-tests.txt) | python -m pytest -q tests/test_derivatives.py | 历史 red：尚无 Born / adjoint modes | 6,950 |
| [evidence/failing-graph-tests.txt](evidence/failing-graph-tests.txt) | python -m pytest -q test_graphs.py | 历史 red：缺少 plot_graphs | 824 |
| [evidence/failing-modes-tests.txt](evidence/failing-modes-tests.txt) | python -m pytest -q test_modes.py | 历史 red：缺少 compare_modes | 827 |
| [evidence/failing-seismic-tests.txt](evidence/failing-seismic-tests.txt) | python -m pytest -q tests/test_seismic.py | 历史 red：尚无 seismic executable | 7,894 |
| [evidence/failing-tests.txt](evidence/failing-tests.txt) | python -m pytest -q test_lj_ad.py | 历史 red：缺少 lj_ad | 832 |
| [evidence/final-tests.txt](evidence/final-tests.txt) | python -m pytest -q | 最终完整测试结果 | 100 |
| [evidence/format-check.txt](evidence/format-check.txt) | cargo fmt --check | 最终格式检查；退出码 0 | 68 |
| [evidence/forward-run.txt](evidence/forward-run.txt) | 上表 `forward` 命令 | 逐炮 L2 输出 | 103 |
| [evidence/input-fingerprints.txt](evidence/input-fingerprints.txt) | python scripts/download_inputs.py | 确切输入字节 SHA-256 | 179 |
| [evidence/install-seismic.txt](evidence/install-seismic.txt) | bash scripts/setup.sh (Rust installation stage) | 隔离的 pinned nightly + Enzyme 安装 | 707 |
| [evidence/marmousi-born-run.txt](evidence/marmousi-born-run.txt) | 上表 `marmousi-born` 命令 | 逐炮 L2 输出 | 265 |
| [evidence/marmousi-image-run.txt](evidence/marmousi-image-run.txt) | 上表 `marmousi-image` 命令 | 逐炮 L2 输出 | 292 |
| [evidence/part1-derivatives.txt](evidence/part1-derivatives.txt) | python scripts/run_ad.py | LJ 单点导数与解析值 | 279 |
| [evidence/part1-graphs.txt](evidence/part1-graphs.txt) | python scripts/plot_graphs.py | JAX 两种 computation graph | 157 |
| [evidence/part1-modes.txt](evidence/part1-modes.txt) | python scripts/compare_modes.py | 601 点最大绝对误差 | 197 |
| [evidence/passing-action-audit-tests.txt](evidence/passing-action-audit-tests.txt) | python -m pytest -q tests/test_action_audit.py | 独立 audit 三项通过 | 98 |
| [evidence/passing-derivative-tests.txt](evidence/passing-derivative-tests.txt) | python -m pytest -q tests/test_seismic.py tests/test_derivatives.py | 当时 forward + derivative 阶段通过 | 99 |
| [evidence/passing-forward-tests.txt](evidence/passing-forward-tests.txt) | python -m pytest -q tests/test_seismic.py | 当时 forward 阶段通过 | 98 |
| [evidence/passing-seismic-tests.txt](evidence/passing-seismic-tests.txt) | python -m pytest -q tests | 当时 21 项新测试通过 | 99 |
| [evidence/plots.txt](evidence/plots.txt) | python scripts/plot_seismic.py | 讲义 PNG 产生路径 | 316 |
| [evidence/python-lock.txt](evidence/python-lock.txt) | inputs/toolchain/python-tools/bin/uv lock | 固定 Python 依赖 | 85 |
| [evidence/python-sync.txt](evidence/python-sync.txt) | inputs/toolchain/python-tools/bin/uv sync --locked | 锁定环境安装 | 1,012 |
| [evidence/python-tools-install.txt](evidence/python-tools-install.txt) | inputs/toolchain/python-tools/bin/python -m pip install uv==0.12.23 | 本地 Python 工具安装 | 429 |
| [evidence/review-edgecases-failing.txt](evidence/review-edgecases-failing.txt) | python -m pytest -q tests/test_seismic.py -k 'far_ricker_tail or unrecorded_rerun' | 历史 red：source overflow 与 stale recording | 3,712 |
| [evidence/review-report.md](evidence/review-report.md) | Independent read-only source review; arithmetic probes; parent regression verification | 源码复核结果、发现与纠正 | 2,524 |
| [evidence/review-ricker-extremes-failing.txt](evidence/review-ricker-extremes-failing.txt) | python -m pytest -q tests/test_seismic.py -k finite_ricker_extremes | 历史 red：三个 finite source 回归失败 | 4,826 |
| [evidence/review-ricker-extremes-passing.txt](evidence/review-ricker-extremes-passing.txt) | python -m pytest -q tests/test_seismic.py -k finite_ricker_extremes | Ricker 修正后三项通过 | 112 |
| [evidence/rustfmt-install.txt](evidence/rustfmt-install.txt) | rustup component add rustfmt --toolchain nightly-2026-09-05 | 格式化工具安装 | 36 |
| [evidence/sabotage-forward-tangent.txt](evidence/sabotage-forward-tangent.txt) | python -m pytest -q test_lj_ad.py | 历史 mutation：错误 tangent 被检出；随后恢复 | 4,440 |
| [evidence/script-check.txt](evidence/script-check.txt) | bash -n scripts/env.sh scripts/setup.sh scripts/run_all.sh | Shell 语法检查 | 23 |
| [evidence/setup-check.txt](evidence/setup-check.txt) | bash scripts/setup.sh | 已有环境的幂等 setup、锁定同步与 fingerprint 检查 | 427 |
| [evidence/stale-baseline-fixture-failure.txt](evidence/stale-baseline-fixture-failure.txt) | python -m pytest -q | 历史失败：新源码与旧本地数组混用；改为测试自建同版本基准 | 8,713 |
| [evidence/verification.txt](evidence/verification.txt) | python scripts/verify_evidence.py | 全部数值与逐炮 action 审计 | 8,497 |
