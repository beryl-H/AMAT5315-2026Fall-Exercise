# Week 9：按学习单逐步复现

本周交付是同一个 Rust CPU 推断程序及其真实运行证据。讲义中的课堂问答不单独作为作业答案。先读 README 中的实测结果和环境，再按下面顺序复现；新运行请换输出目录，程序会拒绝覆盖已有运行文件。

本机使用 WSL Ubuntu。Rust 1.96 安装在本次任务专用目录，不覆盖旧版环境。先在 WSL 中执行：

```bash
source /mnt/d/Keria/Course/AMAT5315/week9-support/activate-local.sh
cd /home/keria_h/AMAT5315-2026Fall-Exercise/week9
```

其他电脑按 README 安装 Rust 和 Python 依赖即可；不需要复制本机的激活脚本。已保存的输出可直接读取；需要从头计算时运行 `bash reproduce.sh`，它自动创建新的运行目录。

## 1. 先读对模型，再比较两种精确算法

`asia.uai` 中变量依次是 A、T、S、L、B、E、X、D；0 表示“有／是／阳性”。因子 scope 是数组的轴顺序，最后一个轴变化最快。例如 T 表的 scope 为 `[1,0]`，所以行是 T、列是 A：

```
             A=0   A=1
T=0          .05   .01
T=1          .95   .99
```

未知旅行史时 `P(T=0)=.01*.05+.99*.01=.0104`。先检查这个小结果，防止枚举和收缩共享同一个读表错误。E 表的实际 scope 是 `[5,3,1]`，即 E,L,T，也应按文件读。

枚举逐一列出 256 种赋值，把八个因子相乘后加总。张量收缩计算同一个和，但通过局部乘法和尽早求和来减少工作。观测固定相应索引；查询索引保持开放；其它索引求和。开放结果还是联合质量，最后除以总质量才得到条件概率。

```
cargo +1.96.0 install --locked --path .
infer --model asia.uai --evidence prior --out artifacts/prior
infer --model asia.uai --evidence AD --out artifacts/AD
infer --model asia.uai --evidence ADX --out artifacts/ADX
```

查看 `evidence/graphs.svg`：A,D 被选为0，只有 T 开放，S 的三处出现共用一个索引。查看 `posteriors.png` 比较疾病概率，查看 `joint.png` 比较 T,L 联合分布与边缘分布乘积。联合减乘积表示**依赖性**，不是数值算法误差。共同观测可能让原本弱相关的疾病变得明显相关。

实测 ADX 下 `P(L=0)=0.4442705`，但再确认 T=0 后，由联合分布计算 `P(L=0|T=0,ADX)=0.0222135254/0.39171172=0.05670886`。在这个教学模型里，结核已能解释阳性 X 光，因此肺癌作为另一解释的概率降低；这也是为什么不能用两个边缘概率的乘积替代联合分布。

同一程序还运行 AB、tr(AB)、三自旋 Ising 环的收缩检查；Python 使用直接求和独立核验。Ising 的权重不要求预先归一，配分函数就是全部八个权重之和。

## 2. 改变计算顺序，不改变概率

```
infer --model asia.uai --evidence AD --order random --seed 42 --out runs/random
infer --model asia.uai --evidence AD --order greedy --out runs/greedy
infer --model asia.uai --evidence AD --analyze runs/random/order.json --out runs/analysis
infer --model asia.uai --evidence AD --plan runs/greedy/order.json --out runs/replay
```

每一步的 work 是两个输入标签并集的维数乘积，整棵树的 work 是各步相加。largest tensor 是所有输入及中间张量的最大元素数。中间结果只可消去子树外面和最终输出都不再需要的索引；不能见到两个表共享一个索引就立即消去，因为第三张表也可能需要它。

`order.json` 保存真正执行的树；`complexity.json` 保存逐步标签、工作和大小。`--analyze` 只检查结构，不执行推断。`evidence/orders.png` 展示实际树及最大中间张量。随机数或同分选择可使成本不同于讲义示例；不以强行得到某个加速比为验收标准。

边缘查询复用已经选好的标量树拓扑，再重新标注需要保留的查询轴，不为每个查询单独重新选树。对一个二元查询，每个中间张量最多比标量树多一个二元轴，因此大小至多增加两倍；T,L 联合查询相应至多四倍。这里仍把查询腿保持开放，用 tenferro 真正收缩。每个查询的实际计划和成本也单独保存，便于重算。

本实现调用 omeco 选择树拓扑，再按讲义规则重建每个节点的保留标签。保存的设置说明这一点；独立 Python 审计重算实际执行树的成本。贪心和 TreeSA 都不证明全局最优。

## 3. 复用局部计算：BP

BP 每条有向边保存一个状态概率向量。变量到因子的消息乘入其它因子的旧消息，因子到变量的消息把其它变量求和；两种消息都排除接收方的消息。同步更新要求本轮全部读取上一状态，不能一边更新一边读取新值。

```
infer --model asia.uai --evidence prior --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-prior
infer --model asia.uai --evidence AD --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-AD
infer --model asia.uai --evidence ADX --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-ADX
```

这里 `m'=.5*m+.5*F(m)`；残差是同一状态的 `max|F(m)-m|`，不能用较小的阻尼步长代替。`run.json` 保存每一步的信念、未阻尼残差和累计消息更新工作。树上的消息对应不重叠分支，应该与独立枚举吻合；有环图重复使用相关信息，因此收敛也可能不准确。

查看 `evidence/bp.png`：残差曲线回答“消息是否停止变化”，边缘误差曲线回答“概率是否接近精确答案”。这两个问题需要分别判断。

## 4. 扩展到 Promedus_14

模型有414个二元变量和9个观测，不能枚举 `2^414` 行。观测来自下载的 `.evid` 文件，不是 ASIA 的 AD/ADX 标签。

```
infer --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --order greedy --out runs/promedus-greedy
infer --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --order treesa --seed 42 --trials 4 --iterations 50 --out runs/promedus-treesa
infer --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-promedus
```

MARKOV 因子没有预先归一要求，所以 `Z_evidence` 是证据下配分和，不能直接叫“证据概率”。每次缩放中间张量都要保存被移除倍率的对数；最终恢复 `log10 Z_evidence`，与 PR 比较。MAR 只检查归一化概率，丢掉全局倍率可能仍通过 MAR 却不能通过 PR。

默认精确查询是59、102、139，JSON 中每个向量按 state0,state1 排列。BP 计算所有节点，因此工作量比较应说明两者查询范围不同。BP 的工作统计是文档所述乘法计数；收缩的 work 是索引组合项数，两者不是完全相同的总 FLOP 指标。

omeco 0.2.6 的 TreeSA 在库内部固定试验种子42、43……，因此本程序只接受 `--seed 42`，并记录每次试验的真实种子与完整搜索设置。不能把仅保存的任意 seed 字段当作已控制随机性的证据。

## 5. 独立核验与提交前检查

```
cargo +1.96.0 test --locked
python -m unittest discover -s tests -p 'test_python_audit.py'
python verify_and_plot.py
```

Python 从原始 UAI 独立枚举 ASIA、直接求和矩阵/Ising、从保存的树重算成本，并读取下载的 PR/MAR 比较；缺失真实运行不会补造结果。具体误差、时间、成本、BP状态及审计路径以 README 和最新完整 audit 为准。图表目视检查补充数值检查，不能相互替代。
