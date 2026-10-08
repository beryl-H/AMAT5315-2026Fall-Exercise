"""Append an exhaustive evidence inventory to the authored README prefix."""
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- GENERATED EVIDENCE INVENTORY -->"

LOGS = {
    "baseline-part1.txt": ("python -m pytest -q test_lj_ad.py test_modes.py test_graphs.py", "原有 13 项测试基线"),
    "failing-tests.txt": ("python -m pytest -q test_lj_ad.py", "历史 red：缺少 lj_ad"),
    "failing-modes-tests.txt": ("python -m pytest -q test_modes.py", "历史 red：缺少 compare_modes"),
    "failing-graph-tests.txt": ("python -m pytest -q test_graphs.py", "历史 red：缺少 plot_graphs"),
    "sabotage-forward-tangent.txt": ("python -m pytest -q test_lj_ad.py", "历史 mutation：错误 tangent 被检出；随后恢复"),
    "failing-seismic-tests.txt": ("python -m pytest -q tests/test_seismic.py", "历史 red：尚无 seismic executable"),
    "failing-derivative-tests.txt": ("python -m pytest -q tests/test_derivatives.py", "历史 red：尚无 Born / adjoint modes"),
    "failing-checkpoint-tests.txt": ("python -m pytest -q tests/test_checkpoint.py", "历史 red：尚无 Treeverse"),
    "failing-action-audit-tests.txt": ("python -m pytest -q tests/test_action_audit.py", "历史 red：尚无 action auditor"),
    "checkpoint-recompute-failure.txt": ("python -m pytest -q tests/test_checkpoint.py", "历史失败：直接递归重算 work 不符"),
    "checkpoint-barrier-failure.txt": ("python -m pytest -q tests/test_checkpoint.py", "历史失败：scalar barrier 尝试未纠正 work"),
    "checkpoint-debug-trace.txt": ("seismic --experiment inputs/debug-schedule/model.json --mode adjoint --data inputs/debug-schedule/weights.npy --storage treeverse --checkpoints 2", "历史诊断：加入口参数打印后得到正确恢复基点；不是最终程序"),
    "review-edgecases-failing.txt": ("python -m pytest -q tests/test_seismic.py -k 'far_ricker_tail or unrecorded_rerun'", "历史 red：source overflow 与 stale recording"),
    "review-ricker-extremes-failing.txt": ("python -m pytest -q tests/test_seismic.py -k finite_ricker_extremes", "历史 red：三个 finite source 回归失败"),
    "review-ricker-extremes-passing.txt": ("python -m pytest -q tests/test_seismic.py -k finite_ricker_extremes", "Ricker 修正后三项通过"),
    "passing-forward-tests.txt": ("python -m pytest -q tests/test_seismic.py", "当时 forward 阶段通过"),
    "passing-derivative-tests.txt": ("python -m pytest -q tests/test_seismic.py tests/test_derivatives.py", "当时 forward + derivative 阶段通过"),
    "passing-seismic-tests.txt": ("python -m pytest -q tests", "当时 21 项新测试通过"),
    "passing-action-audit-tests.txt": ("python -m pytest -q tests/test_action_audit.py", "独立 audit 三项通过"),
    "final-tests.txt": ("python -m pytest -q", "最终完整测试结果"),
    "enzyme-disabled-control.txt": ("rustc --edition=2024 --crate-type=staticlib -C panic=abort src/kernel.rs -o inputs/enzyme-disabled-control.a", "负面对照：未加 -Zautodiff，预期编译失败"),
    "enzyme-smoke.txt": ("target/release/seismic --enzyme-smoke", "cube: primal / JVP / VJP = 8 / 12 / 12"),
    "install-seismic.txt": ("bash scripts/setup.sh (Rust installation stage)", "隔离的 pinned nightly + Enzyme 安装"),
    "rustfmt-install.txt": ("rustup component add rustfmt --toolchain nightly-2026-09-05", "格式化工具安装"),
    "format-check.txt": ("cargo fmt --check", "最终格式检查；退出码 0"),
    "python-tools-install.txt": ("inputs/toolchain/python-tools/bin/python -m pip install uv==0.12.23", "本地 Python 工具安装"),
    "python-lock.txt": ("inputs/toolchain/python-tools/bin/uv lock", "固定 Python 依赖"),
    "python-sync.txt": ("inputs/toolchain/python-tools/bin/uv sync --locked", "锁定环境安装"),
    "part1-derivatives.txt": ("python scripts/run_ad.py", "LJ 单点导数与解析值"),
    "part1-modes.txt": ("python scripts/compare_modes.py", "601 点最大绝对误差"),
    "part1-graphs.txt": ("python scripts/plot_graphs.py", "JAX 两种 computation graph"),
    "plots.txt": ("python scripts/plot_seismic.py", "讲义 PNG 产生路径"),
    "verification.txt": ("python scripts/verify_evidence.py", "全部数值与逐炮 action 审计"),
    "input-fingerprints.txt": ("python scripts/download_inputs.py", "确切输入字节 SHA-256"),
    "compiler-optimization-probe.txt": ("rustc --edition=2024 -C opt-level=0/3 -C panic=abort docs/compiler-private-probe.rs; inputs/compiler-probe-o0/o3 6", "独立 private-entry probe；两档均正确"),
    "compiler-public-probe.txt": ("rustc --edition=2024 -C opt-level=0/3 [-C lto=fat] docs/compiler-probe.rs; inputs/compiler-probe-public-o0/o3/lto 6", "保留导出 schedule 入口后：O0 为 8 步，O3 为 15 步"),
    "script-check.txt": ("bash -n scripts/env.sh scripts/setup.sh scripts/run_all.sh", "Shell 语法检查"),
    "setup-check.txt": ("bash scripts/setup.sh", "已有环境的幂等 setup、锁定同步与 fingerprint 检查"),
    "stale-baseline-fixture-failure.txt": ("python -m pytest -q", "历史失败：新源码与旧本地数组混用；改为测试自建同版本基准"),
    "review-report.md": ("Independent read-only source review; arithmetic probes; parent regression verification", "源码复核结果、发现与纠正"),
}


def main():
    readme = ROOT / "README.md"
    prefix, found, _ = readme.read_text().partition(MARKER)
    assert found, "README inventory marker is required"
    aliases = {}
    for folder in sorted((ROOT / "artifacts").iterdir()):
        if not folder.is_dir() or not (folder / "run.json").exists():
            continue
        arguments = json.loads((folder / "run.json").read_text())["arguments"]
        command = ["target/release/seismic"]
        for key, value in arguments.items():
            command.extend([key, str(value)])
        aliases[folder.name] = shlex.join(command)
    output = [MARKER, "", "| 运行代号 | 完整命令 |", "| --- | --- |"]
    for name, command in aliases.items():
        output.append(f"| `{name}` | `{command}` |")
    output.extend(["", "| 文件 | 产生它的命令／脚本 | 支持的量／状态 | 字节 |",
                   "| --- | --- | --- | ---: |"])
    files = sorted(p for directory in [ROOT / "artifacts", ROOT / "evidence"]
                   for p in directory.rglob("*") if p.is_file())
    count = 0
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix not in [".png", ".json", ".npy", ".txt", ".md"]:
            continue
        size = path.stat().st_size
        if path.suffix != ".npy":
            assert size < 5_000_000, (str(path), size)
        if relative.parts[0] == "artifacts":
            if relative.parts[1] == "ad":
                producer = {"derivatives.json": "python scripts/run_ad.py", "modes.png": "python scripts/compare_modes.py",
                            "graph.png": "python scripts/plot_graphs.py", "grad-graph.png": "python scripts/plot_graphs.py"}[path.name]
                claim = "LJ 导数或 JAX primitive graph"
            elif path.name == "verification.json":
                producer, claim = "python scripts/verify_evidence.py", "机器可读的独立核验结果"
            elif path.suffix == ".png":
                producer, claim = "python scripts/plot_seismic.py", "按真实数组、步号、单位绘制的展示证据"
            else:
                producer = f"上表 `{relative.parts[1]}` 命令"
                claim = {"run.json": "参数、源文件 fingerprint、recording 语义", "result.json": "模式、规模、耗时、actual counters"}.get(path.name,
                         "动作顺序与 live saved slots" if path.name.startswith("actions-") else "local / ignored：原始数值数组")
        elif path.name.startswith("build-"):
            producer, claim = "cargo build --release --locked", "该文件名对应阶段的构建输出（最终见 build-final.txt）"
        elif path.name in LOGS:
            producer, claim = LOGS[path.name]
        elif path.name.endswith("-run.txt"):
            alias = path.name.removesuffix("-run.txt")
            producer, claim = f"上表 `{alias}` 命令", "逐炮 L2 输出"
        else:
            raise ValueError(f"No explicit inventory producer for {relative}")
        producer = producer.replace("|", "&#124;")
        output.append(f"| [{relative}]({relative.as_posix()}) | {producer} | {claim} | {size:,} |")
        count += 1
    readme.write_text(prefix+"\n".join(output)+"\n")
    print(f"inventoried {count} files in README.md")


if __name__ == "__main__":
    main()
