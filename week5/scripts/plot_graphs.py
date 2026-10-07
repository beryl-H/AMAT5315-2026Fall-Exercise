"""Draw the computational graph JAX records for U at r = 1.3 and for the
gradient of U, from the jaxprs captured with jax.make_jaxpr.

One node per operation, labelled with the operation's primitive name.
Saves week5/artifacts/ad/graph.png (forward) and
week5/artifacts/ad/grad-graph.png (gradient).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lj_ad  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

R = 1.3
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "ad"





def _is_literal(v) -> bool:
    return type(v).__name__ == "Literal"


def jaxpr_to_graph(jaxpr, out_label: str, out_names: list[str] | None = None):
    """Turn a jaxpr into (nodes, edges).

    nodes: dicts with id, op (primitive name), label, kind
    ('op' | 'literal' | 'input' | 'out').
    edges: (src_id, dst_id) pairs, producer -> consumer.
    Vars are keyed by object identity: an eqn's invar is the same object
    as the producer eqn's outvar.
    """
    producer: dict[int, str] = {}  # id(var) -> node id
    for v in jaxpr.jaxpr.invars:
        producer[id(v)] = "in"

    nodes: list[dict] = []
    edges: list[tuple[str, str]] = []
    used_literals: dict[int, str] = {}  # id(literal) -> node id

    nodes.append({"id": "in", "op": "r", "label": "r", "kind": "input"})
    for i, eqn in enumerate(jaxpr.jaxpr.eqns):
        nid = f"eqn{i}"
        prim = eqn.primitive.name
        out = eqn.outvars[0]
        name = out_names[i] if out_names and i < len(out_names) else f"v{i}"
        nodes.append(
            {"id": nid, "op": prim, "label": f"{prim}\n→ {name}", "kind": "op"}
        )
        producer[id(out)] = nid
        for v in eqn.invars:
            if _is_literal(v):
                if id(v) not in used_literals:
                    lid = f"lit{i}_{len(used_literals)}"
                    used_literals[id(v)] = lid
                    nodes.append(
                        {"id": lid, "op": "const",
                         "label": str(v.val), "kind": "literal"}
                    )
                edges.append((used_literals[id(v)], nid))
            else:
                edges.append((producer[id(v)], nid))
    nodes.append({"id": "out", "op": out_label,
                  "label": out_label, "kind": "out"})
    for v in jaxpr.jaxpr.outvars:
        edges.append((producer[id(v)], "out"))
    return nodes, edges


def build_energy_graph():
    jp = jax.make_jaxpr(lj_ad.energy)(jnp.asarray(R, jnp.float64))
    return jaxpr_to_graph(jp, "U", ["a", "b", "c", "U"])


def build_grad_graph():
    jp = jax.make_jaxpr(jax.grad(lj_ad.energy))(jnp.asarray(R, jnp.float64))
    return jaxpr_to_graph(jp, "dU/dr")


def _layout(nodes, edges):
    """x = longest path from any source; y = spread within each column."""
    succ: dict[str, list[str]] = {}
    for s, t in edges:
        succ.setdefault(s, []).append(t)
    order = {n["id"]: i for i, n in enumerate(nodes)}
    level: dict[str, int] = {}

    # compute longest-path levels iteratively in topological order
    import collections

    indeg = {n["id"]: 0 for n in nodes}
    for s, t in edges:
        indeg[t] += 1
    q = collections.deque([n["id"] for n in nodes if indeg[n["id"]] == 0])
    topo = []
    while q:
        nid = q.popleft()
        topo.append(nid)
        for t in succ.get(nid, []):
            indeg[t] -= 1
            if indeg[t] == 0:
                q.append(t)
    for nid in topo:
        level[nid] = 0
    for nid in topo:
        for t in succ.get(nid, []):
            level[t] = max(level[t], level[nid] + 1)

    cols: dict[int, list[str]] = {}
    for n in nodes:
        cols.setdefault(level[n["id"]], []).append(n["id"])
    pos = {}
    for lvl, ids in cols.items():
        ids.sort(key=lambda i: order[i])
        k = len(ids)
        for j, nid in enumerate(ids):
            pos[nid] = (lvl, (k - 1) / 2.0 - j)
    return pos


def draw(nodes, edges, title: str, out_path: Path) -> None:
    pos = _layout(nodes, edges)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    width = max(xs) - min(xs) + 2
    height = max(ys) - min(ys) + 2
    fig, ax = plt.subplots(figsize=(2.2 * width, 1.1 * height))

    for s, t in edges:
        (x0, y0), (x1, y1) = pos[s], pos[t]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color="0.45", lw=1.1,
                shrinkA=16, shrinkB=16,
            ),
        )
    styles = {
        "op": dict(fc="#cfe2f3", ec="#1f4e79"),
        "literal": dict(fc="#ececec", ec="#666666"),
        "input": dict(fc="#d5e8d4", ec="#2d6a2d"),
        "out": dict(fc="#ffe6cc", ec="#b46504"),
    }
    for n in nodes:
        x, y = pos[n["id"]]
        box = FancyBboxPatch(
            (x - 0.42, y - 0.28), 0.84, 0.56,
            boxstyle="round,pad=0.06", **styles[n["kind"]],
        )
        ax.add_patch(box)
        ax.text(x, y, n["label"], ha="center", va="center", fontsize=9)

    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes, edges = build_energy_graph()
    draw(nodes, edges, f"JAX computation graph for U(r) at r = {R}",
         OUT_DIR / "graph.png")
    nodes, edges = build_grad_graph()
    draw(nodes, edges, f"JAX computation graph for dU/dr at r = {R}",
         OUT_DIR / "grad-graph.png")


if __name__ == "__main__":
    main()
