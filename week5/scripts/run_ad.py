"""Run all three differentiation passes at r = 1.3 and write
week5/artifacts/ad/derivatives.json (SPEC format).
"""

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lj_ad  # noqa: E402

R = 1.3


def main() -> None:
    primals, tangents = lj_ad.forward_tangents(R, 1.0)
    _, adjoints = lj_ad.reverse_adjoints(R)
    grad = lj_ad.jax_grad(R)

    data = {
        "r": R,
        "energy": primals["U"],
        "tangents": {k: float(v) for k, v in tangents.items()},
        "adjoints": {k: float(v) for k, v in adjoints.items()},
        "jax_grad": grad,
    }
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "ad"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "derivatives.json"
    out.write_text(json.dumps(data, indent=2) + "\n")

    analytic = -48.0 * R ** -13.0 + 24.0 * R ** -7.0
    print(f"U(1.3)            = {data['energy']:.16f}")
    print(f"forward tangent   = {tangents['U']:.16f}")
    print(f"reverse adjoint   = {adjoints['r']:.16f}")
    print(f"jax.grad          = {grad:.16f}")
    print(f"analytic U'(1.3)  = {analytic:.16f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
