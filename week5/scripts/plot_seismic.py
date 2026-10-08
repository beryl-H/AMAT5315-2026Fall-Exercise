"""Plot lecture evidence from the computed arrays and their timestep metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def load_json(path):
    return json.loads(Path(path).read_text())


def units(model):
    return (model["dx"] * model["length_unit_m"] / 1000,
            model["dt"] * model["time_unit_s"],
            model["length_unit_m"] / model["time_unit_s"] / 1000)


def extent(model):
    dx, _, _ = units(model)
    return [0, (model["nx"] - 1) * dx, (model["nz"] - 1) * dx, 0]


def field(ax, values, model, title, label, *, symmetric=False, limits=None):
    if symmetric:
        bound = float(np.max(np.abs(values))) or 1.0
        limits = (-bound, bound) if limits is None else limits
    kwargs = {} if limits is None else dict(vmin=limits[0], vmax=limits[1])
    artist = ax.imshow(values, extent=extent(model), origin="upper", aspect="auto",
                       interpolation="nearest", cmap="RdBu_r" if symmetric else "viridis",
                       **kwargs)
    ax.set(title=title, xlabel="Horizontal position (km)", ylabel="Depth (km)")
    ax.figure.colorbar(artist, ax=ax, label=label, shrink=.88)
    return artist


def geometry(ax, model, shot=None):
    dx, _, _ = units(model)
    shots = np.array(model["shots"]) * dx
    if shot is not None:
        shots = shots[shot:shot+1]
    receivers = np.array(model["receivers"]) * dx
    ax.scatter(shots[:, 0], shots[:, 1], marker="*", s=90, color="#ffad32",
               edgecolor="#493410", label="Sources" if shot is None else "Active source", zorder=3)
    ax.scatter(receivers[:, 0], receivers[:, 1], marker="v", s=24, color="#151f35",
               label="Receivers", zorder=3)
    width = model["sponge_width"] * dx
    xmax, zmax = extent(model)[1:3]
    ax.add_patch(Rectangle((width, width), xmax - 2 * width, zmax - 2 * width,
                           fill=False, linestyle="--", linewidth=1, color="#65718a"))
    ax.legend(loc="lower left", fontsize=8, framealpha=.95)


def save(fig, path):
    path = ART / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def snapshot(folder, step):
    metadata = load_json(ART / folder / "run.json")
    index = metadata["recording"]["steps"].index(step)
    return np.load(ART / folder / "wavefield.npy")[index]


def gather(ax, data, model, title, bound):
    dx, dt, _ = units(model)
    x = np.array(model["receivers"])[:, 0] * dx
    image = ax.imshow(data, extent=[x.min(), x.max(), model["steps"] * dt, dt],
                      origin="upper", aspect="auto", cmap="RdBu_r", interpolation="nearest",
                      vmin=-bound, vmax=bound)
    ax.set(title=title, xlabel="Receiver position (km)", ylabel="Time (s)")
    ax.figure.colorbar(image, ax=ax, label="Pressure (arbitrary units)", shrink=.88)


def reflector_plots():
    model = load_json(ROOT / "inputs/reflector.json")
    background, perturbation = (np.array(model[key]) for key in ["background", "perturbation"])
    dx, dt, velocity_scale = units(model)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    field(axes[0], background * velocity_scale, model, "Background and acquisition geometry",
          "Speed (km/s)", limits=(1.6, 2.0))
    field(axes[1], perturbation * velocity_scale, model, "Thin velocity perturbation",
          "Velocity perturbation (km/s)", symmetric=True)
    for ax in axes:
        geometry(ax, model)
    save(fig, "inputs.png")

    traces = np.load(ART / "forward/traces.npy")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    bound = np.max(np.abs(traces))
    for i, ax in enumerate(axes):
        gather(ax, traces[i], model, f"Forward gather; source x = {model['shots'][i][0]*dx:.1f} km", bound)
    save(fig, "forward/gathers.png")

    for folder, step, title in [("forward", 144, "Forward pressure"),
                                ("adjoint", 132, "Reverse adjoint")]:
        fig, ax = plt.subplots(figsize=(6.8, 5.1), layout="constrained")
        field(ax, snapshot(folder, step), model,
              f"{title}; shot 0, step {step}, t = {step*dt:.2f} s",
              "Pressure (arbitrary units)" if folder == "forward" else "Adjoint (arbitrary units)",
              symmetric=True)
        geometry(ax, model, shot=0)
        ax.axhline(21 * dx, color="#688c39", linestyle=":", linewidth=1.2, label="Reflector")
        save(fig, f"{folder}/wavefield.png")

    diagonal = np.arange(24)
    distances = diagonal * np.sqrt(2) * dx
    fig, ax = plt.subplots(figsize=(8.1, 4.5), layout="constrained")
    peaks = []
    for step, color in [(108, "#2b6cb0"), (144, "#cf684b")]:
        values = snapshot("forward", step)[8 + diagonal, 10 + diagonal]
        peak = int(np.argmax(values))
        peaks.append(float(distances[peak]))
        ax.plot(distances, values, "o-", ms=3, linewidth=1.5, color=color,
                label=f"step {step}; t = {step*dt:.2f} s")
        ax.axvline(distances[peak], color=color, linestyle="--", alpha=.7)
        ax.scatter(distances[peak], values[peak], s=55, color=color, zorder=3)
    displacement = peaks[1] - peaks[0]
    expected = background[0, 0] * velocity_scale * (144 - 108) * dt
    ax.set(xlabel="Distance along shot-0 diagonal (km)", ylabel="Pressure (arbitrary units)",
           title=f"Positive wavefront peak: measured {displacement:.3f} km; expected {expected:.3f} km")
    ax.grid(alpha=.18)
    ax.legend()
    save(fig, "forward/wave-speed.png")

    image = np.load(ART / "adjoint/image.npy")
    profile = np.linalg.norm(image[10:34, 7:34], axis=1)
    peak = 10 + int(np.argmax(profile))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, .7]})
    field(axes[0], perturbation, model, "Known velocity perturbation", "Velocity perturbation (km/s)", symmetric=True)
    field(axes[1], image, model, "Raw migration image", "Adjoint image (arbitrary units)", symmetric=True)
    axes[1].add_patch(Rectangle((7*dx, 10*dx), 26*dx, 23*dx, fill=False,
                                linewidth=1.4, linestyle="--", edgecolor="#424c62"))
    axes[2].plot(profile, np.arange(10, 34) * dx, color="#2b6cb0")
    axes[2].axhline(21*dx, color="#65718a", linestyle="--", label="True center: 2.1 km")
    axes[2].scatter(profile[peak-10], peak*dx, color="#cf684b", label=f"Peak: {peak*dx:.1f} km")
    axes[2].set(title="Row L2 depth profile\nx = 0.7–3.3 km", xlabel="Image row L2 norm", ylabel="Depth (km)", ylim=(3.3, 1.0))
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.18)
    save(fig, "adjoint/image.png")


def checkpoint_plots():
    events = load_json(ART / "checkpoint-5/actions-0.json")
    fig, ax = plt.subplots(figsize=(10.5, 4.6), layout="constrained")
    for action, color in [("store", "#cf684b"), ("restore", "#2b6cb0"), ("call", "#c7cdd6"),
                          ("grad", "#7655a4"), ("fetch", "#399079")]:
        points = [(i, event["step"]) for i, event in enumerate(events) if event["action"] == action]
        x, y = zip(*points)
        ax.scatter(x, y, s=7 if action == "call" else 16, color=color, label=action.title(),
                   zorder=1 if action == "call" else 2)
    ax.set(title="Treeverse schedule; reflector shot 0, five additional slots",
           xlabel="Operation index", ylabel="Timestep")
    ax.legend(ncols=5, loc="upper center", fontsize=9)
    ax.grid(alpha=.15)
    save(fig, "checkpoint-actions.png")

    budgets = [1, 3, 5, 10]
    statistics = [load_json(ART / f"checkpoint-{b}/result.json")["statistics"] for b in budgets]
    baseline = load_json(ART / "adjoint/result.json")["statistics"]
    calls = [s["per_shot"][0]["scheduler_forward_calls"] for s in statistics]
    byte_counts = [s["peak_saved_bytes"] for s in statistics]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    axes[0].plot(budgets, calls, "o-", color="#2b6cb0", label="Treeverse")
    axes[0].axhline(baseline["per_shot"][0]["scheduler_forward_calls"], color="#cf684b",
                   linestyle="--", label="Full history: 240")
    axes[0].set(yscale="log", xlabel="Additional checkpoint slots", ylabel="Forward steps per shot",
                title="Recomputation cost")
    for b, count in zip(budgets, calls):
        axes[0].annotate(f"{count:,}", (b, count), xytext=(0, 8), textcoords="offset points", ha="center")
    axes[0].set_ylim(150, 65000)
    axes[0].legend()
    axes[1].plot(budgets, byte_counts, "o-", color="#399079")
    for b, s in zip(budgets, statistics):
        axes[1].annotate(f"{s['peak_saved_states']} states", (b, s["peak_saved_bytes"]),
                         xytext=(0, 8), textcoords="offset points", ha="center")
    axes[1].set(xlabel="Additional checkpoint slots", ylabel="Peak saved-state storage (bytes)",
                title="Two wavefields per saved state", ylim=(0, max(byte_counts)*1.25))
    axes[1].text(.04, .95, f"Full history: {baseline['peak_saved_bytes']:,} bytes\n({baseline['peak_saved_states']} states)",
                 transform=axes[1].transAxes, va="top", fontsize=9)
    for ax in axes:
        ax.set_xticks(budgets)
        ax.grid(alpha=.18)
    save(fig, "checkpoint-work.png")


def marmousi_plots():
    model = load_json(ROOT / "inputs/marmousi.json")
    dx, _, scale = units(model)
    shot = next(i for i, position in enumerate(model["shots"]) if np.isclose(position[0]*dx, 10))
    born = np.load(ART / "marmousi-born/born_data.npy")
    image = np.load(ART / "marmousi-image/image.npy")
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.5), layout="constrained")
    field(axes[0, 0], np.array(model["background"])*scale, model,
          "Smoothed Marmousi background", "Speed (km/s)")
    field(axes[0, 1], np.array(model["perturbation"])*scale, model,
          "Short-wavelength velocity perturbation", "Velocity perturbation (km/s)", symmetric=True)
    gather(axes[1, 0], born[shot], model, "Born gather; source x = 10.0 km", np.max(np.abs(born[shot])))
    field(axes[1, 1], image, model, "Checkpointed raw migration image", "Adjoint image (arbitrary units)", symmetric=True)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        ax.add_patch(Rectangle((8, 0), 6, 3, fill=False, edgecolor="#28394a", linewidth=1.2, linestyle="--"))
    stats = load_json(ART / "marmousi-image/result.json")["statistics"]
    fig.suptitle(f"Marmousi: image L2 = {np.linalg.norm(image):.8e}; "
                 f"{stats['peak_saved_states']} saved states = {stats['peak_saved_bytes']:,} bytes\n"
                 "One amplitude scale across depth; dashed box: x = 8–14 km, upper 3 km", fontsize=12)
    save(fig, "marmousi.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reflector-only", action="store_true")
    args = parser.parse_args()
    reflector_plots()
    checkpoint_plots()
    if not args.reflector_only:
        marmousi_plots()
