import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_shard(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # meta is an object array of json strings
    meta = [json.loads(s) for s in data["meta"].tolist()]
    return {
        "A": data["A"],
        "B": data["B"],
        "dt": data["dt"],
        "x": data["x"],
        "u": data["u"],
        "y": data["y"],
        "Q0": data["Q0"],
        "R0": data["R0"],
        "meta": meta,
    }

def validate_shard(shard: dict, max_print: int = 10):
    A, B, dt = shard["A"], shard["B"], shard["dt"]
    x, u, y = shard["x"], shard["u"], shard["y"]
    Q0, R0, meta = shard["Q0"], shard["R0"], shard["meta"]

    N = x.shape[0]
    problems = []

    # Basic shape checks
    if not (A.shape[0] == B.shape[0] == dt.shape[0] == x.shape[0] == u.shape[0] == y.shape[0] == Q0.shape[0] == R0.shape[0] == len(meta)):
        problems.append("Batch dimension mismatch across stored arrays/meta.")

    # Per-episode checks
    bad = 0
    for i in range(N):
        n = A[i].shape[0]
        m = B[i].shape[1]

        # shape consistency
        if x[i].shape[1] != n or y[i].shape[1] != n:
            problems.append(f"Episode {i}: state dimension mismatch.")
            bad += 1
            continue
        if u[i].shape[1] != m:
            problems.append(f"Episode {i}: control dimension mismatch.")
            bad += 1
            continue

        # finiteness
        if not np.isfinite(x[i]).all() or not np.isfinite(u[i]).all() or not np.isfinite(y[i]).all():
            problems.append(f"Episode {i}: non-finite values found.")
            bad += 1
            continue

        # symmetry checks (numerical)
        if np.max(np.abs(Q0[i] - Q0[i].T)) > 1e-5:
            problems.append(f"Episode {i}: Q0 not symmetric within tolerance.")
        if np.max(np.abs(R0[i] - R0[i].T)) > 1e-5:
            problems.append(f"Episode {i}: R0 not symmetric within tolerance.")

        # meta sanity
        if "eig_cl" in meta[i] and len(meta[i]["eig_cl"]) > 0:
            eig_mag = np.array(meta[i]["eig_cl"], dtype=float)
            if np.any(eig_mag < 0) or not np.isfinite(eig_mag).all():
                problems.append(f"Episode {i}: invalid eig_cl magnitudes.")
        else:
            # It's OK to be empty if you ever saved dlqr_failed episodes (you currently skip them, so normally not)
            pass

    print(f"Loaded {N} episodes")
    print(f"x shape: {x.shape}, u shape: {u.shape}, A shape: {A.shape}, B shape: {B.shape}")
    print(f"Bad episodes (hard failures): {bad}")

    if problems:
        print("\nTop issues:")
        for p in problems[:max_print]:
            print(" -", p)
        if len(problems) > max_print:
            print(f" ... and {len(problems)-max_print} more")
    else:
        print("No issues detected.")

def export_examples_to_json(shard: dict, out_path: str, indices=None, downsample:int=1):
    """
    Export a few episodes to a human-readable JSON.
    Warning: full trajectories can be large. Use downsample>1 to reduce size.
    """
    A, B, dt = shard["A"], shard["B"], shard["dt"]
    x, u, y = shard["x"], shard["u"], shard["y"]
    Q0, R0, meta = shard["Q0"], shard["R0"], shard["meta"]

    N = x.shape[0]
    if indices is None:
        indices = list(range(min(3, N)))

    out = {"episodes": []}
    for i in indices:
        ep = {
            "i": int(i),
            "dt": float(dt[i]),
            "A": A[i].tolist(),
            "B": B[i].tolist(),
            "Q0": Q0[i].tolist(),
            "R0": R0[i].tolist(),
            "meta": meta[i],
            "x": x[i][::downsample].tolist(),
            "u": u[i][::downsample].tolist(),
            "y": y[i][::downsample].tolist(),
        }
        out["episodes"].append(ep)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(indices)} episodes to: {out_path}")

def plot_episode(shard: dict, i: int):
    x = shard["x"][i]
    u = shard["u"][i]
    meta = shard["meta"][i]

    T = u.shape[0]
    t_x = np.arange(x.shape[0])
    t_u = np.arange(T)

    # Plot states
    plt.figure()
    for k in range(x.shape[1]):
        plt.plot(t_x, x[:, k], label=f"x[{k}]")
    plt.title(f"Episode {i} - States")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.grid(True)

    # Plot controls
    plt.figure()
    for k in range(u.shape[1]):
        plt.plot(t_u, u[:, k], label=f"u[{k}]")
    plt.title(f"Episode {i} - Controls")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)

    # Plot eigen magnitude summary
    eig_mag = meta.get("eig_cl", [])
    if len(eig_mag) > 0:
        plt.figure()
        eig_mag = np.array(eig_mag, dtype=float)
        plt.stem(np.arange(len(eig_mag)), eig_mag)
        plt.title(f"Episode {i} - |eig(A-BK)| (baseline)")
        plt.xlabel("mode")
        plt.ylabel("|lambda|")
        plt.grid(True)

        # Quick textual stability indicator
        rho = float(np.max(eig_mag))
        print(f"Episode {i}: spectral radius max|lambda| = {rho:.4f}  (stable if < 1.0)")

    plt.show()

def plot_dataset_overview(shard: dict):
    meta = shard["meta"]
    # Gather spectral radii
    rhos = []
    costs = []
    for m in meta:
        eig_mag = m.get("eig_cl", [])
        if len(eig_mag) > 0:
            rhos.append(float(np.max(eig_mag)))
        if "cost" in m and np.isfinite(m["cost"]):
            costs.append(float(m["cost"]))

    if rhos:
        plt.figure()
        plt.hist(rhos, bins=50)
        plt.title("Distribution of baseline closed-loop spectral radius max|eig(A-BK)|")
        plt.xlabel("max|lambda|")
        plt.ylabel("count")
        plt.grid(True)

    plt.show()

def plot_cost_distribution(shard):
    meta = shard["meta"]
    costs = [
        float(m["cost"])
        for m in meta
        if "cost" in m and np.isfinite(m["cost"]) and m["cost"] > 0
    ]

    costs = np.array(costs)

    plt.figure()
    plt.hist(costs, bins=100)
    plt.yscale("log")       # optional
    plt.xscale("log")       # KEY FIX
    plt.title("Cost distribution (log-log scale)")
    plt.xlabel("cost (log)")
    plt.ylabel("count")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Point this at any shard you generated
    shard_path = r"C:\Users\Denisa\Desktop\MPC_new\MPC_dataset\synthetic_lqr_data\synthetic_shard_0013.npz"
    shard = load_shard(shard_path)
    print("\nMeta keys for episode 0:")
    print(shard["meta"][0].keys())


    validate_shard(shard)

    # Export a few examples for inspection (downsample to keep file reasonable)
    export_examples_to_json(
        shard,
        out_path=os.path.join("MPC_dataset", "synthetic_lqr_data", "examples.json"),
        indices=[0, 1, 2],
        downsample=4
    )


    # Plot one episode + overall stats
    plot_episode(shard, i=0)
    plot_dataset_overview(shard)
    plot_cost_distribution(shard)
