# qml_project.py  (replace your existing file with this)
import os
import argparse
import time
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import trange
import pennylane as qml

# ---------------------------
# CONFIG
# ---------------------------
np.random.seed(42)
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Default QML hyperparams (can be overridden)
DEFAULT_QUBITS = 2
DEFAULT_LAYERS = 3
DEFAULT_EPOCHS = 80
DEFAULT_LR = 0.05
DEFAULT_RESTARTS = 2
DEFAULT_NOISE = 0.03

EPS = 1e-8

# ---------------------------
# Helper: interactive prompt only when running in a tty
# ---------------------------
def prompt_if_tty(prompt_text, default, cast=float):
    """Prompt the user if stdin is a TTY. Return cast(default) if empty or not a TTY."""
    if not sys.stdin.isatty():
        # non-interactive environment (e.g. spawned by frontend) -> return default casted
        return cast(default)
    try:
        raw = input(f"{prompt_text} [{default}]: ").strip()
    except EOFError:
        return cast(default)
    if raw == "":
        return cast(default)
    try:
        return cast(raw)
    except Exception:
        print(f"Invalid input '{raw}', using default {default}")
        return cast(default)

# ---------------------------
# DATA
# ---------------------------
def load_data(n_samples=300, noise=0.18, test_size=0.3, seed=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return (X_train, X_test, y_train, y_test), scaler

# ---------------------------
# QUANTUM CIRCUIT HELPERS
# ---------------------------
def angle_encoding(x, n_qubits):
    """Angle encoding: map each feature to angle in [-pi,pi]. If fewer features than qubits, wrap."""
    xr = np.asarray(x)
    if xr.std() < 1e-6:
        scaled = np.tanh(xr)
    else:
        scaled = np.tanh((xr - xr.mean()) / (xr.std() + 1e-8))
    angles = np.clip(scaled, -1, 1) * np.pi
    for i in range(n_qubits):
        val = float(angles[i % len(angles)])
        qml.RY(val, wires=i)

def variational_layer(params_row, n_qubits):
    """Single layer: per-qubit RY + linear chain entangler"""
    for q in range(n_qubits):
        qml.RY(float(params_row[q]), wires=q)
    if n_qubits > 1:
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q+1])

# ---------------------------
# DEVICE & QNODE factory (mixed for noisy expectation)
# ---------------------------
def make_qnode(n_qubits=DEFAULT_QUBITS):
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(params, x, noise=0.0):
        params = np.array(params)
        L = params.shape[0]
        angle_encoding(x, n_qubits)

        if noise and noise > 0.0:
            for w in range(n_qubits):
                qml.DepolarizingChannel(noise, wires=w)

        for l in range(L):
            variational_layer(params[l], n_qubits)
            if noise and noise > 0.0:
                for w in range(n_qubits):
                    qml.DepolarizingChannel(noise, wires=w)

        return qml.expval(qml.PauliZ(0))

    return circuit

# Separate QNode that returns statevector for expressibility (use default.qubit)
def make_state_qnode(n_qubits=DEFAULT_QUBITS):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit_state(params, x):
        params = np.array(params)
        L = params.shape[0]
        angle_encoding(x, n_qubits)
        for l in range(L):
            variational_layer(params[l], n_qubits)
        return qml.state()

    return circuit_state

# ---------------------------
# PREDICTION / METRICS
# ---------------------------
def expval_to_prob(exp_val):
    return float(np.clip((1.0 - exp_val) / 2.0, EPS, 1.0 - EPS))

def predict_prob(circuit, params, X, noise=0.0):
    """X: iterable of samples (shape (N, dim))"""
    probs = []
    for x in X:
        exp_val = float(circuit(params, x, noise))
        probs.append(expval_to_prob(exp_val))
    return np.array(probs)

def accuracy(circuit, params, X, y, noise=0.0):
    probs = predict_prob(circuit, params, X, noise)
    preds = (probs >= 0.5).astype(int)
    return float(np.mean(preds == y))

def cross_entropy(circuit, params, X, y, noise=0.0):
    p = predict_prob(circuit, params, X, noise)
    return -float(np.mean(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)))

def evaluate_and_save_classification(circuit, params, X_test, y_test, prefix="qml", noise=0.0):
    probs = predict_prob(circuit, params, X_test, noise)
    preds = (probs >= 0.5).astype(int)
    acc = float(np.mean(preds == y_test))
    cm = confusion_matrix(y_test, preds)
    clr_text = classification_report(y_test, preds, digits=4)

    # save numpy eval
    np.savez(os.path.join(RESULT_DIR, f"{prefix}_eval.npz"), probs=probs, preds=preds, y=y_test)

    # save textual report
    with open(os.path.join(RESULT_DIR, f"{prefix}_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\nConfusion Matrix:\n{cm}\n\nClassification Report:\n{clr_text}\n")

    # save confusion matrix as image
    fig, ax = plt.subplots(figsize=(4,4))
    ax.matshow(cm, cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='black')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{prefix} Confusion Matrix (acc={acc:.3f})")
    plt.tight_layout()
    cm_path = os.path.join(RESULT_DIR, f"{prefix}_confusion.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved evaluation for {prefix}. Accuracy={acc:.4f}")
    return acc, cm, clr_text

# ---------------------------
# TRAINING
# ---------------------------
def train(params_init,
          circuit,
          X_train, y_train,
          X_test, y_test,
          epochs=DEFAULT_EPOCHS,
          lr=DEFAULT_LR,
          noise=0.0,
          verbose=True,
          clip_norm=None,
          weight_decay=0.0):
    params = np.array(params_init, dtype=float)
    opt = qml.GradientDescentOptimizer(stepsize=lr)

    train_losses = []
    test_accs = []

    for epoch in trange(epochs, desc="Training", leave=False):
        def objective(p):
            return cross_entropy(circuit, p, X_train, y_train, noise)
        params = opt.step(objective, params)

        if clip_norm is not None:
            norm = np.linalg.norm(params)
            if norm > clip_norm:
                params = params * (clip_norm / norm)

        if weight_decay > 0:
            params = params * (1 - lr * weight_decay)

        loss_val = float(objective(params))
        train_losses.append(loss_val)
        test_accs.append(float(accuracy(circuit, params, X_test, y_test, noise=0.0)))

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs}  loss={loss_val:.4f}  test_acc={test_accs[-1]:.4f}")

    history = {"train_loss": np.array(train_losses), "test_acc": np.array(test_accs)}
    return params, history

def train_with_restarts(circuit, X_train, y_train, X_test, y_test,
                        L=DEFAULT_LAYERS, n_qubits=DEFAULT_QUBITS,
                        restarts=3, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR,
                        noise=0.0, verbose=True):
    # Ensure at least one restart so we always produce a history/params
    restarts = max(1, int(restarts))

    best_acc = -np.inf
    best_history = None
    best_params = None
    for r in range(restarts):
        print(f"\n--- Restart {r+1}/{restarts} ---")
        params_init = 0.05 * np.random.randn(L, n_qubits)
        params, history = train(params_init, circuit, X_train, y_train, X_test, y_test,
                                epochs=epochs, lr=lr, noise=noise, verbose=verbose)
        final_acc = float(history["test_acc"][-1])
        print(f"Restart {r+1} final test acc: {final_acc:.4f}")
        if final_acc > best_acc:
            best_acc = final_acc
            best_history = history
            best_params = params

    # Safety net: if something went wrong and best_history is still None, do one training pass
    if best_history is None:
        print("Warning: no successful training found during restarts. Running one training pass as fallback.")
        params_init = 0.05 * np.random.randn(L, n_qubits)
        best_params, best_history = train(params_init, circuit, X_train, y_train, X_test, y_test,
                                          epochs=epochs, lr=lr, noise=noise, verbose=verbose)

    return best_params, best_history

# ---------------------------
# PLOTTING
# ---------------------------
def plot_decision_boundary_model(model_predict_proba, title, filename, X_train=None, y_train=None, bounds=3.0, grid_n=200):
    xx, yy = np.meshgrid(np.linspace(-bounds, bounds, grid_n), np.linspace(-bounds, bounds, grid_n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Ensure model_predict_proba accepts array of points
    probs = model_predict_proba(grid)
    Z = probs.reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.9)
    plt.colorbar(label="Prob(class=1)")
    if X_train is not None:
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, edgecolors='k', linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved", path)

def plot_training(history, filename_prefix):
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.tight_layout()
    p1 = os.path.join(RESULT_DIR, filename_prefix + "_loss.png")
    plt.savefig(p1, dpi=150)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(history["test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.tight_layout()
    p2 = os.path.join(RESULT_DIR, filename_prefix + "_acc.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print("Saved", p1, "and", p2)

# ---------------------------
# BASELINES (SVM and MLP)
# ---------------------------
def classical_svm(X_train, y_train, X_test, y_test):
    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, acc

def classical_mlp(X_train, y_train, X_test, y_test, seed=42):
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=seed)
    mlp.fit(X_train, y_train)
    acc = mlp.score(X_test, y_test)
    return mlp, acc

# ---------------------------
# EXPRESSIBILITY metric (pairwise fidelity stats)
# ---------------------------
def compute_expressibility(n_qubits, L, n_samples=60, rng_seed=0):
    rnd = np.random.RandomState(rng_seed)
    state_qnode = make_state_qnode(n_qubits=n_qubits)
    params_list = []
    for _ in range(n_samples):
        params = 2 * np.pi * rnd.randn(L, n_qubits) * 0.1
        state = state_qnode(params, np.zeros(n_qubits))
        params_list.append(np.array(state))

    mats = np.array(params_list)
    n = len(mats)
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            s = np.vdot(mats[i], mats[j])
            vals.append(np.abs(s)**2)
    vals = np.array(vals)
    plt.figure(figsize=(5,3))
    plt.hist(vals, bins=25)
    plt.xlabel("Pairwise fidelity")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f"expressibility_L{L}_q{n_qubits}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    stats = {"mean": float(vals.mean()), "std": float(vals.std()), "median": float(np.median(vals))}
    print("Saved expressibility histogram:", path)
    return stats, path

# ---------------------------
# INTERPRETABILITY: sweep features
# ---------------------------
def interpretability_sweeps(params, circuit, fixed_values=[-1.0, 0.0, 1.0], noise=0.0, bounds=3.0, n_points=300):
    xs = np.linspace(-bounds, bounds, n_points)
    for fixed in fixed_values:
        probs = []
        for x1 in xs:
            x = np.array([x1, fixed])
            probs.append(expval_to_prob(float(circuit(params, x, noise))))
        plt.figure()
        plt.plot(xs, probs)
        plt.xlabel("x1")
        plt.ylabel("P(class=1)")
        plt.title(f"Sweep x1 (x2 fixed={fixed}, noise={noise})")
        path = os.path.join(RESULT_DIR, f"sweep_x1_fixed_{str(fixed).replace('.','p')}_noise{str(noise).replace('.','p')}.png")
        plt.grid(True)
        plt.savefig(path, dpi=150)
        plt.close()
        print("Saved", path)

        probs = []
        for x2 in xs:
            x = np.array([fixed, x2])
            probs.append(expval_to_prob(float(circuit(params, x, noise))))
        plt.figure()
        plt.plot(xs, probs)
        plt.xlabel("x2")
        plt.ylabel("P(class=1)")
        plt.title(f"Sweep x2 (x1 fixed={fixed}, noise={noise})")
        path = os.path.join(RESULT_DIR, f"sweep_x2_fixed_{str(fixed).replace('.','p')}_noise{str(noise).replace('.','p')}.png")
        plt.grid(True)
        plt.savefig(path, dpi=150)
        plt.close()
        print("Saved", path)

# ---------------------------
# LAYER SWEEP (missing previously)
# ---------------------------
def layer_sweep(n_qubits=DEFAULT_QUBITS, layers_list=[1,2,3], epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, noise=0.0, restarts=DEFAULT_RESTARTS):
    """Sweep over different numbers of variational layers, train QML for each, save metrics."""
    (X_train, X_test, y_train, y_test), scaler = load_data()
    results = {}
    for L in layers_list:
        print(f"\n=== Layer sweep: L={L} ===")
        circuit = make_qnode(n_qubits=n_qubits)
        params_clean, history_clean = train_with_restarts(circuit, X_train, y_train, X_test, y_test,
                                                          L=L, n_qubits=n_qubits, restarts=restarts,
                                                          epochs=epochs, lr=lr, noise=0.0, verbose=True)
        acc_clean, _, _ = evaluate_and_save_classification(circuit, params_clean, X_test, y_test, prefix=f"layers_L{L}_clean", noise=0.0)
        # compute expressibility for this L
        expr_stats, expr_path = compute_expressibility(n_qubits=n_qubits, L=L, n_samples=40, rng_seed=42)
        results[L] = {"clean_acc": acc_clean, "expr_stats": expr_stats, "expr_path": os.path.basename(expr_path)}
        # save training plots
        plot_training(history_clean, f"layers_L{L}_clean")
        # save decision boundary for clean
        model_predict_clean = lambda Xgrid, c=circuit, p=params_clean: predict_prob(c, p, Xgrid, 0.0)
        plot_decision_boundary_model(model_predict_clean, f"QML L={L} (clean)", f"qml_db_layers_L{L}.png", X_train=X_train, y_train=y_train)

    # write layer sweep summary
    with open(os.path.join(RESULT_DIR, "layer_sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved layer sweep summary to", os.path.join(RESULT_DIR, "layer_sweep_summary.json"))
    return results

# ---------------------------
# NOISE SWEEP (missing previously)
# ---------------------------
def noise_sweep(n_qubits=DEFAULT_QUBITS, L=DEFAULT_LAYERS, noise_levels=[0.0, 0.01, 0.03, 0.05], epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, restarts=DEFAULT_RESTARTS):
    """Sweep over different noise levels and train QML for each noise level (training uses noise during circuit)."""
    (X_train, X_test, y_train, y_test), scaler = load_data()
    circuit = make_qnode(n_qubits=n_qubits)
    results = {}
    for nlev in noise_levels:
        print(f"\n=== Noise sweep: noise={nlev} ===")
        params_noisy, history_noisy = train_with_restarts(circuit, X_train, y_train, X_test, y_test,
                                                          L=L, n_qubits=n_qubits, restarts=restarts,
                                                          epochs=epochs, lr=lr, noise=nlev, verbose=True)
        acc_noisy, _, _ = evaluate_and_save_classification(circuit, params_noisy, X_test, y_test, prefix=f"noise_n{str(nlev).replace('.','p')}", noise=nlev)
        # decision boundary using noisy evaluation
        model_predict_noisy = lambda Xgrid, c=circuit, p=params_noisy, n=nlev: predict_prob(c, p, Xgrid, n)
        plot_decision_boundary_model(model_predict_noisy, f"QML noise={nlev}", f"qml_db_noise_{str(nlev).replace('.','p')}.png", X_train=X_train, y_train=y_train)
        results[str(nlev)] = {"noisy_acc": acc_noisy}
        plot_training(history_noisy, f"noise_n{str(nlev).replace('.','p')}")

    with open(os.path.join(RESULT_DIR, "noise_sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved noise sweep summary to", os.path.join(RESULT_DIR, "noise_sweep_summary.json"))
    return results

# ---------------------------
# SINGLE RUN that trains QML (clean & noisy), baselines, and writes summary + metrics.json
# ---------------------------
def run_full_experiment(n_qubits=DEFAULT_QUBITS, L=DEFAULT_LAYERS,
                        epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, restarts=DEFAULT_RESTARTS, noise_level=DEFAULT_NOISE, quick=False):
    start_time = time.time()
    (X_train, X_test, y_train, y_test), scaler = load_data()
    circuit = make_qnode(n_qubits=n_qubits)

    if quick:
        epochs = max(3, min(epochs, 10))
        restarts = 1

    print("\n>>> Training QML (clean)")
    params_clean, history_clean = train_with_restarts(circuit, X_train, y_train, X_test, y_test,
                                                      L=L, n_qubits=n_qubits, restarts=restarts,
                                                      epochs=epochs, lr=lr, noise=0.0, verbose=True)

    # Defensive: history_clean should not be None after train_with_restarts, but check anyway
    if history_clean is None:
        print("Warning: training history for clean run is None — skipping training plots.")
    else:
        plot_training(history_clean, "qml_clean")

    model_predict_clean = lambda Xgrid, c=circuit, p=params_clean: predict_prob(c, p, Xgrid, 0.0)
    plot_decision_boundary_model(model_predict_clean, "QML (clean)", "qml_db_clean.png", X_train=X_train, y_train=y_train)
    np.save(os.path.join(RESULT_DIR, f"params_clean_L{L}.npy"), params_clean)
    acc_clean, cm_clean, clr_clean = evaluate_and_save_classification(circuit, params_clean, X_test, y_test, prefix="qml_clean", noise=0.0)


    print("\n>>> Training QML (noisy)")
    params_noisy, history_noisy = train_with_restarts(circuit, X_train, y_train, X_test, y_test,
                                                      L=L, n_qubits=n_qubits, restarts=restarts,
                                                      epochs=epochs, lr=lr, noise=noise_level, verbose=True)
    plot_training(history_noisy, "qml_noisy")
    model_predict_noisy = lambda Xgrid, c=circuit, p=params_noisy, n=noise_level: predict_prob(c, p, Xgrid, n)
    plot_decision_boundary_model(model_predict_noisy, f"QML (noise={noise_level})", "qml_db_noisy.png", X_train=X_train, y_train=y_train)
    np.save(os.path.join(RESULT_DIR, f"params_noisy_L{L}_n{str(noise_level).replace('.','p')}.npy"), params_noisy)
    acc_noisy, cm_noisy, clr_noisy = evaluate_and_save_classification(circuit, params_noisy, X_test, y_test, prefix="qml_noisy", noise=noise_level)

    print("\n>>> Classical SVM baseline")
    svm_clf, svm_acc = classical_svm(X_train, y_train, X_test, y_test)
    print("SVM test acc:", svm_acc)
    model_predict_svm = lambda Xgrid, clf=svm_clf: clf.predict_proba(Xgrid)[:, 1]
    plot_decision_boundary_model(model_predict_svm, "SVM baseline", "svm_db.png", X_train=X_train, y_train=y_train)
    svm_preds = svm_clf.predict(X_test)
    with open(os.path.join(RESULT_DIR, "svm_report.txt"), "w") as f:
        f.write(f"SVM Accuracy: {svm_acc:.4f}\n\n")
        f.write(classification_report(y_test, svm_preds, digits=4))
    cm = confusion_matrix(y_test, svm_preds)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.matshow(cm, cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='black')
    plt.title(f"SVM Confusion (acc={svm_acc:.3f})")
    plt.savefig(os.path.join(RESULT_DIR, "svm_confusion.png"), dpi=150)
    plt.close()

    print("\n>>> Classical MLP baseline")
    mlp_clf, mlp_acc = classical_mlp(X_train, y_train, X_test, y_test)
    print("MLP test acc:", mlp_acc)
    model_predict_mlp = lambda Xgrid, clf=mlp_clf: clf.predict_proba(Xgrid)[:, 1]
    plot_decision_boundary_model(model_predict_mlp, "MLP baseline", "mlp_db.png", X_train=X_train, y_train=y_train)
    mlp_preds = mlp_clf.predict(X_test)
    with open(os.path.join(RESULT_DIR, "mlp_report.txt"), "w") as f:
        f.write(f"MLP Accuracy: {mlp_acc:.4f}\n\n")
        f.write(classification_report(y_test, mlp_preds, digits=4))
    cm = confusion_matrix(y_test, mlp_preds)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.matshow(cm, cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='black')
    plt.title(f"MLP Confusion (acc={mlp_acc:.3f})")
    plt.savefig(os.path.join(RESULT_DIR, "mlp_confusion.png"), dpi=150)
    plt.close()

    print("\n>>> Interpretability sweeps (clean params)")
    interpretability_sweeps(params_clean, circuit, fixed_values=[-1.0, 0.0, 1.0], noise=0.0)

    print("\n>>> Expressibility (random parameter) analysis")
    expr_stats, expr_path = compute_expressibility(n_qubits=n_qubits, L=L, n_samples=60, rng_seed=42)

    total_runtime = time.time() - start_time
    summary_path = os.path.join(RESULT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"QML clean final acc: {acc_clean:.4f}\n")
        f.write(f"QML noisy final acc: {acc_noisy:.4f}\n")
        f.write(f"SVM acc: {svm_acc:.4f}\n")
        f.write(f"MLP acc: {mlp_acc:.4f}\n")
        f.write(f"Total runtime (s): {total_runtime:.1f}\n")
    print("Saved summary to", summary_path)

    metrics = {
        "qml_clean_acc": float(acc_clean),
        "qml_noisy_acc": float(acc_noisy),
        "svm_acc": float(svm_acc),
        "mlp_acc": float(mlp_acc),
        "expr_stats": expr_stats,
        "expr_path": os.path.basename(expr_path),
        "params": {"n_qubits": n_qubits, "layers": L, "epochs": epochs, "lr": lr, "restarts": restarts, "noise": noise_level},
        "runtime_s": total_runtime
    }
    with open(os.path.join(RESULT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json")

    print("All done. Results saved in", RESULT_DIR)

# ---------------------------
# CLI parsing + interactive prompts when appropriate
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="QML classifier experiments")
    parser.add_argument("--run-all", action="store_true", help="Run full experiment (clean/noisy + baselines + interpretability)")
    parser.add_argument("--layer-sweep", action="store_true", help="Run expressibility (layer) sweep")
    parser.add_argument("--noise-sweep", action="store_true", help="Run noise sweep")
    parser.add_argument("--single-run", action="store_true", help="Train single QML run (clean+noisy) and baselines")
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--quick", action="store_true", help="Run short quick experiment for UI testing")
    parser.add_argument("--task", type=str, default="all", help="Optional: 'all', 'layers', 'noise', 'single'")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # If running interactively (tty) and the user didn't set particular CLI args, prompt them.
    # If non-interactive (spawned by frontend), do not prompt and use defaults or provided CLI args.
    n_qubits = args.n_qubits if args.n_qubits is not None else prompt_if_tty("n_qubits (number of qubits)", DEFAULT_QUBITS, int)
    layers = args.layers if args.layers is not None else prompt_if_tty("layers (variational layers)", DEFAULT_LAYERS, int)
    epochs = args.epochs if args.epochs is not None else prompt_if_tty("epochs (training epochs)", DEFAULT_EPOCHS, int)
    lr = args.lr if args.lr is not None else prompt_if_tty("lr (learning rate)", DEFAULT_LR, float)
    restarts = args.restarts if args.restarts is not None else prompt_if_tty("restarts (random restarts)", DEFAULT_RESTARTS, int)
    noise = args.noise if args.noise is not None else prompt_if_tty("noise (depolarizing prob)", DEFAULT_NOISE, float)

    # Run tasks based on flags or default to run_all behavior
    if args.run_all or args.task == "all" or args.single_run:
        run_full_experiment(n_qubits=n_qubits, L=layers, epochs=epochs, lr=lr, restarts=restarts, noise_level=noise, quick=args.quick)

    if args.layer_sweep or args.task == "layers":
        # Use a reasonable default set of layer values if not provided
        layers_list = [1, 2, 3, 4]
        layer_sweep(n_qubits=n_qubits, layers_list=layers_list, epochs=epochs, lr=lr, noise=0.0, restarts=restarts)

    if args.noise_sweep or args.task == "noise":
        noise_levels = [0.0, 0.01, 0.03, 0.05, 0.1]
        noise_sweep(n_qubits=n_qubits, L=layers, noise_levels=noise_levels, epochs=epochs, lr=lr, restarts=restarts)
