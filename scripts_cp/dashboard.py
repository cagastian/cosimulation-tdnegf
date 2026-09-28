model = "kitaev_test"

import qutip as qt
import numpy as np
import re
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:            # running in a notebook / REPL
    SCRIPT_DIR = Path.cwd()

DATA_DIR = SCRIPT_DIR / f"data_{model}" # "data_disconnect"     # <- change "data" to whatever folder name you want
DATA_DIR.mkdir(exist_ok=True)      # creates it if missing, no error if already there

N = 14  # The construction of the lattice could be automated, but I won't do it unless it's useful 

# ==============================================================================================
# =     Functions for tensor product for the generation of operators in different sites
# ==============================================================================================

def op_at(op, i, N):
    """single-site operator op acting on site i, identity elsewhere"""
    return qt.tensor([op if k == i else qt.qeye(2) for k in range(N)])

def two_site(op_i, i, op_j, j, N):
    ops = [qt.qeye(2)] * N
    ops[i], ops[j] = op_i, op_j
    return qt.tensor(ops)

def plaquette_operator(op_i:dict, N):
    ops = [qt.qeye(2)] * N
    for op,sites in op_i.items():
        for site in sites:
            ops[site]=pauli[op]
    return qt.tensor(ops)

#Plaquettes
def W():
    #Plaquettes {0,1,2}
    W_plaquettes = []
    W = []
    n=N-1
    print("Generating plaquette operators (3)")
    for i in range(3): #Could be generalized for more plaquettes, this part isn't as important.
        W_dict = {
            'y': [0 + 2 * i, n - 2 - 2 * i],
            'z': [1 + 2 * i, n - 1 - 2 * i],
            'x': [2 + 2 * i, n - 0 - 2 * i]
        }
        W_plaquettes.append(W_dict)
        W.append(plaquette_operator(W_plaquettes[i],N))
    return W

pauli = {'x': qt.sigmax(), 'y': qt.sigmay(), 'z': qt.sigmaz()} #Pauli operators 
Sx,Sy,Sz=0,0,0 #Making the general S_σ for the entire lattice
for k in range(N):
    Sx+=op_at(qt.sigmax(),k,N)
    Sy+=op_at(qt.sigmay(),k,N)
    Sz+=op_at(qt.sigmaz(),k,N)

# ==============================================================================================
# =     Generating hamiltonians based on neighbour table for hexagonal lattice with 3 full  
# =     hexagons (could be generalized for N=6+4*n_hex easily)
# =     Heisenberg: list of (i, j)
# =     Kitaev:     list of (i, j, 'x'|'y'|'z')
# ==============================================================================================

bonds_kitaev = [(0, 1, 'x'),(2, 3, 'x'),(4, 5, 'x'),(7, 8, 'x'),( 9, 10, 'x'),(11, 12, 'x'), 
                (1, 2, 'y'),(3, 4, 'y'),(5, 6, 'y'),(8, 9, 'y'),(10, 11, 'y'),(12, 13, 'y'), 
                (0, 13,'z'), (2,11,'z'),(4, 9, 'z'),(6, 7, 'z'),]

def Hamiltonian_kitaev(*,J=(1,1,1)):
    Jx,Jy,Jz = J
    J_kitaev={'x': -Jx, 'y': -Jy, 'z': -Jz}

    H_kitaev = sum(
        J_kitaev[a] * two_site(pauli[a], i,pauli[a], j,N)
        for i, j, a in bonds_kitaev
    )
    return H_kitaev

def Hamiltonian_Heisenberg(*,J=(1,1,1)):
    """
    Constructs the Heisenberg spin Hamiltonian for a given lattice.
    ----------
    J : Coupling constants (Jx, Jy, Jz).
        AFM (Antiferromagnetic): J < 0 
        FM  (Ferromagnetic):     J > 0
    """
    Jx,Jy,Jz=J

    J_heisenberg = {'x': -Jx, 'y': -Jy, 'z': -Jz}
    H_heisenberg = sum(
        J_heisenberg[a] * two_site(pauli[a], i,pauli[a], j,N)
        for a,_ in J_heisenberg.items()
        for i, j,_ in bonds_kitaev
    )
    return H_heisenberg

def THETA(t,t0,*,k=0.01,tol=1e-5):
    x = k*(t0-t)
    sig = 0.5*((x/(1+np.abs(x)))+1)
    return sig

# ==============================================================================================
# =     Dynamics
# ==============================================================================================

def main_dynamics(*, hamiltonian=None, Jd=0.001, n_periods=10,n_disconnect=8,omega0=0.01, ops, save_prefix=None, n_wp=3):
    
    if hamiltonian is None:
        hamiltonian = Hamiltonian_kitaev()

    Jex, theta = 0.1, np.pi/8
    Hk = Jex * hamiltonian

    # initial state: equilibrated with the tilted field at t=0
    H0 = Hk + Jd*(np.sin(theta)*Sx + np.cos(theta)*Sz)

    print("Calculating groundstate for t=0")
    _, psi0 = H0.groundstate(sparse=True, tol=1e-7, maxiter=50000)

    T = 2*np.pi/omega0
    tf = n_periods*T
    dt= 1.25
    n_sample = int(tf/dt)
    tlist1 = np.linspace(0, tf, n_sample)

    if n_disconnect>n_periods:
        print("Disconnection has to be before the total time, setting n_disconnect=8")
        n_disconnect=8

    t0 = n_disconnect*T 

    A = Jd*np.sin(theta)
    H = [Hk,
         [Sz, lambda t: THETA(t,t0,k=0.1)*Jd*np.cos(theta)],
         [Sx, lambda t: THETA(t,t0,k=0.1)*A*np.cos(omega0*t)],
         [Sy, lambda t: THETA(t,t0,k=0.1)*A*np.sin(omega0*t)]]

    print("Solving Schrodinger's Eq")
    res = qt.sesolve(H, psi0, tlist1, e_ops=ops,
                    options={"progress_bar": "enhanced", "store_states": True})

    if save_prefix is not None:
        save_result(res, save_prefix, N=N, n_wp=n_wp,
                    Jd=Jd, omega0=omega0, Jex=Jex, theta=theta,out_dir=DATA_DIR)

    return res

def save_result(res, prefix, *, N, n_wp, out_dir=DATA_DIR, **params):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    base = out_dir / prefix

    t = np.asarray(res.times)
    e = np.asarray(res.expect).real

    # --- Save Expectation Values & Metadata ---
    np.savez_compressed(base.with_suffix(".npz"), times=t, expect=e, N=N, n_wp=n_wp, **params)

    # --- Save Full State Trajectory ---
    if hasattr(res, 'states') and res.states:
        # Flatten state kets to shape (n_times, 2^N)
        state_mat = np.array([psi.full().ravel() for psi in res.states], dtype=np.complex128)
        np.savez_compressed(out_dir / f"{prefix}_states.npz", states=state_mat)

    # --- CSV (wide form) ---
    axis_labels = ['x', 'y', 'z']
    spin = e[:3*N].reshape(N, 3, -1)
    wide_cols = {f"sigma_{a}_site{i}": spin[i, a] for i in range(N) for a, a_lbl in enumerate(axis_labels)}
    for p in range(n_wp):
        wide_cols[f"W_{p}"] = e[3*N + p]

    pd.DataFrame({"t": t, **wide_cols}).to_csv(base.parent / f"{base.name}_wide.csv", index=False)

def export_rho_for_julia_gmn(res,sites_subsystem,time_indices=None,target_times=None,prefix_julia="M2",theta=0.0,out_dir=DATA_DIR / "GMN_results",):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = np.array(res.times)

    # 1. Determine which time step indices to export
    if target_times is not None:
        # Find closest simulation step index for each requested physical time value
        selected_indices = [np.argmin(np.abs(times - t_val)) for t_val in target_times]
    elif time_indices is not None:
        if isinstance(time_indices, slice):
            selected_indices = list(range(*time_indices.indices(len(res.states))))
        else:
            selected_indices = list(time_indices)
    else:
        # Default: export all time steps
        selected_indices = list(range(len(res.states)))

    print(f"Exporting density matrices for {len(selected_indices)} time slices...")

    # 2. Export matrices (j_idx starts at 1 to align with Julia's `for j in 1:N` loop)
    for j_idx, step_idx in enumerate(selected_indices, start=1):
        psi = res.states[step_idx]
        t_val = times[step_idx]

        # Partial trace to get reduced density matrix
        rho_sub = psi.ptrace(sites_subsystem)
        mat = rho_sub.full()

        
        # Filename matching Julia format: "M2_{j}_1_theta{ind1}_r.txt"
        
        filename_base = f"{prefix_julia}_{j_idx}_1_theta{theta}"


        np.savetxt(out_dir / f"{filename_base}_r.txt", mat.real, fmt="%.18e", delimiter="\t")
        np.savetxt(out_dir / f"{filename_base}_i.txt", mat.imag, fmt="%.18e", delimiter="\t")

        print(f"  [Slice j={j_idx}] Saved t = {t_val:.3f} (step index {step_idx}) -> {filename_base}_*.txt")

    print(f"Successfully exported files to: {out_dir}")

#==============================================================================================
#=   PLOTTER
#==============================================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from qutip.entropy import negativity

IMG_DIR = DATA_DIR / "imag"
IMG_DIR_WP = DATA_DIR / "imag_wp"
IMG_DIR.mkdir(parents=True, exist_ok=True)
from types import SimpleNamespace  # already imported in your script; harmless repeat

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # Formats scientific notation numbers in CM
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.ymargin": 0.4,
    "axes.formatter.useoffset": True,
    'legend.frameon': False,
    'legend.handlelength': 0.35,
    'legend.labelspacing': 0.2,
    'legend.columnspacing': 0.3,
    'legend.handletextpad':0.4,
})

def panel_S_tot(ax, ctx):
    """Column: total spin components <S^alpha_tot>/N."""
    e = np.array(ctx.res.expect).real            # (n_ops, n_times)
    t = ctx.res.times
    spin = e[:3 * ctx.N].reshape(ctx.N, 3, -1)   # (site, alpha, time)
    S_tot = spin.sum(axis=0)

    axis_labels = ['x', 'y', 'z']
    colors = ['blue', 'green', 'red']

    for a in range(3):
        i = 2 - a                                 # plot z, y, x (same order as before)
        lbl = fr'$\alpha = {axis_labels[i]}$' if a == 2 else fr'$\alpha ={axis_labels[i]}$'
        ax.plot(t, S_tot[i] / ctx.N, color=colors[i], label=lbl,lw=0.75)

    ax.set_title(r'$\langle\hat{S}^{\alpha}_{\mathrm{tot}}\rangle/N$')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    return ax


def panel_Wp(ax, ctx):
    """Column: plaquette operators <W_p>."""
    e = np.array(ctx.res.expect).real
    t = ctx.res.times

    lw_list = [1.0, 1.0, 1.2]
    color_wp = ['gray', 'darkorange', 'black']
    style_line = ["-", "-", "dotted"]

    for p in range(ctx.n_wp):
        wp = e[3 * ctx.N + p]                     # assumes e_ops = [3N spin ops] + [W_p]
        ax.plot(t, wp,
                color=color_wp[p % len(color_wp)],
                ls=style_line[p % len(style_line)],
                lw=lw_list[p % len(lw_list)],
                label=fr'$p={p}$')

    #ax.axhline(y=1.0, color='k', ls='--', lw=0.6, alpha=0.5)
    ax.set_ylim(-0.1, 1.3)
    ax.set_title(r'$\langle \hat{W}_p \rangle$')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    return ax


def panel_E_N(ax, ctx, color="black"):
    """Column: bipartite logarithmic negativity E_N, read from the CSVs."""
    for label, csv_file in (ctx.csv_dict or {}).items():
        path = Path(csv_file)
        if not path.is_absolute():
            path = Path(ctx.csv_dir) / csv_file
        if not path.exists():
            print(f"[panel_E_N] missing: {path}")
            continue

        df = pd.read_csv(path)
        ax.plot(df["t"].to_numpy(), df["E_N"].to_numpy(),
                color=color, lw=1.0, label=r"$\mathcal{N}$")

    ax.set_title(r'$\mathcal{N}$')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    return ax


def panel_GMN(ax, ctx, ylim=None):
    """Column: genuine multipartite negativity, markers at t_targets."""
    red_styles = [
        {"color": "#f86503", "marker": "o"},   # crimson circle
        {"color": "#f86503", "marker": "s"},   # dark red square
        {"color": "#f86503", "marker": "^"},   # light coral triangle
    ]
    red_axis_color = "black"

    t_targets = ctx.t_targets
    if t_targets is None:
        t_targets = np.linspace(ctx.res.times[0], ctx.res.times[-1], 9)

    plotted = False
    for idx, (label, txt_file) in enumerate((ctx.gmn_dict or {}).items()):
        path = Path(txt_file)
        if not path.is_absolute():
            path = Path(ctx.gmn_dir) / txt_file
        if not path.exists():
            print(f"[panel_GMN] missing: {path}")
            continue

        gmn_vals = np.loadtxt(path)
        if len(gmn_vals) != len(t_targets):
            print(f"[panel_GMN] length mismatch for {path.name}: "
                  f"{len(gmn_vals)} values vs {len(t_targets)} target times")
            continue

        style = red_styles[idx % len(red_styles)]
        ax.plot(t_targets, gmn_vals,
                ls="--", marker=style["marker"], markersize=2.2,lw=0.6,
                color=style["color"], alpha=0.9, label=label)
        plotted = True

    ax.set_title(r'GMN')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

    if plotted:
        # scientific 10^-n multiplier on top, in the red palette
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        offset = ax.yaxis.get_offset_text()
        offset.set_color(red_axis_color)
        offset.set_fontsize(8)
        offset.set_ha("left")

        ax.tick_params(axis="y", colors=red_axis_color)
        ax.spines["left"].set_color(red_axis_color)

    if ylim is not None:
        ax.set_ylim(*ylim)
    return ax

# ----------------------------------------------------------------------------------------------
#  Registry + per-column cosmetics.  EDIT PANEL_ORDER TO REORDER THE COLUMNS.
# ----------------------------------------------------------------------------------------------

PANEL_FUNCS = {
    "S_tot": panel_S_tot,
    "Wp":    panel_Wp,
    "E_N":   panel_E_N,
    "GMN":   panel_GMN,
}

PANEL_ORDER = ["S_tot", "Wp", "E_N", "GMN"]

# legend: loc per column, None = no legend at all
PANEL_LEGEND = {
    "S_tot": dict(loc="upper right", ncol=1, frameon=False, reverse=True),
    "Wp":    dict(loc="lower left", ncol=1, frameon=False,edgecolor="white", framealpha=1.0),
    "E_N":   dict(loc="lower right", ncol=1, frameon=False),
    "GMN":   dict(loc="upper right", ncol=1, frameon=False),
}

# (x, y) of the (a)-(h) panel letter, in axes fraction
PANEL_LETTER_POS = {
    "S_tot": (0.15, 0.96),
    "Wp":    (0.15, 0.95),
    "E_N":   (0.15, 0.95),
    "GMN":   (0.15, 0.95),
}


def plot_dashboard_2x4_stacks(
    res_list,
    csv_dict_list,
    model_names=("QSL", "QAFI"),
    *,
    panel_order=None,
    gmn_files_dict=None,
    Jd=0.02,
    omega0=0.01,
    n_wp=3,
    t_targets_list=None,
    n_disconnect=8,
    gmn_ylim=None,
    gmn_dir=DATA_DIR / "data_GMN",
    csv_dir=DATA_DIR / "entanglement_csv",
    img_dir=IMG_DIR,
    save_prefix="dashboard_2x4_stacks",
    figsize=(7.8, 3.2),
    save=True,
):
    
    panel_order = list(panel_order) if panel_order is not None else list(PANEL_ORDER)
    ncols = len(panel_order)
    nrows = len(res_list)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 0.9,
    })

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,sharex=True, layout="constrained")
    axes = np.atleast_2d(axes)

    T = 2 * np.pi / omega0

    for r in range(nrows):
        res_r = res_list[r]

        gmn_dict_r = (gmn_files_dict[r]
                      if isinstance(gmn_files_dict, (list, tuple))
                      else gmn_files_dict)

        if t_targets_list is not None:
            t_targets_r = t_targets_list[r]
        else:
            t_targets_r = np.linspace(res_r.times[0], res_r.times[-1], 9)

        ctx = SimpleNamespace(
            res=res_r,
            csv_dict=csv_dict_list[r],
            gmn_dict=gmn_dict_r,
            t_targets=t_targets_r,
            N=N,
            n_wp=n_wp,
            omega0=omega0,
            Jd=Jd,
            csv_dir=csv_dir,
            gmn_dir=gmn_dir,
            row=r,
        )

        for c, key in enumerate(panel_order):
            ax = axes[r, c]

            if key == "GMN":
                PANEL_FUNCS[key](ax, ctx, ylim=gmn_ylim)
            else:
                PANEL_FUNCS[key](ax, ctx)

            # ---- shared cosmetics -------------------------------------------------
            ax.axvline(n_disconnect * T, color="gray", ls="--", lw=0.6, alpha=0.7)

            # panel letter, position taken from the column *type*, index from the grid
            letter = chr(ord("a") + (r * ncols + c))
            xpos, ypos = PANEL_LETTER_POS.get(key, (0.95, 0.20))
            ax.text(xpos, ypos, f"({letter})", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, fontweight="bold")

            # y-span clamp (keeps near-flat curves from being blown up)
            limite = 0.05 if key != "GMN" else 0.0001
            padding_ymax=0.2
            ymin, ymax = ax.get_ylim()
            if (ymax - ymin) < limite:
                ymid = 0.5 * (ymax + ymin)
                new_ymin, new_ymax = ymid - 0.05, ymid + 0.05
                if new_ymin < 0.0:
                    new_ymin, new_ymax = -0.01, 0.05
                ax.set_ylim(new_ymin, new_ymax+ymid*padding_ymax) 
            

            ax.margins(x=0.01, y=0.2)

            # legend on the top row only
            if r == 0:
                cfg = PANEL_LEGEND.get(key)
                if cfg:
                    cfg = dict(cfg)
                    reverse = cfg.pop("reverse", False)
                    h, l = ax.get_legend_handles_labels()
                    if h:
                        if reverse:
                            h, l = h[::-1], l[::-1]
                        ax.legend(h, l, **cfg)
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()
                ax.set_title("")

            # x axis on the bottom row only
            if r < nrows - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(labelbottom=True)
                ax.set_xlabel(r"Time  $(\hbar/\gamma)$")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))

        # ---- right-side bracket + model name (always on the last column) ----------
        if model_names is not None and r < len(model_names):
            ax_last = axes[r, -1]
            x_bracket, tick_len = 1.12, 0.03
            y_bottom, y_top = 0.05, 0.95

            ax_last.plot(
                [x_bracket - tick_len, x_bracket, x_bracket, x_bracket - tick_len],
                [y_top, y_top, y_bottom, y_bottom],
                color="black", lw=1.0,
                transform=ax_last.transAxes, clip_on=False,
            )
            ax_last.annotate(
                model_names[r],
                xy=(x_bracket + 0.04, 0.5), xycoords="axes fraction",
                fontsize=8, fontweight="bold", rotation=-90,
                ha="left", va="center", annotation_clip=False,
            )

    if save:
        out_dir = Path(img_dir) / "dashboards"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{save_prefix}_jd{Jd:.4f}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{save_prefix}_jd{Jd:.4f}.png", dpi=300,
                    bbox_inches="tight")

    plt.show()
    return fig

#=================================
#=      Entanglement
#=================================

# Compute logarithmic negativity for a subsystem index

def calculate_log_negativity_pure_bipartite(res, *, sites_A, Jd=0.001, omega0=0.01, save=True, out_dir=DATA_DIR,prefix=""):
    """
    Computes logarithmic negativity E_N(t) between subsystem A and the rest of the system (B)
    using the pure state Schmidt decomposition shortcut to avoid memory overflow.
    Saves the result to a CSV file if save=True.
    """
    E_N_t = []

    for psi in res.states:
        # 1. Trace out everything EXCEPT sites_A to get a small reduced density matrix
        rho_A = psi.ptrace(sites_A)

        # 2. Get the eigenvalues of rho_A
        evals = rho_A.eigenenergies()
        
        # 3. Clean up tiny numerical artifacts (e.g., -1e-17) and take the square root
        evals = np.clip(evals, 0, None)
        schmidt_coeffs = np.sqrt(evals)
        
        # 4. Calculate Logarithmic Negativity for a pure state: E_N = 2 * log2(sum(sqrt(lambda)))
        E_N = 2.0 * np.log2(np.sum(schmidt_coeffs))
        E_N_t.append(E_N)

    E_N_t = np.array(E_N_t)

    # --- Save to CSV ---
    if save:
        df = pd.DataFrame({
            "t": res.times,
            "E_N": E_N_t
        })
        
        csv_dir = Path(out_dir) / "entanglement_csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        pfx = f"{prefix}_" if prefix and not str(prefix).endswith(("_", "-")) else str(prefix)
        safe_name = f"{pfx}log_neg_bipartite_jd{Jd:.4f}.csv"
        df.to_csv(csv_dir / safe_name, index=False)
        
    return E_N_t

def load_log_negativity_csv(filename, in_dir=DATA_DIR):
    """
    Reads a saved logarithmic negativity CSV file.
    Returns the time array (t) and the logarithmic negativity array (E_N_t).
    
    Example:
        t, E_N_t = load_log_negativity_csv("log_neg_bipartite_6vs8_jd0.2500.csv")
    """
    csv_path = Path(in_dir) / "entanglement_csv" / filename
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    t = df["t"].to_numpy()
    E_N_t = df["E_N"].to_numpy()
    
    return t, E_N_t


#====================================
#=  Saving Files Per Jd 
#====================================

Jo           = 0.005
omega0       = 0.01
n_periods    = 10
n_disconnect = 5


Wp = W()
e_ops = [op_at(pauli[a], i, N) for i in range(N) for a in ('x','y','z')]
e_ops += [Wp[0], Wp[1], Wp[2]]

# Hamiltonians
h = [Hamiltonian_kitaev(J=(-1,-1,-1)),        # 0: Kitaev AFM
     Hamiltonian_kitaev(J=( 1, 1, 1)),        # 1: Kitaev FM 
    ]

model_labels = ["QSL AFM", "QSL FM"]

Compute_Everything = False

if Compute_Everything:
    res_list = []
    csv_dict_list = []

    Hexagon_sites=[2, 3, 4, 9, 10, 11]
    
    # Loop through all 3 models
    for idx, ham in enumerate(h):
        prefix = f"{model_labels[idx]}_jd{Jo:.4f}"
        
        # 1. Run simulation for the current Hamiltonian
        
        res = main_dynamics(hamiltonian=ham,ops=e_ops,Jd=Jo,n_periods=n_periods,n_disconnect=n_disconnect,n_wp=3,)
        res_list.append(res)
        # 2. Compute bipartite entanglement
        calculate_log_negativity_pure_bipartite(res, sites_A=Hexagon_sites, Jd=Jo, plot=False, save=True, prefix=f"Hex1_{idx}")

        t_targets = np.linspace(res.times[0], res.times[-1], 9)

        export_rho_for_julia_gmn(res,sites_subsystem=Hexagon_sites,prefix_julia=prefix,target_times=t_targets,theta=0.393)

        # 3. Store CSV mappings per model
        csv_dict_list.append({
            r"$\mathcal{{N}}$": f"Hex1_{idx}_log_neg_bipartite_jd{Jo:.4f}.csv",
        })
else: 
    pass

# Plot 3x4 Dashboard

gmn_files_dict = [{rf"$\mathcal{{N}}_\mathrm{{GMN}}^{{\,\,6,{i}}}$": f"Partition_6_{i}_GMN1_QSL_AFM.txt" for i in [6]},
                  {rf"$\mathcal{{N}}_\mathrm{{GMN}}^{{\,\,6,{i}}}$": f"Partition_6_{i}_GMN1_QSL_FM.txt" for i in [6]}]

fig = plot_dashboard_2x4_stacks(
    res_list=res_list,
    csv_dict_list=csv_dict_list,
    model_names=model_labels,
    gmn_files_dict=gmn_files_dict,
    Jd=Jo,
    omega0=0.01,
    n_wp=3,
    n_disconnect=n_disconnect,
    # panel_order=["E_N", "GMN", "S_tot", "Wp"],   # <- reorder columns here
    #gmn_ylim=(0.0005, 0.011),                     # <- your old hard-coded GMN limits
)
