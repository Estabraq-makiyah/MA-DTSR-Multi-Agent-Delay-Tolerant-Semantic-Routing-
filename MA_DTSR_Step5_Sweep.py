# ============================================================================
# MA-DTSR Simulation — Step 5: Full Parameter Sweep
# ============================================================================
# Run after Steps 1–4. Call main_step5(step1, step2, step3, step4) from Colab.
#
# What this step produces:
#   1. Full parameter grid sweep — N, TTL, epsilon, alpha, metric, 30 seeds
#   2. Four ablation studies isolating individual protocol contributions
#   3. Publication-ready figures (Figures 15–21) for Section 5 of the paper
#   4. Results tables (Table 1) for Section 5
#   5. All raw data saved to CSV for reproducibility
#
# Parameter grid (Section 5 of the paper):
#   N       : 100, 200, 300
#   TTL     : 10, 20, 30, 40
#   epsilon : 0.5, 1.0, 1.5
#   alpha   : 0.0, 0.01, 0.05
#   metric  : l1, l2, weighted
#   seeds   : 30 per combination
#
# Estimated runtime on Colab (N=300, 30 seeds): ~20–30 minutes
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from tqdm import tqdm
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ── Publication matplotlib style ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 9,
    'axes.labelsize' : 9,
    'axes.titlesize' : 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi'     : 150,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'axes.grid'      : False,
})


# ============================================================================
# SECTION 1: Parameter Grid and Display Styles
# ============================================================================

PARAM_GRID = {
    'N'      : [100, 200, 300],
    'TTL'    : [10, 20, 30, 40],
    'epsilon': [0.5, 1.0, 1.5],
    'alpha'  : [0.0, 0.01, 0.05],
    'metric' : ['l1', 'l2', 'weighted'],
    'n_seeds': 30,
}

RL_TRAIN_EPISODES = 300
RL_EVAL_EPISODES  = 30

STYLES = {
    'Epidemic'        : {'color': '#e63946', 'ls': '-',  'marker': 'o'},
    'RandomWalk'      : {'color': '#adb5bd', 'ls': '--', 'marker': 's'},
    'Heuristic-MADTSR': {'color': '#457b9d', 'ls': '--', 'marker': '^'},
    'RL-MADTSR'       : {'color': '#1d3557', 'ls': '-',  'marker': 'D'},
}
LABELS = {
    'Epidemic'        : 'Epidemic',
    'RandomWalk'      : 'Random walk',
    'Heuristic-MADTSR': 'MA-DTSR (heuristic)',
    'RL-MADTSR'       : 'MA-DTSR (RL)',
}


# ============================================================================
# SECTION 2: Network Builder
# ============================================================================

def build_network(step1_module, N, seed):
    """Build and warm-up a DisasterNetwork of size N."""
    cfg    = dict(step1_module.CONFIG)
    cfg['N'] = N
    net    = step1_module.DisasterNetwork(config=cfg, seed=seed)
    for _ in range(50):
        net.step()
    return net


def populate_network(net, step2_module, seed):
    """Populate WiSARD-calibrated descriptors and rebuild contact DB."""
    step2_module.populate_descriptors(net, seed=seed)
    step2_module.rebuild_contact_database(net)
    return net


# ============================================================================
# SECTION 3: Single-Configuration Runner
# ============================================================================

def run_one_config(net, step2_module, step3_module, step4_module,
                   N, ttl, epsilon, alpha, metric, seed,
                   rl_router=None):
    """
    Run all four protocols for one parameter combination.

    Parameters
    ----------
    rl_router : pre-trained RLRouter or None
                If None, a new one is created and trained.

    Returns
    -------
    records   : list of dict — one per protocol
    rl_router : trained RLRouter
    """
    rng     = np.random.default_rng(seed)
    records = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    baselines = [
        step3_module.EpidemicRouter(epsilon=epsilon, metric='l1'),
        step3_module.RandomWalkRouter(epsilon=epsilon, metric='l1'),
        step3_module.HeuristicRouter(
            epsilon=epsilon, metric=metric,
            beta=2.0, mode='softmin', alpha=alpha),
    ]
    for proto in baselines:
        result = proto.run_episode(
            net, step2_module, ttl,
            np.random.default_rng(seed + abs(hash(proto.name)) % 10000),
            alpha=alpha)
        row = result.to_dict()
        row.update({'N': N, 'ttl': ttl, 'epsilon': epsilon,
                    'alpha': alpha, 'metric': metric, 'seed': seed})
        records.append(row)

    # ── RL protocol ───────────────────────────────────────────────────────────
    if rl_router is None:
        rl_router = step4_module.RLRouter(
            agents      = net.agents,
            epsilon     = epsilon,
            alpha       = alpha,
            beta        = 2.0,
            rho_start   = 0.8,
            rho_end     = 0.05,
            rho_decay   = 0.995,
            lr          = 0.05,
            gamma_q     = 0.9,
            alpha_merge = 0.3,
        )
        train_rng = np.random.default_rng(seed + 9999)
        for _ in range(RL_TRAIN_EPISODES):
            rl_router.run_episode(
                net, step2_module, ttl, train_rng,
                alpha=alpha, is_training=True)

    result = rl_router.run_episode(
        net, step2_module, ttl,
        np.random.default_rng(seed + 1234),
        alpha=alpha, is_training=False)
    row = result.to_dict()
    row.update({'N': N, 'ttl': ttl, 'epsilon': epsilon,
                'alpha': alpha, 'metric': metric, 'seed': seed})
    records.append(row)

    return records, rl_router


# ============================================================================
# SECTION 4: Full Sweep Runner
# ============================================================================

def run_full_sweep(step1_module, step2_module,
                   step3_module, step4_module,
                   param_grid=None, master_seed=42):
    """
    Run the complete parameter sweep.

    For each (N, epsilon, alpha, metric) combination:
      - Build one network per seed
      - Train one RL router on the middle TTL value
      - Evaluate all four protocols across all TTL values

    Returns
    -------
    pd.DataFrame — all raw episode results
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    Ns       = param_grid['N']
    TTLs     = param_grid['TTL']
    epsilons = param_grid['epsilon']
    alphas   = param_grid['alpha']
    metrics  = param_grid['metric']
    n_seeds  = param_grid['n_seeds']

    outer       = list(product(Ns, epsilons, alphas, metrics))
    total_outer = len(outer) * n_seeds

    print("Parameter grid:")
    print(f"  N       : {Ns}")
    print(f"  TTL     : {TTLs}")
    print(f"  epsilon : {epsilons}")
    print(f"  alpha   : {alphas}")
    print(f"  metric  : {metrics}")
    print(f"  seeds   : {n_seeds}")
    print(f"  Outer configs : {total_outer:,}")
    print(f"  Total episodes: {total_outer * len(TTLs) * 4:,}")
    print()

    all_records = []
    rng_master  = np.random.default_rng(master_seed)
    train_ttl_idx = len(TTLs) // 2   # train RL on middle TTL

    pbar = tqdm(total=total_outer,
                desc='Sweep', unit='config')

    for N, epsilon, alpha, metric in outer:
        for seed_idx in range(n_seeds):
            seed = int(rng_master.integers(0, 100_000))

            # Build and populate network
            net = build_network(step1_module, N, seed)
            populate_network(net, step2_module, seed)

            # Train RL router on middle TTL
            train_ttl = TTLs[train_ttl_idx]
            rl_router = step4_module.RLRouter(
                agents      = net.agents,
                epsilon     = epsilon,
                alpha       = alpha,
                beta        = 2.0,
                rho_start   = 0.8,
                rho_end     = 0.05,
                rho_decay   = 0.995,
                lr          = 0.05,
                gamma_q     = 0.9,
                alpha_merge = 0.3,
            )
            train_rng = np.random.default_rng(seed + 777)
            for _ in range(RL_TRAIN_EPISODES):
                rl_router.run_episode(
                    net, step2_module, train_ttl, train_rng,
                    alpha=alpha, is_training=True)

            # Evaluate all protocols across all TTL values
            for ttl in TTLs:
                records, _ = run_one_config(
                    net, step2_module, step3_module, step4_module,
                    N=N, ttl=ttl, epsilon=epsilon,
                    alpha=alpha, metric=metric, seed=seed,
                    rl_router=rl_router)
                all_records.extend(records)

            pbar.update(1)

    pbar.close()
    df = pd.DataFrame(all_records)
    print(f"\nSweep complete. {len(df):,} episode records.")
    return df


# ============================================================================
# SECTION 5: Aggregation
# ============================================================================

def aggregate(df, group_cols):
    """Aggregate raw results by group_cols with mean ± std for all metrics."""
    grp = df.groupby(group_cols)
    return grp.agg(
        success_rate = ('success',     'mean'),
        sr_std       = ('success',     'std'),
        mean_hops    = ('hops',        'mean'),
        std_hops     = ('hops',        'std'),
        mean_messages= ('messages',    'mean'),
        std_messages = ('messages',    'std'),
        mean_mae     = ('match_error', lambda x: x.dropna().mean()),
        std_mae      = ('match_error', lambda x: x.dropna().std()),
        mean_utility = ('utility',     'mean'),
        std_utility  = ('utility',     'std'),
        n            = ('success',     'count'),
    ).reset_index()


def ci95(std, n):
    """95% confidence interval half-width."""
    return 1.96 * std / np.sqrt(np.maximum(n, 1))


# ============================================================================
# SECTION 6: Publication Figures
# ============================================================================

def _style(name, key):
    return STYLES.get(name, {}).get(key, None)


def _add_legend(ax, protos=None):
    if protos is None:
        protos = list(STYLES.keys())
    handles = [
        mlines.Line2D([], [],
                      color=STYLES[p]['color'],
                      linestyle=STYLES[p]['ls'],
                      marker=STYLES[p]['marker'],
                      markersize=5, linewidth=1.8,
                      label=LABELS[p])
        for p in protos if p in STYLES
    ]
    ax.legend(handles=handles, fontsize=7)


def fig_success_vs_ttl(df, N=200, epsilon=1.0, alpha=0.01, metric='l1'):
    """Figure 15 — Success rate vs TTL."""
    sub = df[(df['N']==N)&(df['epsilon']==epsilon)&
             (df['alpha']==alpha)&(df['metric']==metric)]
    if sub.empty:
        print(f"Figure 15: no data for N={N}, ε={epsilon}, α={alpha}")
        return
    agg = aggregate(sub, ['protocol','ttl'])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for proto, grp in agg.groupby('protocol'):
        if proto not in STYLES: continue
        grp = grp.sort_values('ttl')
        sr  = grp['success_rate'] * 100
        ci  = ci95(grp['sr_std'], grp['n']) * 100
        ax.plot(grp['ttl'], sr,
                color=_style(proto,'color'), ls=_style(proto,'ls'),
                marker=_style(proto,'marker'))
        ax.fill_between(grp['ttl'], sr-ci, sr+ci,
                        alpha=0.12, color=_style(proto,'color'))
    ax.set_xlabel('TTL budget')
    ax.set_ylabel('Success rate (%)')
    ax.set_title(f'Success Rate vs TTL\n(N={N}, ε={epsilon}, α={alpha})')
    ax.set_ylim(0, 105)
    _add_legend(ax)
    fig.tight_layout()
    fname = f'fig15_success_vs_ttl_N{N}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure 15 saved: {fname}")


def fig_success_vs_N(df, ttl=20, epsilon=1.0, alpha=0.01, metric='l1'):
    """Figure 16 — Success rate vs N (scalability)."""
    sub = df[(df['ttl']==ttl)&(df['epsilon']==epsilon)&
             (df['alpha']==alpha)&(df['metric']==metric)]
    if sub.empty:
        print(f"Figure 16: no data for TTL={ttl}")
        return
    agg = aggregate(sub, ['protocol','N'])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for proto, grp in agg.groupby('protocol'):
        if proto not in STYLES: continue
        grp = grp.sort_values('N')
        sr  = grp['success_rate'] * 100
        ci  = ci95(grp['sr_std'], grp['n']) * 100
        ax.plot(grp['N'], sr,
                color=_style(proto,'color'), ls=_style(proto,'ls'),
                marker=_style(proto,'marker'))
        ax.fill_between(grp['N'], sr-ci, sr+ci,
                        alpha=0.12, color=_style(proto,'color'))
    ax.set_xlabel('Number of agents (N)')
    ax.set_ylabel('Success rate (%)')
    ax.set_title(f'Scalability: Success Rate vs N\n'
                 f'(TTL={ttl}, ε={epsilon}, α={alpha})')
    ax.set_ylim(0, 105)
    _add_legend(ax)
    fig.tight_layout()
    fname = f'fig16_success_vs_N_ttl{ttl}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure 16 saved: {fname}")


def fig_messages_vs_ttl(df, N=200, epsilon=1.0, alpha=0.01, metric='l1'):
    """Figure 17 — Message count vs TTL (efficiency)."""
    sub = df[(df['N']==N)&(df['epsilon']==epsilon)&
             (df['alpha']==alpha)&(df['metric']==metric)]
    if sub.empty:
        print(f"Figure 17: no data"); return
    agg = aggregate(sub, ['protocol','ttl'])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for proto, grp in agg.groupby('protocol'):
        if proto not in STYLES: continue
        grp = grp.sort_values('ttl')
        ax.plot(grp['ttl'], grp['mean_messages'],
                color=_style(proto,'color'), ls=_style(proto,'ls'),
                marker=_style(proto,'marker'))
    ax.set_xlabel('TTL budget')
    ax.set_ylabel('Mean messages (M)')
    ax.set_title(f'Message Count vs TTL\n(N={N}, ε={epsilon}, α={alpha})')
    _add_legend(ax)
    fig.tight_layout()
    fname = f'fig17_messages_vs_ttl_N{N}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure 17 saved: {fname}")


def fig_mae_vs_epsilon(df, N=200, ttl=20, alpha=0.01, metric='l1'):
    """Figure 18 — Semantic match quality (MAE) vs epsilon."""
    sub = df[(df['N']==N)&(df['ttl']==ttl)&
             (df['alpha']==alpha)&(df['metric']==metric)&
             (df['success']==1)]
    if sub.empty:
        print(f"Figure 18: no successful matches to plot MAE."); return
    agg = aggregate(sub, ['protocol','epsilon'])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for proto, grp in agg.groupby('protocol'):
        if proto not in STYLES: continue
        grp = grp.sort_values('epsilon')
        if grp['mean_mae'].isna().all(): continue
        ax.plot(grp['epsilon'], grp['mean_mae'],
                color=_style(proto,'color'), ls=_style(proto,'ls'),
                marker=_style(proto,'marker'))
    ax.set_xlabel('Admissibility threshold (ε)')
    ax.set_ylabel('MAE (semantic match error)')
    ax.set_title(f'Match Quality vs ε\n(N={N}, TTL={ttl}, α={alpha})')
    _add_legend(ax)
    fig.tight_layout()
    fname = f'fig18_mae_vs_epsilon_N{N}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure 18 saved: {fname}")


def fig_utility_vs_alpha(df, N=200, ttl=20, epsilon=1.0, metric='l1'):
    """Figure 19 — Mission utility vs staleness parameter alpha."""
    sub = df[(df['N']==N)&(df['ttl']==ttl)&
             (df['epsilon']==epsilon)&(df['metric']==metric)]
    if sub.empty:
        print(f"Figure 19: no data"); return
    agg = aggregate(sub, ['protocol','alpha'])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for proto, grp in agg.groupby('protocol'):
        if proto not in STYLES: continue
        grp = grp.sort_values('alpha')
        ax.plot(grp['alpha'], grp['mean_utility'],
                color=_style(proto,'color'), ls=_style(proto,'ls'),
                marker=_style(proto,'marker'))
    ax.set_xlabel('Staleness parameter (α)')
    ax.set_ylabel('Mean utility (U_s)')
    ax.set_title(f'Mission Utility vs α\n(N={N}, TTL={ttl}, ε={epsilon})')
    _add_legend(ax)
    fig.tight_layout()
    fname = f'fig19_utility_vs_alpha_N{N}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure 19 saved: {fname}")


def fig_metric_comparison(df, N=200, ttl=20, epsilon=1.0, alpha=0.01):
    """Figure 20 — Effect of similarity metric on RL-MADTSR."""
    sub = df[(df['N']==N)&(df['ttl']==ttl)&
             (df['epsilon']==epsilon)&(df['alpha']==alpha)&
             (df['protocol']=='RL-MADTSR')]
    if sub.empty:
        print("Figure 20: no RL-MADTSR data."); return
    agg = aggregate(sub, ['metric'])

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    fig.suptitle('Similarity Metric Comparison (RL-MADTSR)', fontsize=10)

    metrics_order  = ['l1', 'l2', 'weighted']
    metric_labels  = ['L1', 'Euclidean', 'Weighted L1']
    colours        = ['#1d3557', '#457b9d', '#2a9d8f']

    for ax, (col, std_col, ylabel) in zip(axes, [
        ('success_rate', 'sr_std',      'Success rate (%)'),
        ('mean_hops',    'std_hops',    'Mean hops (H)'),
        ('mean_utility', 'std_utility', 'Mean utility (U_s)'),
    ]):
        vals, errs = [], []
        for m in metrics_order:
            row = agg[agg['metric'] == m]
            if row.empty:
                vals.append(0); errs.append(0); continue
            v = float(row[col].values[0])
            s = float(row[std_col].values[0])
            n = float(row['n'].values[0])
            vals.append(v * 100 if col == 'success_rate' else v)
            errs.append(ci95(s, n) * 100 if col == 'success_rate'
                        else ci95(s, n))

        ax.bar(metric_labels, vals, color=colours,
               alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.errorbar(metric_labels, vals, yerr=errs,
                    fmt='none', color='black', capsize=3, linewidth=1)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fname = f'fig20_metric_comparison_N{N}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure 20 saved: {fname}")


# ============================================================================
# SECTION 7: Ablation Studies
# ============================================================================

def run_ablations(step1_module, step2_module,
                  step3_module, step4_module,
                  N=200, ttl=20, epsilon=1.0,
                  alpha=0.01, n_seeds=30, master_seed=55):
    """
    Four ablation studies — each isolates one design choice.

    A1: Self-organisation ON (heuristic) vs OFF (random walk)
    A2: Deterministic (eq.5) vs softmin (eq.6) next-hop
    A3: Staleness ON (alpha>0) vs OFF (alpha=0)
    A4: Cooperative exchange ON vs OFF (alpha_merge=0.3 vs 0.0)

    Returns
    -------
    dict of pd.DataFrame — keyed by ablation name
    """
    print("\nRunning ablation studies...")
    rng_master = np.random.default_rng(master_seed)
    results    = {}

    # ── A1: Self-organisation ─────────────────────────────────────────────────
    print("  A1: Self-organisation (heuristic vs random walk)...")
    records_a1 = []
    for _ in tqdm(range(n_seeds), desc='  A1', leave=False):
        seed = int(rng_master.integers(0, 100_000))
        net  = build_network(step1_module, N, seed)
        populate_network(net, step2_module, seed)

        for RouterClass, label in [
            (step3_module.HeuristicRouter,  'SelfOrg-ON'),
            (step3_module.RandomWalkRouter,  'SelfOrg-OFF'),
        ]:
            router = RouterClass(epsilon=epsilon)
            if hasattr(router, 'alpha'):
                router.alpha = alpha
            result = router.run_episode(
                net, step2_module, ttl,
                np.random.default_rng(seed), alpha=alpha)
            row = result.to_dict()
            row['variant'] = label
            records_a1.append(row)

    results['A1_self_org'] = pd.DataFrame(records_a1)

    # ── A2: Deterministic vs softmin ──────────────────────────────────────────
    print("  A2: Deterministic vs softmin next-hop...")
    records_a2 = []
    for _ in tqdm(range(n_seeds), desc='  A2', leave=False):
        seed = int(rng_master.integers(0, 100_000))
        net  = build_network(step1_module, N, seed)
        populate_network(net, step2_module, seed)

        for mode, label in [('deterministic', 'Deterministic (eq.5)'),
                             ('softmin',       'Softmin (eq.6)')]:
            router = step3_module.HeuristicRouter(
                epsilon=epsilon, alpha=alpha, beta=2.0, mode=mode)
            result = router.run_episode(
                net, step2_module, ttl,
                np.random.default_rng(seed), alpha=alpha)
            row = result.to_dict()
            row['variant'] = label
            records_a2.append(row)

    results['A2_selection'] = pd.DataFrame(records_a2)

    # ── A3: Staleness ON vs OFF ───────────────────────────────────────────────
    print("  A3: Staleness penalty (alpha=0 vs alpha>0)...")
    records_a3 = []
    for _ in tqdm(range(n_seeds), desc='  A3', leave=False):
        seed = int(rng_master.integers(0, 100_000))
        net  = build_network(step1_module, N, seed)
        populate_network(net, step2_module, seed)

        for a, label in [(0.0,  'No staleness (α=0)'),
                         (0.01, 'Staleness (α=0.01)'),
                         (0.05, 'Staleness (α=0.05)')]:
            router = step3_module.HeuristicRouter(
                epsilon=epsilon, alpha=a, beta=2.0)
            result = router.run_episode(
                net, step2_module, ttl,
                np.random.default_rng(seed), alpha=a)
            row = result.to_dict()
            row['variant'] = label
            records_a3.append(row)

    results['A3_staleness'] = pd.DataFrame(records_a3)

    # ── A4: Cooperative exchange ON vs OFF ────────────────────────────────────
    print("  A4: Cooperative weight exchange (alpha_merge=0.3 vs 0.0)...")
    records_a4 = []
    for _ in tqdm(range(n_seeds), desc='  A4', leave=False):
        seed = int(rng_master.integers(0, 100_000))
        net  = build_network(step1_module, N, seed)
        populate_network(net, step2_module, seed)

        for merge, label in [(0.3, 'CoopExchange-ON'),
                             (0.0, 'CoopExchange-OFF')]:
            rl = step4_module.RLRouter(
                agents=net.agents, epsilon=epsilon,
                alpha=alpha, alpha_merge=merge)
            train_rng = np.random.default_rng(seed + 42)
            for _ in range(RL_TRAIN_EPISODES):
                rl.run_episode(net, step2_module, ttl, train_rng,
                               alpha=alpha, is_training=True)
            result = rl.run_episode(
                net, step2_module, ttl,
                np.random.default_rng(seed + 100),
                alpha=alpha, is_training=False)
            row = result.to_dict()
            row['variant'] = label
            records_a4.append(row)

    results['A4_coop'] = pd.DataFrame(records_a4)
    print("  Ablations complete.")
    return results


def plot_ablations(ablation_results):
    """
    Figure 21 — Four-panel ablation study.

    Uses raw episode columns (success, utility) not aggregated names,
    computing means and CIs directly from the episode DataFrames.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('MA-DTSR Ablation Studies', fontsize=12)

    # Each spec: (key, title, group_col, raw_col, multiply_100, ylabel)
    ablation_specs = [
        ('A1_self_org',  'A1: Self-Organisation',
         'variant', 'success', True,  'Success rate (%)'),
        ('A2_selection', 'A2: Next-Hop Selection Rule',
         'variant', 'success', True,  'Success rate (%)'),
        ('A3_staleness', 'A3: Staleness Penalty (α)',
         'variant', 'utility', False, 'Mean utility (U_s)'),
        ('A4_coop',      'A4: Cooperative Exchange',
         'variant', 'success', True,  'Success rate (%)'),
    ]

    colours = ['#1d3557', '#457b9d', '#2a9d8f', '#e63946', '#f4a261']

    for ax, (key, title, grp_col, raw_col, pct, ylabel) in zip(
            axes.flatten(), ablation_specs):

        if key not in ablation_results:
            ax.set_visible(False)
            continue

        df       = ablation_results[key]
        grouped  = df.groupby(grp_col)[raw_col]
        means    = grouped.mean()
        stds     = grouped.std().fillna(0)
        ns       = grouped.count()
        cis      = ci95(stds, ns)

        if pct:
            means = means * 100
            cis   = cis   * 100

        variants = means.index.tolist()
        cols     = colours[:len(variants)]

        ax.bar(range(len(variants)), means.values,
               color=cols, alpha=0.85,
               edgecolor='white', linewidth=0.5, width=0.5)
        ax.errorbar(range(len(variants)), means.values,
                    yerr=cis.values,
                    fmt='none', color='black',
                    capsize=4, linewidth=1)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, fontsize=7,
                           rotation=15, ha='right')
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    plt.savefig('fig21_ablations.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 21 saved: fig21_ablations.png")


# ============================================================================
# SECTION 8: Results Table
# ============================================================================

def print_results_table(df, N=200, ttl=20,
                        epsilon=1.0, alpha=0.01, metric='l1'):
    """Print Table 1 — main results for a given parameter setting."""
    sub = df[(df['N']==N)&(df['ttl']==ttl)&
             (df['epsilon']==epsilon)&
             (df['alpha']==alpha)&(df['metric']==metric)]
    if sub.empty:
        print(f"  No data for N={N}, TTL={ttl}, ε={epsilon}, "
              f"α={alpha}, metric={metric}")
        return

    agg   = aggregate(sub, ['protocol'])
    order = ['Epidemic','RandomWalk','Heuristic-MADTSR','RL-MADTSR']

    print()
    print("=" * 75)
    print(f"  Table 1 — N={N}, TTL={ttl}, ε={epsilon}, "
          f"α={alpha}, metric={metric}")
    print("=" * 75)
    print(f"  {'Protocol':<22} {'SR(%)':>7} {'±':>5} "
          f"{'Hops':>7} {'Msgs':>7} {'MAE':>8} {'Utility':>9}")
    print("  " + "-" * 68)

    for proto in order:
        row = agg[agg['protocol'] == proto]
        if row.empty:
            continue
        r      = row.iloc[0]
        sr     = r['success_rate'] * 100
        sr_ci  = ci95(r['sr_std'], r['n']) * 100
        mae    = r['mean_mae'] if not pd.isna(r['mean_mae']) else 0.0
        label  = LABELS.get(proto, proto)
        print(f"  {label:<22} {sr:>6.1f}% {sr_ci:>5.1f} "
              f"{r['mean_hops']:>7.2f} {r['mean_messages']:>7.1f} "
              f"{mae:>8.3f} {r['mean_utility']:>9.4f}")

    print("=" * 75)


# ============================================================================
# SECTION 9: Main
# ============================================================================

def main_step5(step1_module, step2_module,
               step3_module, step4_module,
               param_grid=None,
               run_ablations_flag=True,
               master_seed=42):
    """
    Run the full parameter sweep and generate all publication figures.

    Parameters
    ----------
    step1_module      : MA_DTSR_Step1_Mobility
    step2_module      : MA_DTSR_Step2_Descriptors
    step3_module      : MA_DTSR_Step3_Baselines
    step4_module      : MA_DTSR_Step4_RL
    param_grid        : dict or None — uses PARAM_GRID if None
    run_ablations_flag: bool — set False to skip ablations
    master_seed       : int

    Returns
    -------
    df        : pd.DataFrame — all raw results
    ablations : dict of pd.DataFrame — ablation results
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    print("=" * 65)
    print("  MA-DTSR \u2014 Step 5: Full Parameter Sweep")
    print("=" * 65)
    print()

    # ── Phase 1: Full sweep ───────────────────────────────────────────────────
    print("Phase 1: Running full parameter sweep...")
    df = run_full_sweep(
        step1_module, step2_module,
        step3_module, step4_module,
        param_grid=param_grid,
        master_seed=master_seed)

    df.to_csv('step5_full_results.csv', index=False)
    print(f"Raw results saved: step5_full_results.csv ({len(df):,} rows)")

    # ── Phase 2: Publication figures ──────────────────────────────────────────
    print("\nPhase 2: Generating publication figures...")
    fig_success_vs_ttl(df,  N=200, epsilon=1.0, alpha=0.01, metric='l1')
    fig_success_vs_N(df,    ttl=20, epsilon=1.0, alpha=0.01, metric='l1')
    fig_messages_vs_ttl(df, N=200, epsilon=1.0, alpha=0.01, metric='l1')
    fig_mae_vs_epsilon(df,  N=200, ttl=20, alpha=0.01, metric='l1')
    fig_utility_vs_alpha(df, N=200, ttl=20, epsilon=1.0, metric='l1')
    fig_metric_comparison(df, N=200, ttl=20, epsilon=1.0, alpha=0.01)

    # ── Phase 3: Results tables ───────────────────────────────────────────────
    print("\nPhase 3: Results tables...")
    for N in param_grid['N']:
        print_results_table(df, N=N, ttl=20,
                            epsilon=1.0, alpha=0.01, metric='l1')

    # ── Phase 4: Ablations ────────────────────────────────────────────────────
    ablations = {}
    if run_ablations_flag:
        print("\nPhase 4: Ablation studies...")
        ablations = run_ablations(
            step1_module, step2_module,
            step3_module, step4_module,
            N=200, ttl=20, epsilon=1.0, alpha=0.01,
            n_seeds=param_grid['n_seeds'],
            master_seed=master_seed + 1)

        plot_ablations(ablations)

        for name, abl_df in ablations.items():
            fname = f'step5_ablation_{name}.csv'
            abl_df.to_csv(fname, index=False)
            print(f"  Saved: {fname}")

    print()
    print("=" * 65)
    print("  Step 5 complete.")
    print("  Output files:")
    print("    step5_full_results.csv")
    print("    fig15 \u2013 fig21 (seven publication figures)")
    print("    step5_ablation_A1-A4.csv")
    print("  Next: use step5_full_results.csv to write \u00a75.")
    print("=" * 65)

    return df, ablations


# ── Entry point guard ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    raise RuntimeError(
        "Import this module and call main_step5(step1, step2, step3, step4)."
    )
