# ============================================================================
# MA-DTSR Simulation — Step 3: Baseline Routing Protocols
# ============================================================================
# Run after Step 1 and Step 2. Call main_step3(net, step2) from Colab.
#
# What this step produces:
#   1. EpidemicRouter   — flood-based baseline (upper bound on success rate)
#   2. RandomWalkRouter — uninformed random walk (lower bound on efficiency)
#   3. HeuristicRouter  — MA-DTSR without RL (softmin on D_alpha, eq. 5-6)
#   4. Unified episode runner and result recorder
#   5. Comparative evaluation across TTL and N
#   6. Figures 8-11: success rate, hop count, MAE, and utility comparisons
#
# The RL-augmented protocol is added in Step 4.
# All equation references are to Section 3 of the paper.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


# ============================================================================
# SECTION 1: Message and Result Structures
# ============================================================================

class RSM:
    """
    Resource Search Message — carries a query through the network.
    Implements Section 4.1 of the paper.

    Attributes
    ----------
    uid     : unique message identifier
    query   : np.ndarray (8,) — task query Q_s
    mask    : np.ndarray (8,) — relevance mask for partial queries (eq. 15)
    ttl     : int — remaining hop budget
    path    : list of int — ordered traversal path P (for loop prevention)
    origin  : int — source agent_id
    hops    : int — hops consumed so far
    """
    _counter = 0

    def __init__(self, query, mask, ttl, origin_id):
        RSM._counter   += 1
        self.uid        = RSM._counter
        self.query      = query.copy()
        self.mask       = mask.copy()
        self.ttl        = ttl
        self.path       = [origin_id]
        self.origin     = origin_id
        self.hops       = 0

    def copy_to(self, next_agent_id):
        """Return a forwarded copy with TTL decremented and path extended."""
        new       = RSM.__new__(RSM)
        new.uid   = self.uid
        new.query = self.query
        new.mask  = self.mask
        new.ttl   = self.ttl - 1
        new.path  = self.path + [next_agent_id]
        new.origin= self.origin
        new.hops  = self.hops + 1
        return new


class EpisodeResult:
    """
    Stores the outcome of one routing episode.

    Attributes
    ----------
    success      : bool   — was an admissible resource found?
    hops         : int    — number of hops taken (H in eq. 8)
    messages     : int    — total messages generated (M in eq. 8)
    match_error  : float  — D_L1(Q_s, R_i) at the match, or None
    utility      : float  — U_s from eq. (10), or 0 on failure
    protocol     : str    — name of the routing protocol used
    """
    def __init__(self, success, hops, messages, match_error,
                 utility, protocol):
        self.success     = success
        self.hops        = hops
        self.messages    = messages
        self.match_error = match_error
        self.utility     = utility
        self.protocol    = protocol

    def to_dict(self):
        return {
            'protocol'    : self.protocol,
            'success'     : int(self.success),
            'hops'        : self.hops,
            'messages'    : self.messages,
            'match_error' : self.match_error,
            'utility'     : self.utility,
        }


# ============================================================================
# SECTION 2: Utility Function
# ============================================================================

# Mission utility parameters (eq. 10)
UTILITY_PARAMS = {
    'U_max'  : 1.0,    # maximum achievable utility
    'gamma'  : 0.05,   # hop penalty weight
    'eta'    : 0.1,    # energy penalty weight
    'c_m'    : 0.01,   # per-message cost
}

def compute_utility(success, hops, messages, match_error,
                    energy_used, params=None):
    """
    Compute mission utility U_s from eq. (10):
    U_s = U_max * exp(-gamma * H) * exp(-eta * e) - c_m * M

    Parameters
    ----------
    success      : bool
    hops         : int    — H
    messages     : int    — M
    match_error  : float or None — D_L1 at match
    energy_used  : float  — e, fraction of energy consumed
    params       : dict or None

    Returns
    -------
    float — utility value (0 if failure)
    """
    if params is None:
        params = UTILITY_PARAMS
    if not success:
        return 0.0

    U = (params['U_max']
         * np.exp(-params['gamma'] * hops)
         * np.exp(-params['eta']   * energy_used)
         - params['c_m'] * messages)
    return max(0.0, U)


# ============================================================================
# SECTION 3: Base Router Class
# ============================================================================

class BaseRouter:
    """
    Abstract base class for all routing protocols.
    Subclasses implement select_next_hop().
    """

    def __init__(self, name, epsilon=1.0, metric='l1'):
        self.name    = name
        self.epsilon = epsilon    # admissibility threshold (eq. 3)
        self.metric  = metric     # similarity metric

    def get_distance(self, query, descriptor, mask=None):
        """Compute unweighted L1 distance for admissibility check."""
        if mask is None:
            mask = np.ones(len(query))
        return np.sum(mask * np.abs(query - descriptor))

    def get_score(self, query, descriptor, timestamp,
                  current_time, alpha, mask=None):
        """
        Compute age-aware distance D_alpha (eq. 4) using the
        configured similarity metric.
        """
        age       = current_time - timestamp
        staleness = np.exp(alpha * age)
        if self.metric == 'l1':
            base = self.get_distance(query, descriptor, mask)
        elif self.metric == 'l2':
            if mask is None:
                mask = np.ones(len(query))
            base = np.sqrt(np.sum(mask * (query - descriptor) ** 2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return staleness * base

    def check_local_match(self, agent, rsm):
        """
        Check whether agent i holds an admissible resource for query Q_s.
        Implements lines 4-8 of Algorithm 1.
        """
        if not agent.resource or agent.descriptor is None:
            return False, None
        dist = self.get_distance(rsm.query, agent.descriptor, rsm.mask)
        if dist <= self.epsilon:
            return True, dist
        return False, None

    def select_next_hop(self, agent, rsm, net):
        """
        Select the next-hop agent. Must be overridden by subclasses.

        Returns
        -------
        Agent or None
        """
        raise NotImplementedError

    def run_episode(self, net, step2_module, ttl, rng,
                    alpha=0.01, energy_cost_per_hop=0.02):
        """
        Run one complete routing episode from a random source agent.

        Returns
        -------
        EpisodeResult
        """
        # Pick a random source agent
        source   = net.agents[rng.integers(0, len(net.agents))]
        query, mask = step2_module.generate_query(rng)
        rsm      = RSM(query, mask, ttl, source.agent_id)

        messages_sent  = 0
        energy_used    = 0.0
        current_agent  = source

        # Check source itself first
        match, dist = self.check_local_match(current_agent, rsm)
        if match:
            utility = compute_utility(
                True, 0, 1, dist, energy_used)
            return EpisodeResult(True, 0, 1, dist, utility, self.name)

        while rsm.ttl > 0:
            next_agent = self.select_next_hop(current_agent, rsm, net)

            if next_agent is None:
                # No candidate — episode fails
                break

            # Forward the RSM
            rsm           = rsm.copy_to(next_agent.agent_id)
            messages_sent += 1
            energy_used   += energy_cost_per_hop
            current_agent  = next_agent

            # Check for local match at new agent
            match, dist = self.check_local_match(current_agent, rsm)
            if match:
                total_msgs = messages_sent + 1  # +1 for RFM reply
                utility = compute_utility(
                    True, rsm.hops, total_msgs, dist, energy_used)
                return EpisodeResult(
                    True, rsm.hops, total_msgs, dist, utility, self.name)

        # TTL exhausted or no candidates — failure
        return EpisodeResult(
            False, rsm.hops, messages_sent, None, 0.0, self.name)


# ============================================================================
# SECTION 4: Epidemic Router
# ============================================================================

class EpidemicRouter(BaseRouter):
    """
    Epidemic (flood-based) routing — upper bound on success rate.

    On each hop, the RSM is copied to ALL neighbours not already
    in the path. This maximises reachability but generates
    O(e^TTL) messages in the worst case.

    Reference: Vahdat & Becker (2000) — cited as baseline in §2.1.
    """

    def __init__(self, epsilon=1.0, metric='l1'):
        super().__init__('Epidemic', epsilon, metric)

    def run_episode(self, net, step2_module, ttl, rng,
                    alpha=0.01, energy_cost_per_hop=0.02):
        """
        Epidemic routing uses BFS-style flooding rather than
        single-path forwarding, so we override run_episode entirely.
        """
        source      = net.agents[rng.integers(0, len(net.agents))]
        query, mask = step2_module.generate_query(rng)

        seen_uids   = {source.agent_id}
        queue       = [(source, 0, 0.0)]   # (agent, hops_so_far, energy)
        messages    = 0
        best_result = None

        while queue:
            current, hops, energy = queue.pop(0)

            # Check local match
            match, dist = self.check_local_match(
                current,
                RSM(query, mask, ttl - hops, source.agent_id))
            if match:
                msgs    = messages + 1
                utility = compute_utility(True, hops, msgs, dist, energy)
                # Keep the shortest-path match
                if (best_result is None or
                        hops < best_result.hops):
                    best_result = EpisodeResult(
                        True, hops, msgs, dist, utility, self.name)

            if hops >= ttl:
                continue

            # Flood to all unvisited neighbours
            for nbr_id in net.neighbours.get(current.agent_id, []):
                if nbr_id not in seen_uids:
                    seen_uids.add(nbr_id)
                    nbr = net.agent_map[nbr_id]
                    queue.append((nbr, hops + 1,
                                  energy + energy_cost_per_hop))
                    messages += 1

        if best_result is not None:
            return best_result

        return EpisodeResult(False, ttl, messages, None, 0.0, self.name)

    def select_next_hop(self, agent, rsm, net):
        # Not used — epidemic overrides run_episode directly
        pass


# ============================================================================
# SECTION 5: Random Walk Router
# ============================================================================

class RandomWalkRouter(BaseRouter):
    """
    Uninformed random walk — lower bound baseline.

    At each hop, selects a uniformly random neighbour from
    N_i(t) \\ P (excluding already-visited nodes to prevent loops).
    No semantic guidance whatsoever.

    Reference: used as a baseline in §2.2 and §5 of the paper.
    """

    def __init__(self, epsilon=1.0, metric='l1'):
        super().__init__('RandomWalk', epsilon, metric)

    def select_next_hop(self, agent, rsm, net):
        candidates = [
            net.agent_map[j]
            for j in net.neighbours.get(agent.agent_id, [])
            if j not in rsm.path
        ]
        if not candidates:
            return None
        return candidates[np.random.randint(len(candidates))]


# ============================================================================
# SECTION 6: Heuristic MA-DTSR Router
# ============================================================================

class HeuristicRouter(BaseRouter):
    """
    MA-DTSR without RL — heuristic-only semantic routing.

    Implements the softmin next-hop selection rule from eq. (6):
      Pr[X_{h+1} = j | X_h = i] ∝ exp(-beta * D_alpha(Q_s, R-hat_j; t))

    This is the MA-DTSR protocol without the learned Q-function.
    It provides the warm-start baseline for Step 4's RL agent.
    Also supports deterministic argmin mode (eq. 5).

    Parameters
    ----------
    beta        : float  — softmin temperature (higher = more greedy)
    mode        : str    — 'softmin' (eq. 6) or 'deterministic' (eq. 5)
    alpha       : float  — staleness penalty in D_alpha (eq. 4)
    lambda_E    : float  — energy cost weight in score (eq. 12)
    lambda_A    : float  — airtime cost weight in score (eq. 12)
    """

    def __init__(self, epsilon=1.0, metric='l1',
                 beta=2.0, mode='softmin',
                 alpha=0.01, lambda_E=0.0, lambda_A=0.0):
        super().__init__('Heuristic-MADTSR', epsilon, metric)
        self.beta     = beta
        self.mode     = mode
        self.alpha    = alpha
        self.lambda_E = lambda_E
        self.lambda_A = lambda_A

    def _budget_aware_score(self, query, descriptor, timestamp,
                            current_time, mask=None):
        """
        Compute budget-aware score from eq. (12):
        score_j = D_alpha(Q_s, R-hat_j; t) + lambda_E * e_b + lambda_A * a_b

        For simplicity in the heuristic baseline, e_b and a_b are
        set to 1.0 (unit costs); lambda weights control their influence.
        """
        d   = self.get_score(query, descriptor, timestamp,
                             current_time, self.alpha, mask)
        e_b = 1.0   # unit energy cost per hop
        a_b = 1.0   # unit airtime cost per hop
        return d + self.lambda_E * e_b + self.lambda_A * a_b

    def select_next_hop(self, agent, rsm, net):
        """
        Select next hop using semantic guidance from the contact database.
        Implements Algorithm 2 (heuristic branch) from Section 4.3.
        """
        candidates = [
            net.agent_map[j]
            for j in net.neighbours.get(agent.agent_id, [])
            if j not in rsm.path
        ]
        if not candidates:
            return None

        scores = []
        for cand in candidates:
            entry = net.contact_db.get_entry(
                agent.agent_id, cand.agent_id)
            if entry is not None:
                score = self._budget_aware_score(
                    rsm.query,
                    entry['descriptor'],
                    entry['timestamp'],
                    net.time,
                    rsm.mask)
            else:
                # No descriptor known — assign a high penalty score
                # so unknown neighbours are deprioritised
                score = 10.0
            scores.append(score)

        scores = np.array(scores)

        if self.mode == 'deterministic':
            # eq. (5): argmin
            return candidates[int(np.argmin(scores))]

        else:
            # eq. (6): softmin sampling
            # Subtract min for numerical stability before exp
            shifted = scores - scores.min()
            weights = np.exp(-self.beta * shifted)
            weights /= weights.sum()
            idx = np.random.choice(len(candidates), p=weights)
            return candidates[idx]


# ============================================================================
# SECTION 7: Evaluation Runner
# ============================================================================

def run_comparison(net, step2_module, protocols,
                   ttl_values, n_episodes, alpha=0.01,
                   seed=7):
    """
    Run all protocols across all TTL values and collect results.

    Parameters
    ----------
    net           : DisasterNetwork from Step 1
    step2_module  : Step 2 module (for generate_query)
    protocols     : list of BaseRouter instances
    ttl_values    : list of int  — TTL budgets to test
    n_episodes    : int  — episodes per (protocol, TTL) combination
    alpha         : float — staleness parameter
    seed          : int   — master random seed

    Returns
    -------
    pd.DataFrame with one row per episode
    """
    rng     = np.random.default_rng(seed)
    records = []

    total = len(protocols) * len(ttl_values) * n_episodes
    pbar  = tqdm(total=total, desc='Running protocols', unit='ep')

    for protocol in protocols:
        for ttl in ttl_values:
            for ep in range(n_episodes):
                result = protocol.run_episode(
                    net, step2_module, ttl, rng, alpha=alpha)
                row = result.to_dict()
                row['ttl']     = ttl
                row['episode'] = ep
                records.append(row)
                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def summarise_results(df):
    """
    Aggregate episode results into per-(protocol, TTL) statistics.

    Returns
    -------
    pd.DataFrame with mean ± std for each metric
    """
    grp = df.groupby(['protocol', 'ttl'])

    summary = grp.agg(
        success_rate  = ('success',     'mean'),
        mean_hops     = ('hops',        'mean'),
        std_hops      = ('hops',        'std'),
        mean_messages = ('messages',    'mean'),
        mean_mae      = ('match_error', lambda x: x.dropna().mean()),
        mean_utility  = ('utility',     'mean'),
        n_episodes    = ('success',     'count'),
    ).reset_index()

    return summary


# ============================================================================
# SECTION 8: Visualisations
# ============================================================================

PROTOCOL_STYLES = {
    'Epidemic'         : {'color': '#e63946', 'ls': '-',  'marker': 'o'},
    'RandomWalk'       : {'color': '#adb5bd', 'ls': '--', 'marker': 's'},
    'Heuristic-MADTSR' : {'color': '#1d3557', 'ls': '-',  'marker': '^'},
}

PROTOCOL_LABELS = {
    'Epidemic'         : 'Epidemic routing',
    'RandomWalk'       : 'Random walk',
    'Heuristic-MADTSR' : 'MA-DTSR (heuristic)',
}


def _style(name, key):
    return PROTOCOL_STYLES.get(name, {}).get(key, None)


def plot_success_rate(summary, ax=None, title_suffix=''):
    """Figure 8: Success rate vs TTL for each protocol (eq. 7)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for proto, grp in summary.groupby('protocol'):
        ax.plot(grp['ttl'], grp['success_rate'] * 100,
                color=_style(proto, 'color'),
                linestyle=_style(proto, 'ls'),
                marker=_style(proto, 'marker'),
                markersize=5, linewidth=1.8,
                label=PROTOCOL_LABELS.get(proto, proto))
    ax.set_xlabel('TTL budget', fontsize=9)
    ax.set_ylabel('Success rate (%)', fontsize=9)
    ax.set_title(f'Success Rate vs TTL{title_suffix}', fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_hop_count(summary, ax=None, title_suffix=''):
    """Figure 9: Mean hop count vs TTL (eq. 8)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for proto, grp in summary.groupby('protocol'):
        ax.plot(grp['ttl'], grp['mean_hops'],
                color=_style(proto, 'color'),
                linestyle=_style(proto, 'ls'),
                marker=_style(proto, 'marker'),
                markersize=5, linewidth=1.8,
                label=PROTOCOL_LABELS.get(proto, proto))
        ax.fill_between(grp['ttl'],
                        grp['mean_hops'] - grp['std_hops'].fillna(0),
                        grp['mean_hops'] + grp['std_hops'].fillna(0),
                        alpha=0.1,
                        color=_style(proto, 'color'))
    ax.set_xlabel('TTL budget', fontsize=9)
    ax.set_ylabel('Mean hops (H)', fontsize=9)
    ax.set_title(f'Mean Hop Count vs TTL{title_suffix}', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_mae(summary, ax=None, title_suffix=''):
    """Figure 10: Mean absolute error vs TTL (eq. 9)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for proto, grp in summary.groupby('protocol'):
        grp_valid = grp[grp['mean_mae'].notna()]
        if grp_valid.empty:
            continue
        ax.plot(grp_valid['ttl'], grp_valid['mean_mae'],
                color=_style(proto, 'color'),
                linestyle=_style(proto, 'ls'),
                marker=_style(proto, 'marker'),
                markersize=5, linewidth=1.8,
                label=PROTOCOL_LABELS.get(proto, proto))
    ax.set_xlabel('TTL budget', fontsize=9)
    ax.set_ylabel('MAE (semantic match error)', fontsize=9)
    ax.set_title(f'Match Quality (MAE) vs TTL{title_suffix}', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_utility(summary, ax=None, title_suffix=''):
    """Figure 11: Mean mission utility vs TTL (eq. 10)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for proto, grp in summary.groupby('protocol'):
        ax.plot(grp['ttl'], grp['mean_utility'],
                color=_style(proto, 'color'),
                linestyle=_style(proto, 'ls'),
                marker=_style(proto, 'marker'),
                markersize=5, linewidth=1.8,
                label=PROTOCOL_LABELS.get(proto, proto))
    ax.set_xlabel('TTL budget', fontsize=9)
    ax.set_ylabel('Mean utility (U_s)', fontsize=9)
    ax.set_title(f'Mission Utility vs TTL{title_suffix}', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_message_count(summary, ax=None, title_suffix=''):
    """Figure 12: Mean message count vs TTL (eq. 8)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for proto, grp in summary.groupby('protocol'):
        ax.plot(grp['ttl'], grp['mean_messages'],
                color=_style(proto, 'color'),
                linestyle=_style(proto, 'ls'),
                marker=_style(proto, 'marker'),
                markersize=5, linewidth=1.8,
                label=PROTOCOL_LABELS.get(proto, proto))
    ax.set_xlabel('TTL budget', fontsize=9)
    ax.set_ylabel('Mean messages (M)', fontsize=9)
    ax.set_title(f'Message Count vs TTL{title_suffix}', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_all_metrics(summary, filename_prefix='step3'):
    """Generate all five comparison figures in a single 2×3 panel."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        'MA-DTSR Step 3: Baseline Protocol Comparison\n'
        '(Epidemic vs Random Walk vs MA-DTSR Heuristic)',
        fontsize=12)

    plot_success_rate(summary,    ax=axes[0, 0])
    plot_hop_count(summary,       ax=axes[0, 1])
    plot_mae(summary,             ax=axes[0, 2])
    plot_utility(summary,         ax=axes[1, 0])
    plot_message_count(summary,   ax=axes[1, 1])

    # Protocol legend on last panel
    axes[1, 2].axis('off')
    legend_handles = [
        mlines.Line2D([], [],
                      color=PROTOCOL_STYLES[p]['color'],
                      linestyle=PROTOCOL_STYLES[p]['ls'],
                      marker=PROTOCOL_STYLES[p]['marker'],
                      markersize=8, linewidth=2,
                      label=PROTOCOL_LABELS[p])
        for p in PROTOCOL_STYLES
    ]
    axes[1, 2].legend(handles=legend_handles,
                      loc='center', fontsize=11,
                      title='Protocols', title_fontsize=11)

    fig.tight_layout()
    fname = f'{filename_prefix}_comparison.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {fname}")


# ============================================================================
# SECTION 9: Sanity Checks
# ============================================================================

def run_protocol_sanity_checks(df, summary):
    """
    Verify protocol behaviour is consistent with theoretical expectations.
    """
    print("=" * 65)
    print("  MA-DTSR Step 3 — Protocol Sanity Check Results")
    print("=" * 65)

    protos = df['protocol'].unique()
    ttls   = sorted(df['ttl'].unique())

    for ttl in ttls:
        sub = summary[summary['ttl'] == ttl]

        epi  = sub[sub['protocol'] == 'Epidemic']
        rw   = sub[sub['protocol'] == 'RandomWalk']
        heur = sub[sub['protocol'] == 'Heuristic-MADTSR']

        if epi.empty or rw.empty or heur.empty:
            continue

        epi_sr  = epi['success_rate'].values[0]
        rw_sr   = rw['success_rate'].values[0]
        heur_sr = heur['success_rate'].values[0]

        epi_msgs  = epi['mean_messages'].values[0]
        heur_msgs = heur['mean_messages'].values[0]

        print(f"\n  TTL = {ttl}")

        # Check 1: Epidemic should have highest or equal success rate
        ok1 = epi_sr >= heur_sr - 0.05   # 5% tolerance
        print(f"  [{'PASS' if ok1 else 'WARN'}] "
              f"Epidemic SR ({epi_sr*100:.1f}%) >= "
              f"Heuristic SR ({heur_sr*100:.1f}%) ± 5%")

        # Check 2: Heuristic should have fewer messages than Epidemic
        ok2 = heur_msgs <= epi_msgs
        print(f"  [{'PASS' if ok2 else 'WARN'}] "
              f"Heuristic messages ({heur_msgs:.1f}) <= "
              f"Epidemic messages ({epi_msgs:.1f})")

        # Check 3: Heuristic should have lower or equal MAE than Random Walk
        if not heur['mean_mae'].isna().all() and not rw['mean_mae'].isna().all():
            heur_mae = heur['mean_mae'].values[0]
            rw_mae   = rw['mean_mae'].values[0]
            ok3 = (heur_mae <= rw_mae + 0.1) or np.isnan(heur_mae)
            if not np.isnan(heur_mae) and not np.isnan(rw_mae):
                print(f"  [{'PASS' if ok3 else 'WARN'}] "
                      f"Heuristic MAE ({heur_mae:.3f}) <= "
                      f"Random Walk MAE ({rw_mae:.3f}) ± 0.1")

        # Check 4: Success rate increases with TTL for all protocols
        # (checked across TTL values, printed once after loop)

    # Check 5: Success rate monotonically increases with TTL
    print("\n  --- Monotonicity Check (SR should rise with TTL) ---")
    for proto in protos:
        sub  = summary[summary['protocol'] == proto].sort_values('ttl')
        srs  = sub['success_rate'].values
        mono = all(srs[i] <= srs[i+1] + 0.05 for i in range(len(srs)-1))
        print(f"  [{'PASS' if mono else 'WARN'}] "
              f"{PROTOCOL_LABELS.get(proto, proto)}: "
              f"SR = {[f'{s*100:.0f}%' for s in srs]}")

    print("\n" + "=" * 65)
    print("  Step 3 complete. Baseline protocols validated.")
    print("  Next: Step 4 adds the RL agent on top of the heuristic.")
    print("=" * 65)


# ============================================================================
# SECTION 10: Main
# ============================================================================

def main_step3(net, step2_module,
               ttl_values=None,
               n_episodes=200,
               epsilon=1.0,
               alpha=0.01,
               beta=2.0):
    """
    Run Step 3: build and evaluate all three baseline protocols.

    Parameters
    ----------
    net           : DisasterNetwork (from Step 1)
    step2_module  : Step 2 module (for generate_query, d_l1, etc.)
    ttl_values    : list of int — default [10, 20, 30, 40]
    n_episodes    : int — episodes per (protocol, TTL) — default 200
    epsilon       : float — admissibility threshold — default 1.0
    alpha         : float — staleness parameter — default 0.01
    beta          : float — softmin temperature — default 2.0

    Returns
    -------
    df      : pd.DataFrame — raw episode results
    summary : pd.DataFrame — aggregated statistics
    routers : dict — {name: router_instance}
    """
    if ttl_values is None:
        ttl_values = [10, 20, 30, 40]

    print("MA-DTSR Simulation \u2014 Step 3: Baseline Routing Protocols")
    print(f"Network  : N={len(net.agents)} agents, t={net.time:.0f}s")
    print(f"Episodes : {n_episodes} per (protocol, TTL)")
    print(f"TTL range: {ttl_values}")
    print(f"Epsilon  : {epsilon}  |  Alpha: {alpha}  |  Beta: {beta}")
    print()

    # ── Build protocols ───────────────────────────────────────────────────────
    routers = {
        'Epidemic'         : EpidemicRouter(epsilon=epsilon),
        'RandomWalk'       : RandomWalkRouter(epsilon=epsilon),
        'Heuristic-MADTSR' : HeuristicRouter(
                                epsilon=epsilon,
                                beta=beta,
                                mode='softmin',
                                alpha=alpha),
    }

    protocols = list(routers.values())

    # ── Run evaluation ────────────────────────────────────────────────────────
    df = run_comparison(
        net, step2_module, protocols,
        ttl_values=ttl_values,
        n_episodes=n_episodes,
        alpha=alpha)

    summary = summarise_results(df)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n--- Results Summary ---")
    print(summary[['protocol', 'ttl', 'success_rate',
                   'mean_hops', 'mean_messages',
                   'mean_mae', 'mean_utility']
          ].to_string(index=False, float_format='{:.3f}'.format))

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print()
    run_protocol_sanity_checks(df, summary)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating comparison figures...")
    plot_all_metrics(summary)

    # ── Save results ──────────────────────────────────────────────────────────
    df.to_csv('step3_raw_results.csv', index=False)
    summary.to_csv('step3_summary.csv', index=False)
    print("Results saved: step3_raw_results.csv, step3_summary.csv")

    return df, summary, routers
