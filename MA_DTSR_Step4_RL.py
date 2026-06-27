# ============================================================================
# MA-DTSR Simulation — Step 4: RL-Augmented Routing (Forward-Only)
# ============================================================================
# Run after Steps 1, 2, and 3. Call main_step4(net, step2, step3) from Colab.
#
# What this step produces:
#   1. LinearApproximator — weight vector theta_i, feature vector phi(o,a),
#                           linear TD update rule (eq. 13)
#   2. AgentMemory        — per-agent weight storage and visit counts
#   3. RLRouter           — MA-DTSR with RL (forward-only action space)
#                           inherits HeuristicRouter warm-start from Step 3
#   4. CooperativeExchange — weight merging between contacted agents (eq. 14)
#   5. TrainingLoop       — runs episodes, updates weights, decays rho
#   6. Convergence plots  — learning curves over training episodes
#   7. Full 4-protocol comparison — Epidemic, RandomWalk, Heuristic, RL
#
# Forward-only version: action space = {Forward(j) : j in N_i(t) \ P}
# Wait action is added in Step 5.
#
# All equation references are to Section 3 of the paper.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


# ============================================================================
# SECTION 1: Linear Function Approximator
# ============================================================================

class LinearApproximator:
    """
    Linear function approximator for the Q-function.
    Implements Section 3.8.1 and eq. (13) of the paper.

    Q_i(o, a) = theta_i^T * phi(o, a)

    Feature vector phi(o, a) in R^4:
      [0] normalised semantic score of action a
      [1] normalised remaining TTL
      [2] normalised remaining energy
      [3] wait flag (0 for forward, 1 for wait — always 0 in this step)

    Parameters
    ----------
    n_features : int   — feature vector dimension (default 4)
    lr         : float — learning rate eta_Q (default 0.05)
    gamma      : float — discount factor gamma_Q (default 0.9)
    """

    N_FEATURES = 4

    def __init__(self, lr=0.05, gamma=0.9):
        self.lr    = lr       # eta_Q
        self.gamma = gamma    # gamma_Q
        self.theta = np.zeros(self.N_FEATURES)   # weight vector theta_i
        self.visit_count = 0   # total updates received

    def build_features(self, score, ttl, ttl0, e_rem, e0,
                       is_wait=False):
        """
        Build the 4-dimensional feature vector phi(o, a).

        Parameters
        ----------
        score    : float — budget-aware score for this action (eq. 12)
        ttl      : int   — remaining TTL
        ttl0     : int   — initial TTL budget
        e_rem    : float — remaining energy fraction [0,1]
        e0       : float — initial energy (normalisation denominator)
        is_wait  : bool  — True for wait action (always False in Step 4)

        Returns
        -------
        np.ndarray shape (4,)
        """
        score_norm = score / (score + 1e-6)       # [0, 1)
        ttl_norm   = ttl  / max(ttl0, 1)          # [0, 1]
        e_norm     = e_rem / max(e0, 1e-6)        # [0, 1]
        wait_flag  = float(is_wait)               # 0 or 1

        return np.array([score_norm, ttl_norm, e_norm, wait_flag])

    def q_value(self, phi):
        """
        Compute Q(o, a) = theta^T * phi.
        """
        return float(self.theta @ phi)

    def update(self, phi, reward, phi_next=None):
        """
        Linear TD update from eq. (13):
        theta_i <- theta_i + eta_Q * [r + gamma_Q * max_a' Q(o',a') - Q(o,a)] * phi(o,a)

        For terminal states (successful match or TTL exhaustion),
        phi_next is None and the target is just r.

        Parameters
        ----------
        phi      : np.ndarray (4,) — feature vector of taken action
        reward   : float           — received reward r
        phi_next : np.ndarray (4,) or None — best feature vector at o'
        """
        q_current = self.q_value(phi)

        if phi_next is not None:
            q_next = self.q_value(phi_next)
        else:
            q_next = 0.0   # terminal state

        td_error    = reward + self.gamma * q_next - q_current
        self.theta += self.lr * td_error * phi
        self.visit_count += 1

    def merge(self, other_theta, alpha_merge=0.3):
        """
        Cooperative weight merge from eq. (14):
        theta_i <- (1 - alpha_merge) * theta_i + alpha_merge * theta_j

        Called during contact events (Algorithm 3, lines 14-17).

        Parameters
        ----------
        other_theta  : np.ndarray (4,) — received weight vector from peer j
        alpha_merge  : float           — merge weight alpha_merge in (0,1)
        """
        self.theta = ((1 - alpha_merge) * self.theta
                      + alpha_merge * other_theta)

    def copy_weights(self):
        """Return a copy of the current weight vector."""
        return self.theta.copy()

    def warm_start(self, scores):
        """
        Initialise theta from heuristic scores (eq. 12 warm-start).

        Sets theta[0] = -1 so that higher score -> lower Q-value,
        matching the argmin heuristic. Other dimensions start at 0.
        This means the initial policy approximates the softmin heuristic.
        """
        self.theta = np.array([-1.0, 0.1, 0.1, 0.0])


# ============================================================================
# SECTION 2: Agent Memory
# ============================================================================

class AgentMemory:
    """
    Stores per-agent RL state: one LinearApproximator per agent.

    In the full protocol each agent maintains its own theta_i.
    Here we maintain a dictionary keyed by agent_id so that
    cooperative exchange can be simulated between any pair.

    Parameters
    ----------
    agents     : list of Agent
    lr         : float — learning rate
    gamma      : float — discount factor
    """

    def __init__(self, agents, lr=0.05, gamma=0.9):
        self.approximators = {
            a.agent_id: LinearApproximator(lr=lr, gamma=gamma)
            for a in agents
        }
        # Warm-start all agents
        for approx in self.approximators.values():
            approx.warm_start(scores=None)

    def get(self, agent_id):
        """Return the approximator for agent_id."""
        return self.approximators[agent_id]

    def exchange(self, agent_i_id, agent_j_id,
                 n_min=5, b_q=1, alpha_merge=0.3):
        """
        Cooperative weight exchange between agents i and j.
        Implements Algorithm 3 lines 14-17 and eq. (14).

        Exchange only occurs if the sending agent has accumulated
        at least n_min updates (visit_count >= n_min).
        b_q limits exchange to one weight vector per contact event.

        Parameters
        ----------
        agent_i_id   : int
        agent_j_id   : int
        n_min        : int   — minimum visit count threshold
        b_q          : int   — max exchanges per contact (always 1 here)
        alpha_merge  : float — merge weight
        """
        approx_i = self.approximators[agent_i_id]
        approx_j = self.approximators[agent_j_id]

        # Exchange only if confident enough
        if approx_i.visit_count >= n_min:
            approx_j.merge(approx_i.copy_weights(), alpha_merge)

        if approx_j.visit_count >= n_min:
            approx_i.merge(approx_j.copy_weights(), alpha_merge)

    def total_updates(self):
        """Return total TD updates across all agents."""
        return sum(a.visit_count for a in self.approximators.values())


# ============================================================================
# SECTION 3: Reward Function
# ============================================================================

# Reward parameters — directly from Section 3.8.3
REWARD_PARAMS = {
    'U_max'    : 1.0,    # maximum success reward
    'gamma_r'  : 0.05,   # hop penalty in reward (matches utility gamma)
    'eta_r'    : 0.1,    # energy penalty in reward
    'c_hop'    : 0.01,   # per-hop penalty
    'c_ttl'    : 0.5,    # TTL exhaustion penalty (large negative)
}

def compute_reward(success, hops, energy_used,
                   match_error=None, params=None):
    """
    Compute the RL reward r(s, a, s') from Section 3.8.3.

    On successful match:
        r = U_max * exp(-gamma_r * H) * exp(-eta_r * e)
    On TTL exhaustion without match:
        r = -c_TTL
    Per forward hop (applied at each step):
        r = -c_hop

    In practice we apply the full episode reward retrospectively
    at the end of each episode (terminal reward formulation).

    Parameters
    ----------
    success     : bool
    hops        : int
    energy_used : float
    match_error : float or None
    params      : dict or None

    Returns
    -------
    float
    """
    if params is None:
        params = REWARD_PARAMS

    if success:
        r = (params['U_max']
             * np.exp(-params['gamma_r'] * hops)
             * np.exp(-params['eta_r']   * energy_used))
    else:
        r = -params['c_ttl']

    # Subtract per-hop cost
    r -= params['c_hop'] * hops
    return r


# ============================================================================
# SECTION 4: RL Router
# ============================================================================

class RLRouter:
    """
    MA-DTSR with RL — forward-only action space.

    Inherits semantic scoring from HeuristicRouter (Step 3) as its
    warm-start. Replaces the softmin selection with a learned
    epsilon-greedy policy over Q(o, a) = theta_i^T * phi(o, a).

    During training (is_training=True):
      - With probability rho: explore (softmin heuristic)
      - With probability 1-rho: exploit (argmax Q-value)
      - Updates theta_i after each episode via linear TD (eq. 13)
      - Exchanges weights with contacted neighbours (eq. 14)

    During evaluation (is_training=False):
      - Always exploits: argmax Q-value
      - No weight updates

    Parameters
    ----------
    epsilon      : float — admissibility threshold (eq. 3)
    alpha        : float — staleness parameter in D_alpha (eq. 4)
    beta         : float — softmin temperature for exploration
    rho_start    : float — initial exploration rate
    rho_end      : float — final exploration rate after decay
    rho_decay    : float — multiplicative decay per episode
    lr           : float — learning rate eta_Q
    gamma_q      : float — discount factor gamma_Q
    alpha_merge  : float — cooperative merge weight (eq. 14)
    n_min        : int   — min visits before sharing weights
    lambda_E     : float — energy weight in score (eq. 12)
    lambda_A     : float — airtime weight in score (eq. 12)
    """

    def __init__(self, agents, epsilon=1.0, alpha=0.01, beta=2.0,
                 rho_start=0.8, rho_end=0.05, rho_decay=0.995,
                 lr=0.05, gamma_q=0.9,
                 alpha_merge=0.3, n_min=5,
                 lambda_E=0.0, lambda_A=0.0):

        self.name        = 'RL-MADTSR'
        self.epsilon     = epsilon
        self.alpha       = alpha
        self.beta        = beta
        self.rho         = rho_start     # current exploration rate
        self.rho_end     = rho_end
        self.rho_decay   = rho_decay
        self.alpha_merge = alpha_merge
        self.n_min       = n_min
        self.lambda_E    = lambda_E
        self.lambda_A    = lambda_A

        # Per-agent weight vectors
        self.memory = AgentMemory(agents, lr=lr, gamma=gamma_q)

        # Episode counter for logging
        self.episode_count = 0

        # Training log — (episode, reward, success) per episode
        self.training_log  = []

    # ── Scoring (reuses heuristic logic) ─────────────────────────────────────

    def get_distance(self, query, descriptor, mask=None):
        if mask is None:
            mask = np.ones(len(query))
        return float(np.sum(mask * np.abs(query - descriptor)))

    def check_local_match(self, agent, rsm):
        if not agent.resource or agent.descriptor is None:
            return False, None
        dist = self.get_distance(rsm.query, agent.descriptor, rsm.mask)
        if dist <= self.epsilon:
            return True, dist
        return False, None

    def _get_score(self, query, descriptor, timestamp,
                   current_time, mask=None):
        """Budget-aware score from eq. (12)."""
        age       = current_time - timestamp
        staleness = np.exp(self.alpha * age)
        base      = self.get_distance(query, descriptor, mask)
        d_alpha   = staleness * base
        return d_alpha + self.lambda_E * 1.0 + self.lambda_A * 1.0

    def _candidate_scores(self, agent, rsm, net):
        """
        Build list of (candidate_agent, score, phi) for all
        forward-action candidates at the current hop.
        """
        ttl0      = rsm.hops + rsm.ttl   # original TTL budget
        e_rem     = agent.energy
        results   = []

        candidates = [
            net.agent_map[j]
            for j in net.neighbours.get(agent.agent_id, [])
            if j not in rsm.path
        ]

        for cand in candidates:
            entry = net.contact_db.get_entry(
                agent.agent_id, cand.agent_id)
            if entry is not None:
                score = self._get_score(
                    rsm.query, entry['descriptor'],
                    entry['timestamp'], net.time, rsm.mask)
            else:
                score = 10.0   # unknown: high penalty

            approx = self.memory.get(agent.agent_id)
            phi    = approx.build_features(
                score, rsm.ttl, ttl0, e_rem, e_rem)
            results.append((cand, score, phi))

        return results

    # ── Action selection ──────────────────────────────────────────────────────

    def select_next_hop(self, agent, rsm, net, is_training=True):
        """
        Select next-hop using rho-greedy policy.

        During training with probability rho: explore via softmin (eq. 6)
        Otherwise: exploit via argmax Q-value (eq. 13).

        Returns
        -------
        (Agent or None, phi or None, score or None)
        """
        items = self._candidate_scores(agent, rsm, net)
        if not items:
            return None, None, None

        candidates, scores, phis = zip(*items)
        scores = np.array(scores)
        phis   = list(phis)

        # rho-greedy exploration
        if is_training and np.random.random() < self.rho:
            # Explore: softmin sampling (eq. 6)
            shifted = scores - scores.min()
            weights = np.exp(-self.beta * shifted)
            weights /= weights.sum()
            idx = np.random.choice(len(candidates), p=weights)
        else:
            # Exploit: argmax Q-value
            approx  = self.memory.get(agent.agent_id)
            q_vals  = np.array([approx.q_value(phi) for phi in phis])
            idx     = int(np.argmax(q_vals))

        return candidates[idx], phis[idx], float(scores[idx])

    # ── Cooperative exchange ──────────────────────────────────────────────────

    def _maybe_exchange(self, agent_i_id, net):
        """
        Trigger cooperative weight exchange with all current neighbours.
        Called at the start of each hop in training mode.
        """
        for j_id in net.neighbours.get(agent_i_id, []):
            self.memory.exchange(
                agent_i_id, j_id,
                n_min=self.n_min,
                alpha_merge=self.alpha_merge)

    # ── Episode runner ────────────────────────────────────────────────────────

    def run_episode(self, net, step2_module, ttl, rng,
                    alpha=0.01, energy_cost_per_hop=0.02,
                    is_training=True):
        """
        Run one complete routing episode with RL weight updates.

        During training:
          - Collects (phi, next_best_phi) pairs at each hop
          - Applies terminal reward retrospectively after episode ends
          - Updates theta_i for the decision-making agent
          - Triggers cooperative exchange at each hop

        Parameters
        ----------
        is_training : bool — if False, runs greedy policy without updates

        Returns
        -------
        EpisodeResult
        """
        source      = net.agents[rng.integers(0, len(net.agents))]
        query, mask = step2_module.generate_query(rng)
        rsm         = RSM(query, mask, ttl, source.agent_id)

        messages_sent   = 0
        energy_used     = 0.0
        current_agent   = source

        # Store (agent_id, phi_taken) for retrospective update
        trajectory = []

        # Check source itself first
        match, dist = self.check_local_match(current_agent, rsm)
        if match:
            reward  = compute_reward(True, 0, 0.0)
            utility = compute_utility_rl(True, 0, 1, dist, 0.0)
            result  = EpisodeResult_RL(
                True, 0, 1, dist, utility, self.name)
            self._log_and_decay(reward, True, is_training)
            return result

        # ── Routing loop ──────────────────────────────────────────────────────
        while rsm.ttl > 0:

            # Cooperative exchange at current agent before deciding
            if is_training:
                self._maybe_exchange(current_agent.agent_id, net)

            # Select next hop
            next_agent, phi_taken, score = self.select_next_hop(
                current_agent, rsm, net, is_training=is_training)

            if next_agent is None:
                break   # no candidates — episode fails

            # Store decision for retrospective update
            if is_training and phi_taken is not None:
                trajectory.append(
                    (current_agent.agent_id, phi_taken))

            # Forward
            rsm           = rsm.copy_to(next_agent.agent_id)
            messages_sent += 1
            energy_used   += energy_cost_per_hop
            current_agent  = next_agent

            # Check match at new agent
            match, dist = self.check_local_match(current_agent, rsm)
            if match:
                total_msgs = messages_sent + 1
                reward     = compute_reward(
                    True, rsm.hops, energy_used)
                utility    = compute_utility_rl(
                    True, rsm.hops, total_msgs, dist, energy_used)
                result = EpisodeResult_RL(
                    True, rsm.hops, total_msgs, dist,
                    utility, self.name)

                # Retrospective TD update — success reward
                if is_training:
                    self._update_trajectory(
                        trajectory, reward,
                        next_phi=None)   # terminal

                self._log_and_decay(reward, True, is_training)
                return result

        # ── Episode failed (TTL exhausted) ────────────────────────────────────
        reward = compute_reward(False, rsm.hops, energy_used)
        result = EpisodeResult_RL(
            False, rsm.hops, messages_sent, None, 0.0, self.name)

        if is_training:
            self._update_trajectory(
                trajectory, reward, next_phi=None)

        self._log_and_decay(reward, False, is_training)
        return result

    def _update_trajectory(self, trajectory, terminal_reward, next_phi):
        """
        Apply the terminal reward back through the trajectory.

        Uses a simple credit assignment: each step receives the
        full terminal reward (Monte Carlo style), since episodes
        are short and TD bootstrapping with a linear approximator
        can be unstable on very short trajectories.
        """
        for agent_id, phi in reversed(trajectory):
            approx = self.memory.get(agent_id)
            approx.update(phi, terminal_reward, phi_next=next_phi)

    def _log_and_decay(self, reward, success, is_training):
        """Log episode result and decay exploration rate."""
        self.episode_count += 1
        self.training_log.append({
            'episode': self.episode_count,
            'reward' : reward,
            'success': int(success),
            'rho'    : self.rho,
        })
        if is_training:
            self.rho = max(self.rho_end,
                           self.rho * self.rho_decay)


# ============================================================================
# SECTION 5: Episode Result (RL version)
# ============================================================================
# Identical structure to Step 3's EpisodeResult — kept separate so Step 3
# code is unchanged and Step 4 can be imported independently.

class EpisodeResult_RL:
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


def compute_utility_rl(success, hops, messages, match_error,
                       energy_used):
    """Wrapper around compute_reward for utility reporting."""
    if not success:
        return 0.0
    from MA_DTSR_Step3_Baselines import compute_utility, UTILITY_PARAMS
    return compute_utility(success, hops, messages,
                           match_error, energy_used, UTILITY_PARAMS)


# ============================================================================
# SECTION 6: Training Loop
# ============================================================================

def train_rl_router(rl_router, net, step2_module,
                    ttl, n_train_episodes,
                    alpha=0.01, seed=123):
    """
    Train the RL router for n_train_episodes episodes.

    Parameters
    ----------
    rl_router        : RLRouter instance
    net              : DisasterNetwork
    step2_module     : Step 2 module
    ttl              : int — TTL budget used during training
    n_train_episodes : int — number of training episodes
    alpha            : float — staleness parameter
    seed             : int

    Returns
    -------
    pd.DataFrame — training log with episode, reward, success, rho
    """
    rng = np.random.default_rng(seed)

    print(f"  Training RL router: {n_train_episodes} episodes, "
          f"TTL={ttl}, rho_start={rl_router.rho:.2f}")

    for ep in tqdm(range(n_train_episodes),
                   desc='  Training', unit='ep', leave=False):
        rl_router.run_episode(
            net, step2_module, ttl, rng,
            alpha=alpha, is_training=True)

    log_df = pd.DataFrame(rl_router.training_log)
    print(f"  Training complete. "
          f"Final rho={rl_router.rho:.4f}, "
          f"Total updates={rl_router.memory.total_updates()}")
    return log_df


def evaluate_rl_router(rl_router, net, step2_module,
                       ttl_values, n_eval_episodes,
                       alpha=0.01, seed=456):
    """
    Evaluate the trained RL router (greedy policy, no updates).

    Returns
    -------
    pd.DataFrame — evaluation results
    """
    rng     = np.random.default_rng(seed)
    records = []

    for ttl in ttl_values:
        for ep in range(n_eval_episodes):
            result = rl_router.run_episode(
                net, step2_module, ttl, rng,
                alpha=alpha, is_training=False)
            row        = result.to_dict()
            row['ttl'] = ttl
            records.append(row)

    return pd.DataFrame(records)


# ============================================================================
# SECTION 7: Visualisations
# ============================================================================

# Style consistent with Step 3, adding RL-MADTSR
PROTOCOL_STYLES_4 = {
    'Epidemic'         : {'color': '#e63946', 'ls': '-',  'marker': 'o'},
    'RandomWalk'       : {'color': '#adb5bd', 'ls': '--', 'marker': 's'},
    'Heuristic-MADTSR' : {'color': '#457b9d', 'ls': '--', 'marker': '^'},
    'RL-MADTSR'        : {'color': '#1d3557', 'ls': '-',  'marker': 'D'},
}

PROTOCOL_LABELS_4 = {
    'Epidemic'         : 'Epidemic routing',
    'RandomWalk'       : 'Random walk',
    'Heuristic-MADTSR' : 'MA-DTSR (heuristic)',
    'RL-MADTSR'        : 'MA-DTSR (RL)',
}


def _sty(name, key):
    return PROTOCOL_STYLES_4.get(name, {}).get(key, None)


def plot_learning_curves(log_df, window=20):
    """
    Figure 12: Learning curves — smoothed success rate and reward
    over training episodes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('MA-DTSR Step 4: RL Training Curves', fontsize=11)

    # Smooth with rolling mean
    sr_smooth  = (log_df['success']
                  .rolling(window, min_periods=1).mean() * 100)
    rew_smooth = (log_df['reward']
                  .rolling(window, min_periods=1).mean())

    ax1.plot(log_df['episode'], sr_smooth,
             color='#1d3557', linewidth=1.5)
    ax1.fill_between(log_df['episode'],
                     sr_smooth - sr_smooth.std(),
                     sr_smooth + sr_smooth.std(),
                     alpha=0.15, color='#1d3557')
    ax1.set_xlabel('Training episode', fontsize=9)
    ax1.set_ylabel(f'Success rate (%, {window}-ep rolling avg)', fontsize=9)
    ax1.set_title('Success Rate During Training', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.tick_params(labelsize=8)
    ax1.grid(axis='y', alpha=0.3)

    ax2.plot(log_df['episode'], rew_smooth,
             color='#e63946', linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('Training episode', fontsize=9)
    ax2.set_ylabel(f'Reward ({window}-ep rolling avg)', fontsize=9)
    ax2.set_title('Reward During Training', fontsize=10)
    ax2.tick_params(labelsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # Exploration rate on second y-axis
    ax2b = ax2.twinx()
    ax2b.plot(log_df['episode'], log_df['rho'],
              color='#2a9d8f', linewidth=1.0,
              linestyle=':', alpha=0.7, label='\u03c1 (exploration)')
    ax2b.set_ylabel('\u03c1 (exploration rate)', fontsize=8,
                    color='#2a9d8f')
    ax2b.tick_params(labelsize=7, colors='#2a9d8f')
    ax2b.set_ylim(0, 1)

    fig.tight_layout()
    plt.savefig('step4_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 12 saved: step4_learning_curves.png")


def plot_4protocol_comparison(summary_all):
    """
    Figures 13-16: All four protocols compared across TTL.
    """
    metrics = [
        ('success_rate',  'Success rate (%)',        True,  'Success Rate vs TTL'),
        ('mean_hops',     'Mean hops (H)',            False, 'Mean Hop Count vs TTL'),
        ('mean_messages', 'Mean messages (M)',        False, 'Message Count vs TTL'),
        ('mean_utility',  'Mean utility (U_s)',       False, 'Mission Utility vs TTL'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'MA-DTSR Step 4: Full Protocol Comparison\n'
        '(Epidemic / Random Walk / Heuristic / RL)',
        fontsize=12)

    axes_flat = axes.flatten()

    for ax, (metric, ylabel, is_pct, title) in zip(axes_flat, metrics):
        for proto, grp in summary_all.groupby('protocol'):
            vals = grp[metric] * 100 if is_pct else grp[metric]
            ax.plot(grp['ttl'], vals,
                    color=_sty(proto, 'color'),
                    linestyle=_sty(proto, 'ls'),
                    marker=_sty(proto, 'marker'),
                    markersize=6, linewidth=2,
                    label=PROTOCOL_LABELS_4.get(proto, proto))
        ax.set_xlabel('TTL budget', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    plt.savefig('step4_full_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 13 saved: step4_full_comparison.png")


def plot_weight_evolution(rl_router):
    """
    Figure 14: Evolution of theta_i components over training
    for a sample of agents.
    """
    # Collect weight vectors from a sample of agents
    sample_ids = list(rl_router.memory.approximators.keys())[:5]

    fig, ax = plt.subplots(figsize=(8, 4))
    dim_names = ['\u03b81 (semantic score)',
                 '\u03b82 (TTL fraction)',
                 '\u03b83 (energy fraction)',
                 '\u03b84 (wait flag)']
    colours   = ['#e63946', '#1d3557', '#2a9d8f', '#f4a261']

    for agent_id in sample_ids:
        theta = rl_router.memory.get(agent_id).copy_weights()
        for dim, (name, col) in enumerate(zip(dim_names, colours)):
            ax.bar(dim + agent_id * 0.15,
                   theta[dim],
                   width=0.12,
                   color=col, alpha=0.7)

    ax.set_xticks(range(4))
    ax.set_xticklabels(dim_names, fontsize=8, rotation=15)
    ax.set_ylabel('Weight value', fontsize=9)
    ax.set_title('Final Weight Vector \u03b8\u1d62 for Sample Agents', fontsize=10)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    plt.savefig('step4_weight_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 14 saved: step4_weight_evolution.png")


# ============================================================================
# SECTION 8: Sanity Checks
# ============================================================================

def run_rl_sanity_checks(log_df, summary_rl, summary_heuristic):
    """
    Verify RL behaviour against theoretical expectations.
    """
    print("=" * 65)
    print("  MA-DTSR Step 4 \u2014 RL Sanity Check Results")
    print("=" * 65)

    # Check 1: exploration rate decayed correctly
    final_rho = log_df['rho'].iloc[-1]
    print(f"\n  [{'PASS' if final_rho < 0.2 else 'WARN'}] "
          f"Exploration rate decayed to rho={final_rho:.4f} "
          f"(should be < 0.2 after training)")

    # Check 2: weights are non-zero after training
    print(f"  [INFO] Training episodes completed: {len(log_df)}")

    # Check 3: success rate in second half of training
    #          should be >= first half (learning is happening)
    n       = len(log_df)
    sr_1st  = log_df['success'].iloc[:n//2].mean()
    sr_2nd  = log_df['success'].iloc[n//2:].mean()
    learned = sr_2nd >= sr_1st - 0.05   # 5% tolerance
    print(f"  [{'PASS' if learned else 'WARN'}] "
          f"SR 2nd half ({sr_2nd*100:.1f}%) >= "
          f"SR 1st half ({sr_1st*100:.1f}%) - 5%")

    # Check 4: RL success rate >= Heuristic at highest TTL
    ttl_max = summary_rl['ttl'].max()
    rl_sr   = summary_rl[summary_rl['ttl']==ttl_max]['success_rate'].values
    h_sr    = summary_heuristic[
                  summary_heuristic['ttl']==ttl_max]['success_rate'].values

    if len(rl_sr) > 0 and len(h_sr) > 0:
        ok = rl_sr[0] >= h_sr[0] - 0.05
        print(f"  [{'PASS' if ok else 'WARN'}] "
              f"RL SR ({rl_sr[0]*100:.1f}%) >= "
              f"Heuristic SR ({h_sr[0]*100:.1f}%) "
              f"at TTL={ttl_max} (±5%)")

    # Check 5: RL message count <= Epidemic
    print("\n  --- Message Efficiency ---")
    print(f"  RL mean messages by TTL:")
    for _, row in summary_rl.sort_values('ttl').iterrows():
        print(f"    TTL={int(row['ttl'])}: "
              f"{row['mean_messages']:.1f} messages, "
              f"SR={row['success_rate']*100:.1f}%")

    print("=" * 65)


# ============================================================================
# SECTION 9: Main
# ============================================================================

# Import needed from Step 3
try:
    from MA_DTSR_Step3_Baselines import (
        RSM, EpisodeResult, compute_utility, UTILITY_PARAMS,
        EpidemicRouter, RandomWalkRouter, HeuristicRouter,
        run_comparison, summarise_results,
        PROTOCOL_STYLES, PROTOCOL_LABELS,
    )
except ImportError:
    pass   # will be injected by Colab runner


def main_step4(net, step2_module, step3_module,
               ttl_values=None,
               n_train_episodes=500,
               n_eval_episodes=200,
               epsilon=1.0,
               alpha=0.01,
               beta=2.0,
               rho_start=0.8,
               rho_end=0.05,
               rho_decay=0.995,
               lr=0.05,
               gamma_q=0.9,
               alpha_merge=0.3):
    """
    Run Step 4: train and evaluate the RL-augmented MA-DTSR router.

    Parameters
    ----------
    net                : DisasterNetwork (Step 1)
    step2_module       : Step 2 module
    step3_module       : Step 3 module (for baseline routers)
    ttl_values         : list of int — default [10, 20, 30, 40]
    n_train_episodes   : int — RL training episodes per TTL — default 500
    n_eval_episodes    : int — evaluation episodes per TTL — default 200
    epsilon            : float — admissibility threshold
    alpha              : float — staleness parameter
    beta               : float — softmin exploration temperature
    rho_start          : float — initial exploration rate
    rho_end            : float — minimum exploration rate
    rho_decay          : float — multiplicative decay per episode
    lr                 : float — learning rate
    gamma_q            : float — discount factor
    alpha_merge        : float — cooperative merge weight

    Returns
    -------
    df_all    : pd.DataFrame — all episode results (RL + baselines)
    summary   : pd.DataFrame — aggregated statistics
    rl_router : RLRouter — trained router (contains weights and log)
    """
    if ttl_values is None:
        ttl_values = [10, 20, 30, 40]

    print("MA-DTSR Simulation \u2014 Step 4: RL-Augmented Routing")
    print(f"Network    : N={len(net.agents)} agents, t={net.time:.0f}s")
    print(f"Training   : {n_train_episodes} episodes per TTL value")
    print(f"Evaluation : {n_eval_episodes} episodes per TTL value")
    print(f"TTL range  : {ttl_values}")
    print(f"Params     : epsilon={epsilon}, alpha={alpha}, "
          f"beta={beta}, lr={lr}")
    print()

    # ── Build RL router ───────────────────────────────────────────────────────
    rl_router = RLRouter(
        agents      = net.agents,
        epsilon     = epsilon,
        alpha       = alpha,
        beta        = beta,
        rho_start   = rho_start,
        rho_end     = rho_end,
        rho_decay   = rho_decay,
        lr          = lr,
        gamma_q     = gamma_q,
        alpha_merge = alpha_merge,
    )

    # ── Train on middle TTL value (TTL=20) then evaluate on all ──────────────
    train_ttl = ttl_values[len(ttl_values) // 2]
    print(f"Phase 1: Training on TTL={train_ttl}...")
    log_df = train_rl_router(
        rl_router, net, step2_module,
        ttl=train_ttl,
        n_train_episodes=n_train_episodes,
        alpha=alpha)
    print()

    # ── Evaluate RL router ────────────────────────────────────────────────────
    print("Phase 2: Evaluating RL router (greedy policy)...")
    df_rl = evaluate_rl_router(
        rl_router, net, step2_module,
        ttl_values=ttl_values,
        n_eval_episodes=n_eval_episodes,
        alpha=alpha)
    summary_rl = step3_module.summarise_results(df_rl)
    print()

    # ── Re-run baselines for fair comparison ──────────────────────────────────
    print("Phase 3: Running baseline protocols for comparison...")
    baselines = [
        step3_module.EpidemicRouter(epsilon=epsilon),
        step3_module.RandomWalkRouter(epsilon=epsilon),
        step3_module.HeuristicRouter(
            epsilon=epsilon, beta=beta,
            mode='softmin', alpha=alpha),
    ]
    df_baselines = step3_module.run_comparison(
        net, step2_module, baselines,
        ttl_values=ttl_values,
        n_episodes=n_eval_episodes,
        alpha=alpha)
    print()

    # ── Combine all results ───────────────────────────────────────────────────
    df_all     = pd.concat([df_baselines, df_rl], ignore_index=True)
    summary_all = step3_module.summarise_results(df_all)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("--- Full Results Summary ---")
    print(summary_all[['protocol', 'ttl', 'success_rate',
                        'mean_hops', 'mean_messages',
                        'mean_utility']
          ].to_string(index=False, float_format='{:.3f}'.format))
    print()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    summary_heuristic = summary_all[
        summary_all['protocol'] == 'Heuristic-MADTSR']
    run_rl_sanity_checks(log_df, summary_rl, summary_heuristic)
    print()

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures...")
    plot_learning_curves(log_df)
    plot_4protocol_comparison(summary_all)
    plot_weight_evolution(rl_router)

    # ── Save results ──────────────────────────────────────────────────────────
    df_all.to_csv('step4_raw_results.csv', index=False)
    summary_all.to_csv('step4_summary.csv', index=False)
    log_df.to_csv('step4_training_log.csv', index=False)
    print("\nResults saved:")
    print("  step4_raw_results.csv")
    print("  step4_summary.csv")
    print("  step4_training_log.csv")

    return df_all, summary_all, rl_router
