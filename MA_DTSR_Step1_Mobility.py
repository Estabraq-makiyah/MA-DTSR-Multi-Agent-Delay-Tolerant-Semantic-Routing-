# ============================================================================
# MA-DTSR Simulation — Step 1: Network and Mobility Model
# ============================================================================
# Paste this entire file into a Google Colab cell and run it.
# It builds and tests the disaster network mobility model that underpins
# the full MA-DTSR simulation.
#
# What this step produces:
#   1. Agent class  — position, movement, random waypoint mobility
#   2. DisasterNetwork class — manages all agents, links, and time steps
#   3. Visual test  — animated snapshot of the network at several time steps
#   4. Sanity checks — degree distribution, connectivity statistics
#
# All parameters match the formal model in Section 3 of the paper.
# ============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
from tqdm import tqdm

# Fix random seed for reproducibility in testing
# (removed in the full simulation where we sweep seeds)
MASTER_SEED = 42

# ============================================================================
# SECTION 1: Constants and Configuration
# ============================================================================
# These match the parameter grid defined in the paper (Section 5).
# You can change these to test different configurations.

CONFIG = {
    # Network size
    'N'           : 100,        # number of agents (paper: 100, 200, 300)
    'grid_size'   : 1000,       # simulation area in metres (1000 x 1000)
    'comm_range'  : 100,        # communication range in metres

    # Mobility (Random Waypoint model)
    'speed_min'   : 0.5,        # minimum agent speed (m/s)
    'speed_max'   : 2.0,        # maximum agent speed (m/s)
    'pause_min'   : 0.0,        # minimum pause time at waypoint (seconds)
    'pause_max'   : 5.0,        # maximum pause time at waypoint (seconds)
    'dt'          : 1.0,        # simulation time step (seconds)

    # Contact database (Section 3.3)
    'K'           : 15,         # max neighbours per agent (paper: 10,15,20,25)
    'alpha'       : 0.01,       # staleness penalty parameter (eq. 4)

    # Resource assignment
    'resource_frac': 0.30,      # fraction of agents that hold a resource

    # Simulation duration for the mobility test
    'T_test'      : 100,        # time steps for the visual test
}


# ============================================================================
# SECTION 2: Agent Class
# ============================================================================

class Agent:
    """
    Represents one autonomous field device in the disaster network.
    Implements Random Waypoint mobility as the movement model.

    Attributes
    ----------
    agent_id    : unique integer identifier
    pos         : current (x, y) position in the grid
    dest        : current waypoint destination
    speed       : current movement speed (m/s)
    pause_timer : seconds remaining at current waypoint pause
    resource    : True if this agent holds a resource, else False
    descriptor  : numpy array in R^8 (populated in Step 2)
    energy      : remaining energy fraction [0, 1]
    """

    def __init__(self, agent_id, config, rng):
        self.agent_id    = agent_id
        self.config      = config
        self.rng         = rng

        # Initialise position uniformly at random in the grid
        self.pos         = rng.uniform(0, config['grid_size'], size=2)

        # Pick a random initial waypoint destination
        self.dest        = rng.uniform(0, config['grid_size'], size=2)

        # Pick a random initial speed
        self.speed       = rng.uniform(config['speed_min'], config['speed_max'])

        # No initial pause
        self.pause_timer = 0.0

        # Resource assignment — filled by DisasterNetwork.__init__
        self.resource    = False
        self.descriptor  = None         # numpy array, shape (8,)

        # Energy — starts full, decreases with transmission (used in Step 4+)
        self.energy      = rng.uniform(0.6, 1.0)

    # ── Movement ─────────────────────────────────────────────────────────────

    def step(self):
        """
        Advance agent position by one time step (dt seconds).
        Implements the Random Waypoint model:
          - If pausing at a waypoint, count down the pause timer.
          - Otherwise move toward the destination at current speed.
          - On arrival, pick a new destination, new speed, new pause.
        """
        dt = self.config['dt']

        # Count down pause
        if self.pause_timer > 0:
            self.pause_timer -= dt
            return

        # Vector toward destination
        direction = self.dest - self.pos
        dist_to_dest = np.linalg.norm(direction)

        # Check if we arrive this step
        if dist_to_dest <= self.speed * dt:
            # Arrive at destination
            self.pos = self.dest.copy()
            # Pick new waypoint
            self._new_waypoint()
        else:
            # Move toward destination
            unit_dir = direction / dist_to_dest
            self.pos += unit_dir * self.speed * dt

    def _new_waypoint(self):
        """Pick a new random destination, speed, and pause duration."""
        cfg = self.config
        rng = self.rng
        self.dest        = rng.uniform(0, cfg['grid_size'], size=2)
        self.speed       = rng.uniform(cfg['speed_min'], cfg['speed_max'])
        self.pause_timer = rng.uniform(cfg['pause_min'], cfg['pause_max'])

    # ── Location zone (maps position to a discrete grid cell index) ──────────

    def location_zone(self, n_zones=10):
        """
        Map (x, y) position to a discrete zone index in [0, n_zones^2 - 1].
        Used to populate descriptor dimension 3 (location zone, Section 3.10.1).
        """
        gs    = self.config['grid_size']
        cell_size = gs / n_zones
        col   = int(self.pos[0] / cell_size)
        row   = int(self.pos[1] / cell_size)
        col   = min(col, n_zones - 1)
        row   = min(row, n_zones - 1)
        return (row * n_zones + col) / (n_zones ** 2 - 1)   # normalised to [0,1]

    def __repr__(self):
        return (f'Agent({self.agent_id}, '
                f'pos=({self.pos[0]:.1f},{self.pos[1]:.1f}), '
                f'resource={self.resource})')


# ============================================================================
# SECTION 3: Contact Database
# ============================================================================

class ContactDatabase:
    """
    Maintains each agent's local knowledge of neighbour descriptors,
    implementing Section 3.3 of the paper.

    Structure:
        db[i][j] = {
            'descriptor' : R^8 numpy array  (R-hat_j(t)),
            'timestamp'  : float            (tau_j(t))
        }
    """

    def __init__(self, agents):
        self.db = {a.agent_id: {} for a in agents}

    def update(self, agent_i_id, agent_j, current_time):
        """
        Record agent j's current descriptor as seen by agent i.
        Called whenever i and j are within communication range.
        """
        if agent_j.descriptor is not None:
            self.db[agent_i_id][agent_j.agent_id] = {
                'descriptor' : agent_j.descriptor.copy(),
                'timestamp'  : current_time
            }

    def get_entry(self, agent_i_id, agent_j_id):
        """Return the stored entry for j as known by i, or None."""
        return self.db[agent_i_id].get(agent_j_id, None)

    def age_aware_distance(self, query, agent_i_id, agent_j_id,
                           current_time, alpha):
        """
        Compute D_alpha(Q_s, R-hat_j(t); t) from eq. (4).

        D_alpha = exp(alpha * (t - tau_j)) * ||Q_s - R-hat_j||_1
        """
        entry = self.get_entry(agent_i_id, agent_j_id)
        if entry is None:
            return np.inf       # unknown neighbour — treat as infinitely far

        age       = current_time - entry['timestamp']
        staleness = np.exp(alpha * age)
        l1_dist   = np.sum(np.abs(query - entry['descriptor']))
        return staleness * l1_dist


# ============================================================================
# SECTION 4: Disaster Network
# ============================================================================

class DisasterNetwork:
    """
    Manages the full agent population, their mobility, link formation,
    and contact database updates.

    This is the environment that all four routing protocols run inside.

    Parameters
    ----------
    config : dict   — simulation configuration (see CONFIG above)
    seed   : int    — random seed for reproducibility
    """

    def __init__(self, config=None, seed=MASTER_SEED):
        if config is None:
            config = CONFIG

        self.config  = config
        self.rng     = np.random.default_rng(seed)
        self.time    = 0.0

        # Create agents
        self.agents  = [
            Agent(i, config, np.random.default_rng(seed + i))
            for i in range(config['N'])
        ]
        self.agent_map = {a.agent_id: a for a in self.agents}

        # Assign resources to a fraction of agents
        self._assign_resources()

        # Initialise contact database
        self.contact_db = ContactDatabase(self.agents)

        # Compute initial links
        self.neighbours = {}        # {agent_id: [neighbour_ids]}
        self._update_links()

        # Statistics log — grows with each time step
        self.stats_log = []

    # ── Resource assignment ───────────────────────────────────────────────────

    def _assign_resources(self):
        """
        Randomly assign resources to a fraction of agents.
        Descriptors are populated here with placeholder values;
        Step 2 will replace this with the full 8-dimensional schema.
        """
        n_resources = int(self.config['N'] * self.config['resource_frac'])
        resource_agents = self.rng.choice(
            self.agents, size=n_resources, replace=False
        )

        for agent in resource_agents:
            agent.resource = True
            # Placeholder 8-dim descriptor — Step 2 replaces this
            agent.descriptor = self.rng.uniform(0, 1, size=8)
            # Set dimension 3 (location zone) from actual position
            agent.descriptor[2] = agent.location_zone()

    # ── Link formation ────────────────────────────────────────────────────────

    def _update_links(self):
        """
        Form links between all agent pairs within comm_range.
        Updates self.neighbours and triggers contact database updates.

        Complexity: O(N^2) — acceptable for N <= 300.
        For larger N, a spatial index (e.g. KD-tree) would be used.
        """
        r   = self.config['comm_range']
        N   = len(self.agents)
        nbrs = {a.agent_id: [] for a in self.agents}

        for i in range(N):
            for j in range(i + 1, N):
                ai = self.agents[i]
                aj = self.agents[j]
                dist = np.linalg.norm(ai.pos - aj.pos)
                if dist <= r:
                    nbrs[ai.agent_id].append(aj.agent_id)
                    nbrs[aj.agent_id].append(ai.agent_id)

                    # Update contact databases in both directions
                    self.contact_db.update(ai.agent_id, aj, self.time)
                    self.contact_db.update(aj.agent_id, ai, self.time)

        self.neighbours = nbrs

    # ── Time step ─────────────────────────────────────────────────────────────

    def step(self):
        """
        Advance the simulation by one time step (dt seconds):
          1. Move all agents (Random Waypoint)
          2. Recompute links
          3. Update contact databases
          4. Log statistics
        """
        # 1. Move all agents
        for agent in self.agents:
            agent.step()

        self.time += self.config['dt']

        # 2 & 3. Recompute links and update contact DB
        self._update_links()

        # 4. Log statistics
        self._log_stats()

    def run(self, T):
        """Run the simulation for T time steps."""
        for _ in tqdm(range(T), desc='Simulating', unit='step'):
            self.step()

    # ── Statistics ────────────────────────────────────────────────────────────

    def _log_stats(self):
        """Record per-step network statistics."""
        degrees      = [len(nbrs) for nbrs in self.neighbours.values()]
        n_links      = sum(degrees) // 2
        avg_degree   = np.mean(degrees)
        max_degree   = np.max(degrees)
        isolated     = sum(1 for d in degrees if d == 0)

        # Largest connected component (simple BFS)
        lcc_size     = self._largest_component()

        self.stats_log.append({
            'time'       : self.time,
            'n_links'    : n_links,
            'avg_degree' : avg_degree,
            'max_degree' : max_degree,
            'isolated'   : isolated,
            'lcc_frac'   : lcc_size / len(self.agents),
        })

    def _largest_component(self):
        """BFS to find the size of the largest connected component."""
        visited = set()
        largest = 0

        for start in self.agent_map:
            if start in visited:
                continue
            # BFS from start
            queue     = [start]
            component = set()
            while queue:
                node = queue.pop(0)
                if node in component:
                    continue
                component.add(node)
                for nbr in self.neighbours.get(node, []):
                    if nbr not in component:
                        queue.append(nbr)
            visited |= component
            largest = max(largest, len(component))

        return largest

    def get_stats_df(self):
        """Return logged statistics as a pandas DataFrame."""
        return pd.DataFrame(self.stats_log)

    # ── Snapshot for visualisation ────────────────────────────────────────────

    def snapshot(self):
        """Return current state for visualisation."""
        return {
            'time'       : self.time,
            'positions'  : np.array([a.pos for a in self.agents]),
            'has_resource': np.array([a.resource for a in self.agents]),
            'neighbours' : dict(self.neighbours),
            'agents'     : self.agents,
        }


# ============================================================================
# SECTION 5: Visualisation
# ============================================================================

def plot_network_snapshot(snap, ax, title=''):
    """
    Plot a single network snapshot:
      - Blue dots  = agents without resources
      - Red dots   = agents with resources
      - Grey lines = active communication links
    """
    positions   = snap['positions']
    has_res     = snap['has_resource']
    neighbours  = snap['neighbours']
    agents      = snap['agents']

    ax.set_facecolor('#f8f9fa')
    ax.set_xlim(0, CONFIG['grid_size'])
    ax.set_ylim(0, CONFIG['grid_size'])
    ax.set_aspect('equal')

    # Draw links first (behind nodes)
    drawn_links = set()
    for agent in agents:
        i = agent.agent_id
        for j in neighbours.get(i, []):
            link_key = (min(i, j), max(i, j))
            if link_key not in drawn_links:
                drawn_links.add(link_key)
                xi, yi = positions[i]
                xj, yj = positions[j]
                ax.plot([xi, xj], [yi, yj],
                        color='#adb5bd', linewidth=0.4, alpha=0.6, zorder=1)

    # Draw agents
    colors = ['#e63946' if r else '#1d3557' for r in has_res]
    ax.scatter(positions[:, 0], positions[:, 1],
               c=colors, s=18, zorder=2, edgecolors='white', linewidths=0.3)

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel('x (m)', fontsize=8)
    ax.set_ylabel('y (m)', fontsize=8)
    ax.tick_params(labelsize=7)


def plot_stats(stats_df, ax_deg, ax_lcc):
    """Plot degree and connectivity statistics over time."""
    t = stats_df['time']

    ax_deg.plot(t, stats_df['avg_degree'], color='#1d3557',
                linewidth=1.5, label='Mean degree')
    ax_deg.fill_between(t,
                        stats_df['avg_degree'] - 0.5,
                        stats_df['avg_degree'] + 0.5,
                        alpha=0.15, color='#1d3557')
    ax_deg.axhline(y=CONFIG['K'], color='#e63946', linewidth=1,
                   linestyle='--', label=f"K = {CONFIG['K']}")
    ax_deg.set_xlabel('Time (s)', fontsize=8)
    ax_deg.set_ylabel('Degree', fontsize=8)
    ax_deg.set_title('Mean Node Degree over Time', fontsize=9)
    ax_deg.legend(fontsize=7)
    ax_deg.tick_params(labelsize=7)

    ax_lcc.plot(t, stats_df['lcc_frac'] * 100, color='#457b9d',
                linewidth=1.5)
    ax_lcc.set_xlabel('Time (s)', fontsize=8)
    ax_lcc.set_ylabel('LCC size (% of N)', fontsize=8)
    ax_lcc.set_title('Largest Connected Component over Time', fontsize=9)
    ax_lcc.set_ylim(0, 105)
    ax_lcc.tick_params(labelsize=7)


# ============================================================================
# SECTION 6: Sanity Checks
# ============================================================================

def run_sanity_checks(net):
    """
    Print key statistics to verify the network is behaving correctly.

    Expected values for N=100, comm_range=100, grid=1000x1000:
      - Mean degree: 2 – 8  (depends on mobility snapshot)
      - Isolated nodes: 0 – 20% at any given time step
      - LCC: typically 60 – 90% of N
    """
    stats = net.get_stats_df()
    print("=" * 55)
    print("  MA-DTSR Step 1 — Sanity Check Results")
    print("=" * 55)
    print(f"  N agents          : {CONFIG['N']}")
    print(f"  Grid size         : {CONFIG['grid_size']} x {CONFIG['grid_size']} m")
    print(f"  Comm range        : {CONFIG['comm_range']} m")
    print(f"  Time steps run    : {len(stats)}")
    print(f"  Resource agents   : {sum(1 for a in net.agents if a.resource)}")
    print()
    print("  --- Degree Statistics ---")
    print(f"  Mean avg degree   : {stats['avg_degree'].mean():.2f}  "
          f"(target: 2–8)")
    print(f"  Max avg degree    : {stats['avg_degree'].max():.2f}")
    print(f"  Min avg degree    : {stats['avg_degree'].min():.2f}")
    print()
    print("  --- Connectivity ---")
    print(f"  Mean LCC fraction : {stats['lcc_frac'].mean()*100:.1f}%  "
          f"(target: 60–90%)")
    print(f"  Mean isolated     : {stats['isolated'].mean():.1f} agents")
    print()
    print("  --- Links ---")
    print(f"  Mean link count   : {stats['n_links'].mean():.1f}")
    print(f"  Max link count    : {stats['n_links'].max()}")
    print("=" * 55)

    # Flag potential problems
    mean_deg = stats['avg_degree'].mean()
    mean_lcc = stats['lcc_frac'].mean()

    warnings = []
    if mean_deg < 1.0:
        warnings.append("WARNING: Mean degree < 1. Increase comm_range or N.")
    if mean_deg > 20:
        warnings.append("WARNING: Mean degree > 20. Decrease comm_range.")
    if mean_lcc < 0.4:
        warnings.append("WARNING: LCC < 40%. Network is too fragmented.")
    if mean_lcc > 0.99:
        warnings.append("WARNING: LCC > 99%. Network is almost fully connected "
                        "— reduce comm_range for a more realistic scenario.")

    if warnings:
        print()
        for w in warnings:
            print(f"  {w}")
    else:
        print("  All checks passed. Network parameters look good.")
    print("=" * 55)


# ============================================================================
# SECTION 7: Main — Run Everything
# ============================================================================

def main():
    print("MA-DTSR Simulation — Step 1: Mobility Model")
    print(f"Config: N={CONFIG['N']}, "
          f"grid={CONFIG['grid_size']}m, "
          f"range={CONFIG['comm_range']}m, "
          f"T={CONFIG['T_test']} steps")
    print()

    # ── Build and run the network ────────────────────────────────────────────
    net = DisasterNetwork(config=CONFIG, seed=MASTER_SEED)

    # Capture snapshots at three points in time for visualisation
    snap_times = [0, CONFIG['T_test'] // 2, CONFIG['T_test'] - 1]
    snapshots  = []
    snap_idx   = 0

    # Capture t=0 snapshot before running
    if 0 in snap_times:
        snapshots.append(net.snapshot())
        snap_idx += 1

    # Run simulation
    for t_step in tqdm(range(1, CONFIG['T_test']),
                       desc='Running mobility simulation', unit='step'):
        net.step()
        if t_step in snap_times[snap_idx:]:
            snapshots.append(net.snapshot())
            snap_idx += 1

    # ── Figure 1: Network snapshots ──────────────────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig1.suptitle(
        f'MA-DTSR: Disaster Network Snapshots\n'
        f'N={CONFIG["N"]} agents, comm range={CONFIG["comm_range"]}m, '
        f'grid={CONFIG["grid_size"]}×{CONFIG["grid_size"]}m',
        fontsize=10, y=1.01
    )

    for i, (snap, ax) in enumerate(zip(snapshots, axes)):
        t_label = int(snap['time'])
        n_links = sum(len(v) for v in snap['neighbours'].values()) // 2
        plot_network_snapshot(
            snap, ax,
            title=f't = {t_label}s  |  links = {n_links}'
        )

    # Legend
    legend_elements = [
        mpatches.Patch(color='#e63946', label='Agent with resource'),
        mpatches.Patch(color='#1d3557', label='Agent without resource'),
        Line2D([0], [0], color='#adb5bd', linewidth=1, label='Active link'),
    ]
    fig1.legend(handles=legend_elements, loc='lower center',
                ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig1.tight_layout()
    plt.savefig('step1_network_snapshots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 1 saved: step1_network_snapshots.png")

    # ── Figure 2: Statistics over time ───────────────────────────────────────
    stats_df = net.get_stats_df()

    fig2, (ax_deg, ax_lcc) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig2.suptitle('MA-DTSR: Network Statistics over Time', fontsize=10)
    plot_stats(stats_df, ax_deg, ax_lcc)
    fig2.tight_layout()
    plt.savefig('step1_network_stats.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 2 saved: step1_network_stats.png")

    # ── Sanity checks ────────────────────────────────────────────────────────
    print()
    run_sanity_checks(net)

    # ── Figure 3: Degree distribution at final time step ─────────────────────
    final_degrees = [len(net.neighbours[a.agent_id]) for a in net.agents]

    fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
    ax3.hist(final_degrees, bins=range(0, max(final_degrees) + 2),
             color='#1d3557', edgecolor='white', linewidth=0.5, alpha=0.85)
    ax3.axvline(x=np.mean(final_degrees), color='#e63946',
                linewidth=1.5, linestyle='--',
                label=f'Mean = {np.mean(final_degrees):.1f}')
    ax3.axvline(x=CONFIG['K'], color='#f4a261',
                linewidth=1.5, linestyle=':',
                label=f"K (max neighbours) = {CONFIG['K']}")
    ax3.set_xlabel('Node degree', fontsize=9)
    ax3.set_ylabel('Number of agents', fontsize=9)
    ax3.set_title(f'Degree Distribution at t = {int(net.time)}s', fontsize=10)
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    plt.savefig('step1_degree_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 3 saved: step1_degree_distribution.png")

    print()
    print("Step 1 complete. Three figures generated.")
    print("Next step: run Step 2 (resource descriptors and contact database).")

    return net


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    net = main()
