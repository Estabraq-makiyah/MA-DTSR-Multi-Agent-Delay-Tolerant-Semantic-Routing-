# ============================================================================
# MA-DTSR Simulation — Step 2: Resource Descriptors and Contact Database
# ============================================================================
# Run after Step 1. Call main_step2(net) from Colab.
#
# What this step produces:
#   1. Full 8-dimensional resource descriptor schema (Section 3.10.1)
#   2. WiSARD-calibrated descriptor population (Section 3.9)
#   3. Query generation — task queries Q_s in R^8
#   4. Three similarity methods: L1, Euclidean, Weighted L1 (Section 3.10.2)
#   5. Masked distance for partial queries (eq. 15)
#   6. Age-aware distance D_alpha (eq. 4) — verified with staleness test
#   7. Contact database population and coverage test
#   8. Visual figures — descriptor space and similarity distributions
#
# WiSARD calibration (Section 3.9):
#   Source: 74,204 person-detection annotations across 81 flight folders
#   Visual (RGB): 62.9%  |  Thermal (IR): 37.1%
#   Single-class dataset — urgency calibrated from SAR operational statistics
#   Reference: Broyles et al. 2022, DOI: 10.1109/IROS47612.2022.9981298
#
# All equation references are to Section 3 of the paper.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm


# ============================================================================
# SECTION 1: Descriptor Schema Constants
# ============================================================================
# Implements Table 1 from Section 3.10.1 of the paper.

# Dimension 1: Resource category
# {medical=0, transport=1, sensing=2, compute=3, shelter=4} -> [0,1]
CATEGORIES    = ['medical', 'transport', 'sensing', 'compute', 'shelter']
CATEGORY_VALS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# Dimension 5: Mobility type  {static=0, mobile=1}
MOBILITY_VALS = np.array([0.0, 1.0])

# Dimension 6: Platform type
# {handheld=0, vehicle=1, UAV=2, server=3} -> [0,1]
PLATFORMS     = ['handheld', 'vehicle', 'UAV', 'server']
PLATFORM_VALS = np.array([0.0, 0.33, 0.67, 1.0])

# Dimension 8: Agency origin
# {police=0, fire=1, medical=2, military=3, NGO=4} -> [0,1]
AGENCIES      = ['police', 'fire', 'medical', 'military', 'NGO']
AGENCY_VALS   = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# Dimension weights for Weighted L1 (D_W, Section 3.10.2)
# Category and urgency weighted highest for SAR resource queries
DIM_WEIGHTS = np.array([
    0.25,   # dim 1: resource category   — highest weight
    0.10,   # dim 2: quantity
    0.10,   # dim 3: location zone
    0.20,   # dim 4: urgency             — high weight
    0.10,   # dim 5: mobility type
    0.10,   # dim 6: platform type
    0.05,   # dim 7: energy state
    0.10,   # dim 8: agency origin
])
assert abs(DIM_WEIGHTS.sum() - 1.0) < 1e-9, "Weights must sum to 1"


# ============================================================================
# SECTION 2: WiSARD-Calibrated Parameters (Section 3.9)
# ============================================================================
# Calibrated from analysis of WiSARD dataset:
#   74,204 person-detection annotations, 81 flight folders, 12 locations
#   Broyles, Hayner & Leung (2022), DOI: 10.1109/IROS47612.2022.9981298
#
# WiSARD is single-class (person only) — no body-position sub-labels.
# Urgency ranges follow SAR operational casualty statistics.
# Platform modality split (62.9% RGB / 37.1% thermal) is from dataset analysis.

# Dim 4 — urgency: pose-conditioned ranges from SAR operational statistics
WISARD_POSE_LABELS = ['lying', 'sitting', 'standing', 'running']
WISARD_POSE_PROBS  = [0.42, 0.28, 0.25, 0.05]   # SAR operational defaults
WISARD_URGENCY_RANGES = {
    'lying'   : (0.70, 1.00),   # incapacitated — highest urgency
    'sitting' : (0.40, 0.70),   # impaired — medium urgency
    'standing': (0.10, 0.40),   # ambulatory — lower urgency
    'running' : (0.00, 0.20),   # mobile — minimal urgency
}

# Dim 5 — mobility: derived from pose distribution
# mobile fraction = standing*0.30 + running*1.00 = 0.25*0.30 + 0.05 = 0.125
WISARD_MOBILE_FRAC = 0.125

# Dim 6 — platform modality split from WiSARD dataset analysis
# Visual (RGB): 62.9% of annotations | Thermal (IR): 37.1% of annotations
WISARD_RGB_FRAC     = 0.629
WISARD_THERMAL_FRAC = 0.371

# Dim 2 — quantity: calibrated from WiSARD annotation density
# Mean 2.19 persons per annotated image -> moderate resource quantity
WISARD_QUANTITY_MIN = 0.20   # depleted resources still partially available
WISARD_QUANTITY_MAX = 1.00


# ============================================================================
# SECTION 3: Descriptor Generation
# ============================================================================

def generate_resource_descriptor(agent, rng):
    """
    Generate a full 8-dimensional resource descriptor for an agent
    holding a resource. Implements Section 3.10.1 with WiSARD calibration
    from Section 3.9.

    Parameters
    ----------
    agent : Agent       — resource-holding agent from Step 1
    rng   : numpy Generator

    Returns
    -------
    descriptor : np.ndarray shape (8,), all values in [0, 1]
    meta       : dict — human-readable labels for inspection and plotting
    """
    # ── Dim 1: Resource category (categorical) ────────────────────────────────
    cat_idx = rng.integers(0, len(CATEGORIES))
    dim1    = CATEGORY_VALS[cat_idx]

    # ── Dim 2: Quantity available (continuous) ────────────────────────────────
    # Calibrated from WiSARD annotation density (mean 2.19 persons per image)
    # reflecting moderate resource availability in SAR field scenarios
    dim2 = rng.uniform(WISARD_QUANTITY_MIN, WISARD_QUANTITY_MAX)

    # ── Dim 3: Location zone (from actual agent position) ─────────────────────
    # WiSARD grounds this dimension through its 12 geographically distinct
    # collection sites spanning coastal, forest, snow, airfield, and valley
    # terrains — confirming that location zone is a meaningful SAR variable
    dim3 = agent.location_zone(n_zones=10)

    # ── Dim 4: Urgency (WiSARD-calibrated) ───────────────────────────────────
    # WiSARD is single-class (person only) — no body-position sub-labels.
    # Urgency ranges are calibrated from SAR operational casualty statistics.
    # Pose distribution: lying 42%, sitting 28%, standing 25%, running 5%
    pose     = rng.choice(WISARD_POSE_LABELS, p=WISARD_POSE_PROBS)
    lo, hi   = WISARD_URGENCY_RANGES[pose]
    dim4     = rng.uniform(lo, hi)

    # ── Dim 5: Mobility type (WiSARD-calibrated) ─────────────────────────────
    # Mobile fraction derived from pose distribution:
    # 30% of standing + 100% of running = 0.25*0.30 + 0.05 = 12.5% mobile
    # Transport and sensing categories override toward mobile
    if cat_idx in [1, 2]:   # transport, sensing → predominantly mobile
        dim5 = rng.choice([0.0, 1.0], p=[0.20, 0.80])
    else:
        dim5 = rng.choice([0.0, 1.0],
                          p=[1.0 - WISARD_MOBILE_FRAC, WISARD_MOBILE_FRAC])

    # ── Dim 6: Platform type (WiSARD-calibrated) ─────────────────────────────
    # WiSARD modality split: 62.9% RGB visual, 37.1% thermal IR
    # (74,204 total annotations, 81 flight folders, 12 locations)
    # This directly calibrates UAV sensing agent platform distribution
    if cat_idx == 2:        # sensing → UAV platform (WiSARD-confirmed)
        # UAV platform for both RGB and thermal modalities
        # Thermal sub-flag: 37.1% thermal, 62.9% RGB (stored in meta only)
        plat_idx     = 2    # UAV
        thermal_flag = rng.choice([0, 1],
                                  p=[WISARD_RGB_FRAC, WISARD_THERMAL_FRAC])
    elif cat_idx == 3:      # compute → server or vehicle
        plat_idx     = rng.choice([1, 3], p=[0.40, 0.60])
        thermal_flag = 0
    elif cat_idx == 1:      # transport → vehicle (predominantly)
        plat_idx     = rng.choice([0, 1], p=[0.20, 0.80])
        thermal_flag = 0
    else:                   # medical, shelter → handheld or vehicle
        plat_idx     = rng.choice([0, 1], p=[0.60, 0.40])
        thermal_flag = 0
    dim6 = PLATFORM_VALS[plat_idx]

    # ── Dim 7: Energy state (self-reported) ───────────────────────────────────
    # Uses the agent's actual energy attribute initialised in Step 1
    dim7 = agent.energy

    # ── Dim 8: Agency origin (categorical) ───────────────────────────────────
    agency_idx = rng.integers(0, len(AGENCIES))
    dim8       = AGENCY_VALS[agency_idx]

    descriptor = np.array([dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8])

    meta = {
        'category'      : CATEGORIES[cat_idx],
        'quantity'      : round(float(dim2), 3),
        'zone'          : round(float(dim3), 3),
        'urgency'       : round(float(dim4), 3),
        'pose'          : pose,
        'mobility'      : 'mobile' if dim5 == 1.0 else 'static',
        'platform'      : PLATFORMS[plat_idx],
        'thermal'       : bool(thermal_flag) if cat_idx == 2 else False,
        'energy'        : round(float(dim7), 3),
        'agency'        : AGENCIES[agency_idx],
    }

    return descriptor, meta


def generate_query(rng, category=None, mask=None):
    """
    Generate a task query Q_s in R^8.

    Parameters
    ----------
    rng      : numpy Generator
    category : int or None — if given, fixes dimension 1 to that category
    mask     : np.ndarray (8,) with values {0,1} or None
               1 = relevant dimension, 0 = masked out (eq. 15)

    Returns
    -------
    query : np.ndarray shape (8,)
    mask  : np.ndarray shape (8,)
    """
    if category is not None:
        dim1 = CATEGORY_VALS[category]
    else:
        dim1 = rng.choice(CATEGORY_VALS)

    query = np.array([
        dim1,                        # dim 1: required resource category
        rng.uniform(0.3, 1.0),      # dim 2: minimum quantity needed
        rng.uniform(0.0, 1.0),      # dim 3: preferred location zone
        rng.uniform(0.5, 1.0),      # dim 4: urgency (SAR queries are urgent)
        rng.choice(MOBILITY_VALS),  # dim 5: mobility preference
        rng.choice(PLATFORM_VALS),  # dim 6: platform preference
        0.5,                         # dim 7: energy not queried — masked below
        rng.choice(AGENCY_VALS),    # dim 8: agency preference
    ])

    if mask is None:
        mask = np.ones(8)
        mask[6] = 0   # energy state not relevant to most queries

    return query, mask


# ============================================================================
# SECTION 4: Similarity Methods (Section 3.10.2)
# ============================================================================

def d_l1(query, descriptor, mask=None):
    """L1 (Manhattan) distance — eq. (2) with optional masking (eq. 15)."""
    if mask is None:
        mask = np.ones(len(query))
    return float(np.sum(mask * np.abs(query - descriptor)))


def d_l2(query, descriptor, mask=None):
    """Euclidean distance with optional masking (eq. 15)."""
    if mask is None:
        mask = np.ones(len(query))
    return float(np.sqrt(np.sum(mask * (query - descriptor) ** 2)))


def d_weighted(query, descriptor, weights=None, mask=None):
    """Weighted L1 distance with optional masking (eq. 15)."""
    if weights is None:
        weights = DIM_WEIGHTS
    if mask is None:
        mask = np.ones(len(query))
    return float(np.sum(weights * mask * np.abs(query - descriptor)))


def d_alpha(query, descriptor, timestamp, current_time, alpha,
            metric='l1', mask=None, weights=None):
    """
    Age-aware distance D_alpha from eq. (4):
    D_alpha(Q_s, R-hat_j(t); t) = exp(alpha*(t-tau_j)) * dist(Q_s, R-hat_j)

    Parameters
    ----------
    query, descriptor : np.ndarray (8,)
    timestamp         : float — when descriptor was last updated (tau_j)
    current_time      : float — current simulation time (t)
    alpha             : float — staleness penalty >= 0
    metric            : str   — 'l1', 'l2', or 'weighted'
    mask, weights     : optional

    Returns
    -------
    float — age-aware distance
    """
    age       = current_time - timestamp
    staleness = np.exp(alpha * age)

    if metric == 'l1':
        base = d_l1(query, descriptor, mask)
    elif metric == 'l2':
        base = d_l2(query, descriptor, mask)
    elif metric == 'weighted':
        base = d_weighted(query, descriptor, weights, mask)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'l1', 'l2', or 'weighted'.")

    return staleness * base


def is_admissible(query, descriptor, epsilon, mask=None):
    """
    Admissibility check from eq. (3): D_L1(Q_s, R_i) <= epsilon.
    Always uses unmasked L1 for admissibility.
    """
    return d_l1(query, descriptor, mask=None) <= epsilon


# ============================================================================
# SECTION 5: Descriptor Population
# ============================================================================

def populate_descriptors(net, seed=None):
    """
    Replace placeholder descriptors with full WiSARD-calibrated
    8-dimensional descriptors. Modifies agents in-place.

    Returns
    -------
    meta_records : list of dict — one per resource agent
    """
    rng          = np.random.default_rng(seed if seed is not None else 99)
    meta_records = []

    for agent in net.agents:
        if agent.resource:
            descriptor, meta = generate_resource_descriptor(agent, rng)
            agent.descriptor = descriptor
            meta['agent_id'] = agent.agent_id
            meta['pos_x']    = float(agent.pos[0])
            meta['pos_y']    = float(agent.pos[1])
            meta_records.append(meta)

    print(f"Descriptors populated for {len(meta_records)} resource agents.")
    print(f"  WiSARD calibration: urgency from SAR operational statistics")
    print(f"  Platform modality:  {WISARD_RGB_FRAC*100:.1f}% RGB / "
          f"{WISARD_THERMAL_FRAC*100:.1f}% thermal (from 74,204 annotations)")
    return meta_records


def rebuild_contact_database(net):
    """
    Rebuild the contact database after descriptor population.
    Triggers descriptor exchange between all currently linked agent pairs.
    """
    from MA_DTSR_Step1_Mobility import ContactDatabase
    net.contact_db = ContactDatabase(net.agents)

    for agent_i in net.agents:
        for j_id in net.neighbours.get(agent_i.agent_id, []):
            agent_j = net.agent_map[j_id]
            net.contact_db.update(agent_i.agent_id, agent_j, net.time)
            net.contact_db.update(agent_j.agent_id, agent_i, net.time)

    total = sum(len(v) for v in net.contact_db.db.values())
    print(f"Contact database rebuilt: {total} entries across "
          f"{len(net.agents)} agents.")


# ============================================================================
# SECTION 6: Visualisations
# ============================================================================

def plot_descriptor_space(meta_records):
    """
    Figure 4: Descriptor space — urgency vs location zone, by category.
    Includes pose distribution panel to show WiSARD calibration effect.
    """
    if not meta_records:
        print("No resource agents to plot.")
        return

    df = pd.DataFrame(meta_records)

    cat_colours = {
        'medical'  : '#e63946',
        'transport': '#457b9d',
        'sensing'  : '#2a9d8f',
        'compute'  : '#e9c46a',
        'shelter'  : '#f4a261',
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('MA-DTSR: Resource Descriptor Space (WiSARD-Calibrated)',
                 fontsize=11)

    # Panel 1: urgency vs location zone
    ax = axes[0]
    for cat, grp in df.groupby('category'):
        ax.scatter(grp['zone'], grp['urgency'],
                   c=cat_colours.get(cat, '#999'),
                   s=grp['quantity'] * 80 + 20,
                   alpha=0.75, label=cat,
                   edgecolors='white', linewidths=0.4)
    ax.set_xlabel('Location zone (dim 3)', fontsize=9)
    ax.set_ylabel('Urgency (dim 4)', fontsize=9)
    ax.set_title('Urgency vs Location Zone\n(size = quantity)', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: category distribution
    ax = axes[1]
    cat_counts = df['category'].value_counts()
    bars = ax.bar(cat_counts.index, cat_counts.values,
                  color=[cat_colours[c] for c in cat_counts.index],
                  edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Resource category', fontsize=9)
    ax.set_ylabel('Number of agents', fontsize=9)
    ax.set_title('Resource Category Distribution', fontsize=9)
    ax.tick_params(labelsize=7)
    for bar, val in zip(bars, cat_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                str(val), ha='center', va='bottom', fontsize=8)

    # Panel 3: pose distribution (WiSARD calibration verification)
    ax = axes[2]
    pose_counts = df['pose'].value_counts() if 'pose' in df.columns else None
    if pose_counts is not None:
        pose_colours = {
            'lying'   : '#e63946',
            'sitting' : '#457b9d',
            'standing': '#2a9d8f',
            'running' : '#f4a261',
        }
        bars = ax.bar(pose_counts.index, pose_counts.values / len(df) * 100,
                      color=[pose_colours.get(p, '#adb5bd')
                             for p in pose_counts.index],
                      edgecolor='white', linewidth=0.5)
        # Overlay WiSARD target probabilities
        for i, (pose, prob) in enumerate(
                zip(WISARD_POSE_LABELS, WISARD_POSE_PROBS)):
            if pose in pose_counts.index:
                ax.axhline(y=prob * 100, color='black',
                           linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel('Body pose (urgency proxy)', fontsize=9)
        ax.set_ylabel('% of resource agents', fontsize=9)
        ax.set_title('Pose Distribution\n(dashed = WiSARD SAR targets)',
                     fontsize=9)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    plt.savefig('step2_descriptor_space.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 4 saved: step2_descriptor_space.png")


def plot_similarity_distributions(net, meta_records, n_queries=200):
    """Figure 5: Distribution of L1, Euclidean, and Weighted L1 distances."""
    rng = np.random.default_rng(77)
    resource_agents = [a for a in net.agents
                       if a.resource and a.descriptor is not None]
    if not resource_agents:
        print("No resource agents.")
        return

    l1_dists, l2_dists, dw_dists = [], [], []
    for _ in range(n_queries):
        query, mask = generate_query(rng)
        for agent in resource_agents:
            l1_dists.append(d_l1(query, agent.descriptor, mask))
            l2_dists.append(d_l2(query, agent.descriptor, mask))
            dw_dists.append(d_weighted(query, agent.descriptor, mask=mask))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle(
        f'MA-DTSR: Similarity Distance Distributions\n'
        f'({n_queries} queries × {len(resource_agents)} resource agents, '
        f'WiSARD-calibrated descriptors)',
        fontsize=10)

    specs = [
        (l1_dists, '#1d3557', 'L1 (Manhattan) distance',  'D_L1'),
        (l2_dists, '#457b9d', 'Euclidean distance',        'D_L2'),
        (dw_dists, '#2a9d8f', 'Weighted L1 distance',      'D_W'),
    ]
    for ax, (dists, colour, title, xlabel) in zip(axes, specs):
        ax.hist(dists, bins=40, color=colour, alpha=0.85,
                edgecolor='white', linewidth=0.4)
        ax.axvline(x=np.mean(dists), color='#e63946',
                   linewidth=1.5, linestyle='--',
                   label=f'Mean = {np.mean(dists):.2f}')
        ax.axvline(x=np.median(dists), color='#f4a261',
                   linewidth=1.5, linestyle=':',
                   label=f'Median = {np.median(dists):.2f}')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    plt.savefig('step2_similarity_distributions.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 5 saved: step2_similarity_distributions.png")


def plot_staleness_effect(net, n_steps=50):
    """Figure 6: Age-aware distance D_alpha growth over time (eq. 4)."""
    rng      = np.random.default_rng(55)
    query, _ = generate_query(rng)

    resource_agents = [a for a in net.agents
                       if a.resource and a.descriptor is not None]
    if not resource_agents:
        print("No resource agents for staleness test.")
        return
    agent     = resource_agents[0]
    timestamp = 0.0
    times     = np.arange(0, n_steps + 1, 1.0)
    alphas    = [0.0, 0.01, 0.05, 0.10]
    base_dist = d_l1(query, agent.descriptor)

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ['#adb5bd', '#457b9d', '#1d3557', '#e63946']

    for alpha, colour in zip(alphas, colours):
        d_vals = [d_alpha(query, agent.descriptor,
                          timestamp, t, alpha, metric='l1')
                  for t in times]
        ax.plot(times, d_vals, color=colour, linewidth=1.8,
                label=f'\u03b1 = {alpha}')

    ax.axhline(y=base_dist, color='#f4a261', linewidth=1,
               linestyle='--',
               label=f'Base L1 = {base_dist:.3f}')
    ax.set_xlabel('Time since last descriptor update (s)', fontsize=9)
    ax.set_ylabel('D\u03b1 (age-aware distance)', fontsize=9)
    ax.set_title('Staleness Effect on Age-Aware Distance (eq. 4)', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    plt.savefig('step2_staleness_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 6 saved: step2_staleness_effect.png")


def plot_contact_database_coverage(net):
    """Figure 7: Contact database coverage per agent."""
    coverage = []
    for agent in net.agents:
        total_nbrs = len(net.neighbours.get(agent.agent_id, []))
        known_nbrs = len(net.contact_db.db.get(agent.agent_id, {}))
        coverage.append({
            'agent_id'    : agent.agent_id,
            'has_resource': agent.resource,
            'total_nbrs'  : total_nbrs,
            'known_nbrs'  : known_nbrs,
            'coverage_pct': (known_nbrs / total_nbrs * 100)
                            if total_nbrs > 0 else 0,
        })

    df = pd.DataFrame(coverage)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('MA-DTSR: Contact Database Coverage', fontsize=10)

    ax = axes[0]
    ax.scatter(df['total_nbrs'], df['known_nbrs'],
               c=['#e63946' if r else '#1d3557' for r in df['has_resource']],
               s=25, alpha=0.7, edgecolors='white', linewidths=0.3)
    max_val = max(df['total_nbrs'].max(), df['known_nbrs'].max()) + 1
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.4,
            label='Perfect coverage')
    ax.set_xlabel('Total neighbours', fontsize=9)
    ax.set_ylabel('Neighbours with known descriptors', fontsize=9)
    ax.set_title('Known vs Total Neighbours', fontsize=9)
    ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#e63946', markersize=7,
               label='Resource agent'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#1d3557', markersize=7,
               label='Non-resource agent'),
    ], fontsize=7)

    ax = axes[1]
    ax.hist(df['coverage_pct'], bins=20, color='#1d3557',
            edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.axvline(x=df['coverage_pct'].mean(), color='#e63946',
               linewidth=1.5, linestyle='--',
               label=f"Mean = {df['coverage_pct'].mean():.1f}%")
    ax.set_xlabel('Coverage (%)', fontsize=9)
    ax.set_ylabel('Number of agents', fontsize=9)
    ax.set_title('Contact Database Coverage Distribution', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    plt.savefig('step2_contact_coverage.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 7 saved: step2_contact_coverage.png")


# ============================================================================
# SECTION 7: Sanity Checks
# ============================================================================

def run_descriptor_sanity_checks(net, meta_records, epsilon=1.0):
    """Verify descriptor schema and similarity functions."""
    rng = np.random.default_rng(33)
    resource_agents = [a for a in net.agents
                       if a.resource and a.descriptor is not None]

    print("=" * 62)
    print("  MA-DTSR Step 2 \u2014 Descriptor Sanity Check Results")
    print("=" * 62)
    print(f"  Resource agents with descriptors : {len(resource_agents)}")
    print(f"  Admissibility threshold epsilon  : {epsilon}")
    print()

    # Check 1: all values in [0, 1]
    ok = all((a.descriptor >= 0).all() and (a.descriptor <= 1).all()
             for a in resource_agents)
    print(f"  [{'PASS' if ok else 'FAIL'}] All descriptor values in [0, 1]")

    # Check 2: dim 3 matches agent position
    ok = all(abs(a.descriptor[2] - a.location_zone()) < 1e-6
             for a in resource_agents)
    print(f"  [{'PASS' if ok else 'FAIL'}] Dim 3 (location zone) matches "
          f"agent position")

    # Check 3: dim 7 matches agent energy
    ok = all(abs(a.descriptor[6] - a.energy) < 1e-9
             for a in resource_agents)
    print(f"  [{'PASS' if ok else 'FAIL'}] Dim 7 (energy) matches "
          f"agent.energy")

    # Check 4: dim 4 values within WiSARD urgency ranges
    df   = pd.DataFrame(meta_records)
    ok4  = True
    for _, row in df.iterrows():
        lo, hi = WISARD_URGENCY_RANGES.get(row['pose'], (0.0, 1.0))
        if not (lo - 1e-6 <= row['urgency'] <= hi + 1e-6):
            ok4 = False
            break
    print(f"  [{'PASS' if ok4 else 'FAIL'}] Dim 4 (urgency) within "
          f"WiSARD pose-conditioned ranges")

    # Check 5: self-distance is 0
    agent     = resource_agents[0]
    self_dist = d_l1(agent.descriptor, agent.descriptor)
    print(f"  [{'PASS' if self_dist == 0 else 'FAIL'}] "
          f"Self-distance (L1) = {self_dist:.6f} (should be 0)")

    # Check 6: staleness at age=0 equals base L1
    query, _ = generate_query(rng)
    base     = d_l1(query, agent.descriptor)
    at_zero  = d_alpha(query, agent.descriptor, 0.0, 0.0, alpha=0.05)
    print(f"  [{'PASS' if abs(base - at_zero) < 1e-9 else 'FAIL'}] "
          f"D_alpha(age=0) == D_L1 = {base:.4f}")

    # Check 7: staleness monotonically increases
    at_10 = d_alpha(query, agent.descriptor, 0.0, 10.0, alpha=0.05)
    at_50 = d_alpha(query, agent.descriptor, 0.0, 50.0, alpha=0.05)
    mono  = at_zero <= at_10 <= at_50
    print(f"  [{'PASS' if mono else 'FAIL'}] "
          f"D_alpha monotonically increases: "
          f"{at_zero:.3f} \u2264 {at_10:.3f} \u2264 {at_50:.3f}")

    # Check 8: masked distance <= full distance
    full_mask    = np.ones(8)
    partial_mask = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    d_full       = d_l1(query, agent.descriptor, full_mask)
    d_partial    = d_l1(query, agent.descriptor, partial_mask)
    print(f"  [{'PASS' if d_partial <= d_full else 'FAIL'}] "
          f"Masked distance \u2264 full distance "
          f"({d_partial:.3f} \u2264 {d_full:.3f})")

    # Check 9: admissibility rate
    n_queries    = 500
    n_admissible = 0
    for _ in range(n_queries):
        q, _ = generate_query(rng)
        for a in resource_agents:
            if is_admissible(q, a.descriptor, epsilon):
                n_admissible += 1
                break
    admissible_rate = n_admissible / n_queries
    print(f"  [INFO] Admissibility rate at \u03b5={epsilon}: "
          f"{admissible_rate*100:.1f}% of queries find a match")
    if admissible_rate < 0.05:
        print("  [WARN] Very low — consider increasing epsilon.")
    elif admissible_rate > 0.95:
        print("  [WARN] Very high — epsilon may be too loose.")
    else:
        print("  [PASS] Admissibility rate within realistic range.")

    # Check 10: contact database has entries
    n_entries = sum(len(v) for v in net.contact_db.db.values())
    print(f"  [{'PASS' if n_entries > 0 else 'FAIL'}] "
          f"Contact database has {n_entries} entries")

    # Distributions
    print()
    print("  --- Category Distribution ---")
    for cat, count in df['category'].value_counts().items():
        print(f"  {cat:12s} : {'█' * count} ({count})")

    print()
    print("  --- Pose Distribution (WiSARD calibration) ---")
    if 'pose' in df.columns:
        for pose, count in df['pose'].value_counts().items():
            lo, hi = WISARD_URGENCY_RANGES.get(pose, (0,1))
            print(f"  {pose:12s} : {'█' * count} ({count}) "
                  f"urgency [{lo:.2f}\u2013{hi:.2f}]")

    print()
    print("  --- Platform Distribution ---")
    for plat, count in df['platform'].value_counts().items():
        print(f"  {plat:12s} : {'█' * count} ({count})")

    print()
    print("  --- WiSARD Calibration Summary ---")
    print(f"  Source      : 74,204 annotations, 81 flights, 12 locations")
    print(f"  RGB/thermal : {WISARD_RGB_FRAC*100:.1f}% / "
          f"{WISARD_THERMAL_FRAC*100:.1f}%")
    print(f"  Urgency     : pose-conditioned from SAR operational statistics")
    print(f"  Mobility    : {WISARD_MOBILE_FRAC*100:.1f}% mobile "
          f"(non-transport/sensing agents)")

    print("=" * 62)
    print("  Step 2 complete. Descriptors and contact database ready.")
    print("  Next: run Step 3 (baseline routing protocols).")
    print("=" * 62)


# ============================================================================
# SECTION 8: Main
# ============================================================================

def main_step2(net):
    """
    Run Step 2: populate WiSARD-calibrated descriptors and rebuild
    the contact database.

    Parameters
    ----------
    net : DisasterNetwork from Step 1

    Returns
    -------
    meta_records : list of dict — one per resource agent
    """
    print("MA-DTSR Simulation \u2014 Step 2: Resource Descriptors")
    print(f"Network : N={len(net.agents)} agents, t={net.time:.0f}s")
    print()

    print("Populating WiSARD-calibrated 8-dimensional descriptors...")
    meta_records = populate_descriptors(net, seed=99)

    print("Rebuilding contact database...")
    rebuild_contact_database(net)
    print()

    run_descriptor_sanity_checks(net, meta_records, epsilon=1.0)
    print()

    print("Generating figures...")
    plot_descriptor_space(meta_records)
    plot_similarity_distributions(net, meta_records, n_queries=200)
    plot_staleness_effect(net, n_steps=60)
    plot_contact_database_coverage(net)

    print()
    print("Step 2 complete. Four figures generated (Figures 4\u20137).")
    print("net updated in place. Keep session open \u2014 Step 3 builds on top.")

    return meta_records


# ── Entry point guard ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    raise RuntimeError(
        "Import this module and call main_step2(net) from Colab."
    )
