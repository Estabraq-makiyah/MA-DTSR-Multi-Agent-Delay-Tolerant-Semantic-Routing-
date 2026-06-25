# ============================================================================
# MA-DTSR Simulation — Step 2: Resource Descriptors and Contact Database
# ============================================================================
# IMPORTANT: Run Step 1 first in the same Colab session.
# This step assumes the `net` object from Step 1 is already in memory.
#
# What this step produces:
#   1. Full 8-dimensional resource descriptor schema (Section 3.10.1)
#   2. Proper descriptor population for all resource agents
#   3. Query generation — task queries Q_s in R^8
#   4. Three similarity methods: L1, Euclidean, Weighted L1 (Section 3.10.2)
#   5. Masked distance for partial queries (eq. 15)
#   6. Age-aware distance D_alpha (eq. 4) — verified with staleness test
#   7. Contact database population test
#   8. Visual tests — descriptor space and similarity distributions
#
# Everything here maps directly to Section 3.2, 3.3, and 3.10 of the paper.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm

# net is passed in via main_step2(net) — see bottom of file


# ============================================================================
# SECTION 1: Descriptor Schema
# ============================================================================
# This implements Table 1 from Section 3.10.1 of the paper exactly.
# Each dimension is defined with its value set and normalisation rule.

# Dimension 1: Resource category
# {medical=0, transport=1, sensing=2, compute=3, shelter=4} -> normalised /4
CATEGORIES     = ['medical', 'transport', 'sensing', 'compute', 'shelter']
CATEGORY_VALS  = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# Dimension 5: Mobility type
# {static=0, mobile=1}
MOBILITY_VALS  = np.array([0.0, 1.0])

# Dimension 6: Platform type
# {handheld=0, vehicle=1, UAV=2, server=3} -> normalised /3
PLATFORMS      = ['handheld', 'vehicle', 'UAV', 'server']
PLATFORM_VALS  = np.array([0.0, 0.33, 0.67, 1.0])

# Dimension 8: Agency origin
# {police=0, fire=1, medical=2, military=3, NGO=4} -> normalised /4
AGENCIES       = ['police', 'fire', 'medical', 'military', 'NGO']
AGENCY_VALS    = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# Dimension weights for Weighted L1 (D_W, Section 3.10.2)
# Category and urgency matter most in typical SAR queries
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
# SECTION 2: Descriptor Generation Functions
# ============================================================================

def generate_resource_descriptor(agent, rng):
    """
    Generate a full 8-dimensional resource descriptor for an agent
    that holds a resource. Implements Section 3.10.1.

    Parameters
    ----------
    agent : Agent   — the resource-holding agent
    rng   : numpy Generator

    Returns
    -------
    descriptor : np.ndarray, shape (8,), all values in [0, 1]
    meta       : dict with human-readable labels for inspection
    """
    # Dim 1: resource category (categorical)
    cat_idx  = rng.integers(0, len(CATEGORIES))
    dim1     = CATEGORY_VALS[cat_idx]

    # Dim 2: quantity available (continuous)
    # Modelled as a random availability fraction; 0 = depleted, 1 = full
    dim2     = rng.uniform(0.2, 1.0)

    # Dim 3: location zone (from actual agent position)
    # Uses the location_zone() method defined in Step 1
    dim3     = agent.location_zone(n_zones=10)

    # Dim 4: urgency level (continuous)
    # For resources, urgency reflects how critically the resource is needed
    # at this location. In field use this would be operator-assigned.
    dim4     = rng.uniform(0.0, 1.0)

    # Dim 5: mobility type (binary)
    # Static resources (medical caches, servers) vs mobile (vehicles, UAVs)
    # Category-informed: transport and UAV tend to be mobile
    if cat_idx in [1, 2]:    # transport, sensing → likely mobile
        dim5 = rng.choice([0.0, 1.0], p=[0.2, 0.8])
    else:
        dim5 = rng.choice([0.0, 1.0], p=[0.7, 0.3])

    # Dim 6: platform type (categorical)
    # Category-informed: sensing → UAV, compute → server, etc.
    if cat_idx == 2:          # sensing → UAV
        plat_idx = rng.choice([1, 2], p=[0.3, 0.7])
    elif cat_idx == 3:        # compute → server or vehicle
        plat_idx = rng.choice([1, 3], p=[0.4, 0.6])
    elif cat_idx == 1:        # transport → vehicle
        plat_idx = rng.choice([0, 1], p=[0.2, 0.8])
    else:                     # medical, shelter → handheld or vehicle
        plat_idx = rng.choice([0, 1], p=[0.6, 0.4])
    dim6     = PLATFORM_VALS[plat_idx]

    # Dim 7: energy state (continuous, self-reported)
    # Use the agent's actual energy attribute from Step 1
    dim7     = agent.energy

    # Dim 8: agency origin (categorical)
    agency_idx = rng.integers(0, len(AGENCIES))
    dim8       = AGENCY_VALS[agency_idx]

    descriptor = np.array([dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8])

    meta = {
        'category' : CATEGORIES[cat_idx],
        'quantity' : round(dim2, 3),
        'zone'     : round(dim3, 3),
        'urgency'  : round(dim4, 3),
        'mobility' : 'mobile' if dim5 == 1.0 else 'static',
        'platform' : PLATFORMS[plat_idx],
        'energy'   : round(dim7, 3),
        'agency'   : AGENCIES[agency_idx],
    }

    return descriptor, meta


def generate_query(rng, category=None, mask=None):
    """
    Generate a task query Q_s in R^8.

    Parameters
    ----------
    rng      : numpy Generator
    category : int or None
        If given, fix dimension 1 to that category value.
        If None, sample randomly.
    mask     : np.ndarray of shape (8,) with values {0, 1} or None
        1 = this dimension is relevant to the query.
        0 = this dimension is irrelevant (masked out in distance computation).
        If None, all dimensions are relevant (mask = all ones).

    Returns
    -------
    query : np.ndarray, shape (8,)
    mask  : np.ndarray, shape (8,)
    """
    if category is not None:
        dim1 = CATEGORY_VALS[category]
    else:
        dim1 = rng.choice(CATEGORY_VALS)

    query = np.array([
        dim1,                          # dim 1: what category we need
        rng.uniform(0.3, 1.0),        # dim 2: minimum quantity needed
        rng.uniform(0.0, 1.0),        # dim 3: preferred location zone
        rng.uniform(0.5, 1.0),        # dim 4: urgency of need (SAR → usually high)
        rng.choice(MOBILITY_VALS),    # dim 5: mobility preference
        rng.choice(PLATFORM_VALS),    # dim 6: platform preference
        0.5,                           # dim 7: energy not queried (midpoint)
        rng.choice(AGENCY_VALS),      # dim 8: agency preference
    ])

    if mask is None:
        mask = np.ones(8)

    return query, mask


# ============================================================================
# SECTION 3: Similarity Methods
# ============================================================================
# Implements Section 3.10.2 — three metrics, all supporting masking (eq. 15)

def d_l1(query, descriptor, mask=None):
    """
    L1 (Manhattan) distance — eq. (2) with optional masking (eq. 15).
    Baseline metric used throughout the formal model.
    """
    if mask is None:
        mask = np.ones(len(query))
    return np.sum(mask * np.abs(query - descriptor))


def d_l2(query, descriptor, mask=None):
    """
    Euclidean distance with optional masking (eq. 15).
    """
    if mask is None:
        mask = np.ones(len(query))
    return np.sqrt(np.sum(mask * (query - descriptor) ** 2))


def d_weighted(query, descriptor, weights=None, mask=None):
    """
    Weighted L1 distance with optional masking (eq. 15).
    Uses DIM_WEIGHTS by default.
    """
    if weights is None:
        weights = DIM_WEIGHTS
    if mask is None:
        mask = np.ones(len(query))
    return np.sum(weights * mask * np.abs(query - descriptor))


def d_alpha(query, descriptor, timestamp, current_time, alpha, metric='l1',
            mask=None, weights=None):
    """
    Age-aware distance D_alpha from eq. (4).

    D_alpha(Q_s, R-hat_j(t); t) = exp(alpha * (t - tau_j)) * dist(Q_s, R-hat_j)

    Parameters
    ----------
    query        : np.ndarray (8,)
    descriptor   : np.ndarray (8,)
    timestamp    : float   — time descriptor was last updated (tau_j)
    current_time : float   — current simulation time (t)
    alpha        : float   — staleness penalty >= 0
    metric       : str     — 'l1', 'l2', or 'weighted'
    mask         : np.ndarray (8,) or None
    weights      : np.ndarray (8,) or None   — used when metric='weighted'

    Returns
    -------
    float — age-aware distance
    """
    age       = current_time - timestamp
    staleness = np.exp(alpha * age)

    if metric == 'l1':
        base_dist = d_l1(query, descriptor, mask)
    elif metric == 'l2':
        base_dist = d_l2(query, descriptor, mask)
    elif metric == 'weighted':
        base_dist = d_weighted(query, descriptor, weights, mask)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'l1', 'l2', or 'weighted'.")

    return staleness * base_dist


def is_admissible(query, descriptor, epsilon, mask=None):
    """
    Check admissibility condition from eq. (3):
    D_L1(Q_s, R_i) <= epsilon

    Always uses unmasked L1 for admissibility check, as per the paper.
    """
    return d_l1(query, descriptor, mask=None) <= epsilon


# ============================================================================
# SECTION 4: Populate Descriptors on the Network
# ============================================================================

def populate_descriptors(net, seed=None):
    """
    Replace the placeholder descriptors from Step 1 with full
    8-dimensional descriptors using the schema above.

    This modifies agents in-place and also stores metadata
    for inspection and visualisation.

    Returns
    -------
    meta_records : list of dicts — one per resource agent
    """
    rng = np.random.default_rng(seed if seed is not None else 99)
    meta_records = []

    for agent in net.agents:
        if agent.resource:
            descriptor, meta = generate_resource_descriptor(agent, rng)
            agent.descriptor = descriptor
            meta['agent_id'] = agent.agent_id
            meta['pos_x']    = agent.pos[0]
            meta['pos_y']    = agent.pos[1]
            meta_records.append(meta)

    print(f"Descriptors populated for {len(meta_records)} resource agents.")
    return meta_records


def rebuild_contact_database(net):
    """
    Rebuild the contact database from scratch now that descriptors
    are properly populated. This triggers a full descriptor exchange
    between all currently linked agent pairs.

    In the live simulation this happens incrementally at each time step.
    Here we do it in one pass to initialise the database cleanly.
    """
    # Reset the database
    from MA_DTSR_Step1_Mobility import ContactDatabase
    net.contact_db = ContactDatabase(net.agents)

    # Trigger updates for all currently active links
    for agent_i in net.agents:
        for j_id in net.neighbours.get(agent_i.agent_id, []):
            agent_j = net.agent_map[j_id]
            net.contact_db.update(agent_i.agent_id, agent_j, net.time)
            net.contact_db.update(agent_j.agent_id, agent_i, net.time)

    # Count populated entries
    total = sum(len(v) for v in net.contact_db.db.values())
    print(f"Contact database rebuilt: {total} entries across {len(net.agents)} agents.")


# ============================================================================
# SECTION 5: Visualisations
# ============================================================================

def plot_descriptor_space(meta_records):
    """
    Visualise the distribution of resource descriptors across
    two of the most important dimensions: category and urgency.

    Each dot is one resource agent. Colour = category, size = quantity.
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('MA-DTSR: Resource Descriptor Space', fontsize=11)

    # ── Plot 1: urgency vs location zone, coloured by category ───────────────
    ax = axes[0]
    for cat, grp in df.groupby('category'):
        ax.scatter(grp['zone'], grp['urgency'],
                   c=cat_colours.get(cat, '#999'),
                   s=grp['quantity'] * 80 + 20,
                   alpha=0.75, label=cat, edgecolors='white', linewidths=0.4)
    ax.set_xlabel('Location zone (dim 3)', fontsize=9)
    ax.set_ylabel('Urgency (dim 4)', fontsize=9)
    ax.set_title('Urgency vs Location Zone\n(size = quantity)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # ── Plot 2: category distribution ────────────────────────────────────────
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
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(val), ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    plt.savefig('step2_descriptor_space.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 4 saved: step2_descriptor_space.png")


def plot_similarity_distributions(net, meta_records, n_queries=200):
    """
    For a sample of random queries, compute the distance to every
    resource agent using all three similarity methods.

    Plots the distribution of distances to show how the metrics differ.
    """
    rng = np.random.default_rng(77)

    resource_agents = [a for a in net.agents if a.resource and a.descriptor is not None]
    if not resource_agents:
        print("No resource agents with descriptors.")
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
        f'({n_queries} random queries × {len(resource_agents)} resource agents)',
        fontsize=10)

    specs = [
        (l1_dists, '#1d3557', 'L1 (Manhattan) distance',     'D_L1'),
        (l2_dists, '#457b9d', 'Euclidean distance',           'D_L2'),
        (dw_dists, '#2a9d8f', 'Weighted L1 distance',         'D_W'),
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
    plt.savefig('step2_similarity_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 5 saved: step2_similarity_distributions.png")


def plot_staleness_effect(net, n_steps=50):
    """
    Show how age-aware distance D_alpha grows with descriptor age,
    for different values of alpha. Validates eq. (4).
    """
    rng      = np.random.default_rng(55)
    query, _ = generate_query(rng)

    # Pick a resource agent with a descriptor
    resource_agents = [a for a in net.agents if a.resource and a.descriptor is not None]
    if not resource_agents:
        print("No resource agents available for staleness test.")
        return
    agent = resource_agents[0]

    # Freeze descriptor at timestamp 0
    timestamp    = 0.0
    times        = np.arange(0, n_steps + 1, 1.0)
    alphas       = [0.0, 0.01, 0.05, 0.10]
    base_dist    = d_l1(query, agent.descriptor)

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ['#adb5bd', '#457b9d', '#1d3557', '#e63946']

    for alpha, colour in zip(alphas, colours):
        d_vals = [
            d_alpha(query, agent.descriptor, timestamp, t, alpha, metric='l1')
            for t in times
        ]
        ax.plot(times, d_vals, color=colour, linewidth=1.8,
                label=f'\u03b1 = {alpha}')

    ax.axhline(y=base_dist, color='#f4a261', linewidth=1,
               linestyle='--', label=f'Base L1 distance = {base_dist:.3f}')
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
    """
    For each agent, plot how many neighbours have descriptors
    stored in their contact database vs total neighbours.
    Shows the coverage of the contact database after rebuild.
    """
    coverage = []
    for agent in net.agents:
        total_nbrs   = len(net.neighbours.get(agent.agent_id, []))
        known_nbrs   = len(net.contact_db.db.get(agent.agent_id, {}))
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

    # Plot 1: known vs total neighbours
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
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Add legend for colours
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e63946',
               markersize=7, label='Resource agent'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1d3557',
               markersize=7, label='Non-resource agent'),
    ]
    ax.legend(handles=legend_elements, fontsize=7)

    # Plot 2: coverage % distribution
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
# SECTION 6: Sanity Checks
# ============================================================================

def run_descriptor_sanity_checks(net, meta_records, epsilon=1.0):
    """
    Verify the descriptor schema and similarity functions are
    behaving as expected.
    """
    rng = np.random.default_rng(33)

    resource_agents = [a for a in net.agents
                       if a.resource and a.descriptor is not None]

    print("=" * 60)
    print("  MA-DTSR Step 2 — Descriptor Sanity Check Results")
    print("=" * 60)
    print(f"  Resource agents with descriptors : {len(resource_agents)}")
    print(f"  Admissibility threshold epsilon  : {epsilon}")
    print()

    # Check 1: descriptor values all in [0, 1]
    all_in_range = all(
        (a.descriptor >= 0).all() and (a.descriptor <= 1).all()
        for a in resource_agents
    )
    print(f"  [{'PASS' if all_in_range else 'FAIL'}] All descriptor values in [0, 1]")

    # Check 2: dimension 3 (location zone) matches actual position
    zone_ok = all(
        abs(a.descriptor[2] - a.location_zone()) < 1e-6
        for a in resource_agents
    )
    print(f"  [{'PASS' if zone_ok else 'FAIL'}] Dim 3 (location zone) matches agent position")

    # Check 3: dimension 7 (energy) matches agent energy attribute
    energy_ok = all(
        abs(a.descriptor[6] - a.energy) < 1e-9
        for a in resource_agents
    )
    print(f"  [{'PASS' if energy_ok else 'FAIL'}] Dim 7 (energy) matches agent.energy")

    # Check 4: self-distance is 0
    agent = resource_agents[0]
    self_dist = d_l1(agent.descriptor, agent.descriptor)
    print(f"  [{'PASS' if self_dist == 0 else 'FAIL'}] "
          f"Self-distance (L1) = {self_dist:.6f} (should be 0)")

    # Check 5: staleness at age 0 equals base distance
    query, _ = generate_query(rng)
    base      = d_l1(query, agent.descriptor)
    at_zero   = d_alpha(query, agent.descriptor, 0.0, 0.0, alpha=0.05)
    print(f"  [{'PASS' if abs(base - at_zero) < 1e-9 else 'FAIL'}] "
          f"D_alpha(age=0) == D_L1 = {base:.4f}")

    # Check 6: staleness increases with age
    at_10  = d_alpha(query, agent.descriptor, 0.0, 10.0, alpha=0.05)
    at_50  = d_alpha(query, agent.descriptor, 0.0, 50.0, alpha=0.05)
    mono   = at_zero <= at_10 <= at_50
    print(f"  [{'PASS' if mono else 'FAIL'}] "
          f"D_alpha monotonically increases: "
          f"{at_zero:.3f} <= {at_10:.3f} <= {at_50:.3f}")

    # Check 7: masking zeroes out irrelevant dimensions
    full_mask    = np.ones(8)
    partial_mask = np.array([1, 1, 0, 1, 0, 0, 0, 0])   # only dims 1,2,4
    d_full       = d_l1(query, agent.descriptor, full_mask)
    d_partial    = d_l1(query, agent.descriptor, partial_mask)
    print(f"  [{'PASS' if d_partial <= d_full else 'FAIL'}] "
          f"Masked distance <= full distance "
          f"({d_partial:.3f} <= {d_full:.3f})")

    # Check 8: admissibility rate at epsilon
    n_queries   = 500
    n_admissible = 0
    for _ in range(n_queries):
        q, _ = generate_query(rng)
        for a in resource_agents:
            if is_admissible(q, a.descriptor, epsilon):
                n_admissible += 1
                break
    admissible_rate = n_admissible / n_queries
    print(f"  [INFO] Admissibility rate at epsilon={epsilon}: "
          f"{admissible_rate*100:.1f}% of queries find a match")
    if admissible_rate < 0.05:
        print("  [WARN] Very low admissibility — consider increasing epsilon.")
    elif admissible_rate > 0.95:
        print("  [WARN] Very high admissibility — epsilon may be too loose.")
    else:
        print("  [PASS] Admissibility rate is within realistic range.")

    # Check 9: contact database has entries
    n_entries = sum(len(v) for v in net.contact_db.db.values())
    print(f"  [{'PASS' if n_entries > 0 else 'FAIL'}] "
          f"Contact database has {n_entries} entries")

    # Summary: category distribution
    print()
    print("  --- Category Distribution ---")
    df = pd.DataFrame(meta_records)
    for cat, count in df['category'].value_counts().items():
        bar = '█' * count
        print(f"  {cat:12s} : {bar} ({count})")

    print()
    print("  --- Platform Distribution ---")
    for plat, count in df['platform'].value_counts().items():
        bar = '█' * count
        print(f"  {plat:12s} : {bar} ({count})")

    print("=" * 60)
    print("  Step 2 complete. Descriptors and contact database ready.")
    print("  Next step: run Step 3 (baseline routing protocols).")
    print("=" * 60)


# ============================================================================
# SECTION 7: Main — Run Everything
# ============================================================================

def main_step2(net):
    print("MA-DTSR Simulation — Step 2: Resource Descriptors")
    print(f"Network: N={len(net.agents)} agents, t={net.time:.0f}s")
    print()

    # ── 1. Populate descriptors ───────────────────────────────────────────────
    print("Populating 8-dimensional resource descriptors...")
    meta_records = populate_descriptors(net, seed=99)

    # ── 2. Rebuild contact database ───────────────────────────────────────────
    print("Rebuilding contact database with populated descriptors...")
    rebuild_contact_database(net)
    print()

    # ── 3. Run sanity checks ──────────────────────────────────────────────────
    run_descriptor_sanity_checks(net, meta_records, epsilon=1.0)
    print()

    # ── 4. Visualisations ─────────────────────────────────────────────────────
    print("Generating figures...")
    plot_descriptor_space(meta_records)
    plot_similarity_distributions(net, meta_records, n_queries=200)
    plot_staleness_effect(net, n_steps=60)
    plot_contact_database_coverage(net)

    print()
    print("Step 2 complete. Four figures generated (Figures 4–7).")
    print("The 'net' object is updated in place with full descriptors.")
    print("Keep this session open — Step 3 builds directly on top.")

    return meta_records


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # When run directly (not imported), net must be provided externally.
    # In Colab: call main_step2(net) after running Step 1.
    raise RuntimeError(
        "Run this as a module from Colab: 'meta = step2.main_step2(net)'"
    )
