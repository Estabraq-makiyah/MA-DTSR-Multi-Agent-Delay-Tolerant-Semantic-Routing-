# ============================================================================
# WiSARD Dataset Analysis — Extract Statistics for MA-DTSR Calibration
# ============================================================================
# Upload this script to Colab and run it BEFORE running Steps 1-5.
# It reads the WiSARD annotation files, extracts body-position distributions,
# and produces the calibrated probabilities to use in Step 2.
#
# HOW TO USE:
#   1. Upload your WiSARD folder to Google Drive or Colab /content/
#   2. Set WISARD_ROOT below to point to your extracted folder
#   3. Run this script — it produces:
#        wisard_stats.json        — statistics to feed into Step 2
#        wisard_analysis.png      — visualisation of distributions
#   4. Copy the printed CALIBRATED_PARAMS block into Step 2
#
# WiSARD annotation format:
#   Each image has a corresponding .txt file (YOLO format):
#     <class_id> <x_centre> <y_centre> <width> <height>
#   Class IDs map to body positions — this script auto-detects the mapping
#   from any classes.txt or dataset.yaml file in the WiSARD folder.
# ============================================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
# Set this to wherever you extracted WiSARD
WISARD_ROOT = '/content/WiSARD'   # change if needed

# Known WiSARD body-position class names
# (will be auto-detected, but listed here as fallback)
KNOWN_POSE_LABELS = ['standing', 'sitting', 'lying', 'running',
                     'standing_person', 'lying_person',
                     'person_standing', 'person_lying',
                     'person_sitting', 'person_running']

# Modality subdirectory names to look for
MODALITY_DIRS = {
    'visual'  : ['visual', 'rgb', 'Visual', 'RGB', 'images_rgb'],
    'thermal' : ['thermal', 'lwir', 'Thermal', 'LWIR', 'images_thermal'],
}


# ============================================================================
# SECTION 1: File Discovery
# ============================================================================

def find_annotation_files(root):
    """
    Recursively find all YOLO-format .txt annotation files.
    Returns dict: {'visual': [paths], 'thermal': [paths], 'all': [paths]}
    """
    root = Path(root)
    all_txts = list(root.rglob('*.txt'))

    # Exclude any file named 'classes.txt' or 'labels.txt' etc.
    anno_files = [
        p for p in all_txts
        if not p.stem.lower() in
        ['classes', 'labels', 'notes', 'readme', 'dataset']
    ]

    visual_files  = []
    thermal_files = []

    for p in anno_files:
        path_str = str(p).lower()
        if any(m in path_str for m in ['thermal', 'lwir', 'ir']):
            thermal_files.append(p)
        elif any(m in path_str for m in ['visual', 'rgb', 'color']):
            visual_files.append(p)
        # unclassified go to 'all' only

    print(f"Found annotation files:")
    print(f"  Visual  : {len(visual_files)}")
    print(f"  Thermal : {len(thermal_files)}")
    unclassified = len(anno_files) - len(visual_files) - len(thermal_files)
    print(f"  Other   : {unclassified}")
    print(f"  Total   : {len(anno_files)}")

    return {
        'visual'  : visual_files,
        'thermal' : thermal_files,
        'all'     : anno_files,
    }


def find_class_file(root):
    """
    Find the class definition file (classes.txt or dataset.yaml).
    Returns dict mapping class_id (int) -> class_name (str).
    """
    root = Path(root)

    # Try classes.txt first
    for candidate in root.rglob('classes.txt'):
        with open(candidate) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            mapping = {i: name for i, name in enumerate(lines)}
            print(f"\nClass file found: {candidate}")
            print(f"  Classes: {mapping}")
            return mapping

    # Try dataset.yaml (common in newer YOLO datasets)
    try:
        import yaml
        for candidate in root.rglob('*.yaml'):
            with open(candidate) as f:
                data = yaml.safe_load(f)
            if 'names' in data:
                names = data['names']
                if isinstance(names, list):
                    mapping = {i: n for i, n in enumerate(names)}
                elif isinstance(names, dict):
                    mapping = {int(k): v for k, v in names.items()}
                print(f"\nYAML class file found: {candidate}")
                print(f"  Classes: {mapping}")
                return mapping
    except ImportError:
        pass

    print("\nNo class file found — will infer from annotation content.")
    return None


# ============================================================================
# SECTION 2: Annotation Parser
# ============================================================================

def parse_annotations(file_list, class_map=None):
    """
    Parse YOLO-format annotation files.

    Returns
    -------
    records : list of dict with keys:
        class_id, class_name, x, y, w, h, modality (visual/thermal)
    class_counts : Counter {class_id: count}
    bbox_sizes   : list of (w, h) tuples for spatial analysis
    """
    records     = []
    class_counts= Counter()
    bbox_sizes  = []
    empty_files = 0
    parse_errors= 0

    for fpath in file_list:
        try:
            with open(fpath) as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                empty_files += 1
                continue

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    class_counts[cid] += 1
                    bbox_sizes.append((w, h))

                    name = (class_map[cid]
                            if class_map and cid in class_map
                            else f'class_{cid}')
                    records.append({
                        'class_id'  : cid,
                        'class_name': name,
                        'x'         : x,
                        'y'         : y,
                        'w'         : w,
                        'h'         : h,
                        'file'      : str(fpath),
                    })
                except (ValueError, IndexError):
                    parse_errors += 1

        except Exception:
            parse_errors += 1

    print(f"\nAnnotation parsing complete:")
    print(f"  Records parsed  : {len(records):,}")
    print(f"  Empty files     : {empty_files:,}")
    print(f"  Parse errors    : {parse_errors}")

    return records, class_counts, bbox_sizes


# ============================================================================
# SECTION 3: Statistics Extraction
# ============================================================================

def infer_pose_mapping(class_map, class_counts):
    """
    Given the class map and counts, identify which class IDs
    correspond to body positions relevant to MA-DTSR.

    Returns
    -------
    pose_map : dict {class_id: canonical_pose_label}
    where canonical label is one of:
        'lying', 'sitting', 'standing', 'running', 'person'
    """
    if class_map is None:
        # Single-class dataset — all annotations are 'person'
        most_common_id = class_counts.most_common(1)[0][0]
        return {most_common_id: 'person'}

    pose_map = {}
    for cid, name in class_map.items():
        name_lower = name.lower().replace('-', '_').replace(' ', '_')

        if any(kw in name_lower for kw in ['lying', 'lie', 'prone', 'down']):
            pose_map[cid] = 'lying'
        elif any(kw in name_lower for kw in ['sitting', 'sit', 'seated']):
            pose_map[cid] = 'sitting'
        elif any(kw in name_lower for kw in ['running', 'run', 'jogging']):
            pose_map[cid] = 'running'
        elif any(kw in name_lower for kw in ['standing', 'stand', 'upright']):
            pose_map[cid] = 'standing'
        elif any(kw in name_lower for kw in ['person', 'human', 'people']):
            pose_map[cid] = 'person'
        else:
            pose_map[cid] = name   # keep original

    return pose_map


def extract_statistics(records, class_counts, class_map,
                       visual_records, thermal_records):
    """
    Compute all statistics needed for MA-DTSR calibration.

    Returns
    -------
    stats : dict with all computed statistics
    """
    pose_map = infer_pose_mapping(class_map, class_counts)

    # ── Pose distribution ─────────────────────────────────────────────────────
    pose_counts = Counter()
    for rec in records:
        canonical = pose_map.get(rec['class_id'], 'unknown')
        pose_counts[canonical] += 1

    total_annotations = sum(pose_counts.values())

    pose_dist = {
        pose: count / total_annotations
        for pose, count in pose_counts.items()
    }

    # ── Modality split ────────────────────────────────────────────────────────
    n_visual  = len(visual_records)
    n_thermal = len(thermal_records)
    n_total   = n_visual + n_thermal

    if n_total > 0:
        visual_frac  = n_visual  / n_total
        thermal_frac = n_thermal / n_total
    else:
        visual_frac  = 0.6
        thermal_frac = 0.4

    # ── Bounding box size distribution ───────────────────────────────────────
    # Small bounding boxes = person is far/high altitude = harder to detect
    # Used to calibrate urgency: small bbox often means lying/obscured
    sizes = np.array([r['w'] * r['h'] for r in records])
    if len(sizes) > 0:
        median_size = float(np.median(sizes))
        mean_size   = float(np.mean(sizes))
    else:
        median_size = 0.001
        mean_size   = 0.001

    # ── Urgency calibration ───────────────────────────────────────────────────
    # Map canonical poses to urgency ranges for dim 4
    # Lying = highest urgency (incapacitated), running = lowest
    POSE_URGENCY = {
        'lying'   : (0.70, 1.00),
        'sitting' : (0.40, 0.70),
        'standing': (0.10, 0.40),
        'running' : (0.00, 0.20),
        'person'  : (0.30, 0.80),   # unknown pose → full range
        'unknown' : (0.30, 0.80),
    }

    # ── Mobility calibration ──────────────────────────────────────────────────
    # Proportion of annotations that are mobile (running or standing)
    mobile_count = (pose_counts.get('running', 0) +
                    pose_counts.get('standing', 0) * 0.3)  # standing=30% mobile
    mobile_frac  = mobile_count / max(total_annotations, 1)

    stats = {
        'total_annotations'  : total_annotations,
        'pose_counts'        : dict(pose_counts),
        'pose_distribution'  : pose_dist,
        'pose_map'           : {str(k): v for k, v in pose_map.items()},
        'pose_urgency_ranges': POSE_URGENCY,
        'n_visual_files'     : n_visual,
        'n_thermal_files'    : n_thermal,
        'visual_fraction'    : visual_frac,
        'thermal_fraction'   : thermal_frac,
        'mobile_fraction'    : float(mobile_frac),
        'median_bbox_area'   : median_size,
        'mean_bbox_area'     : mean_size,
    }

    return stats, pose_map


# ============================================================================
# SECTION 4: Calibrated Parameters Generator
# ============================================================================

def generate_calibrated_params(stats):
    """
    From the WiSARD statistics, generate the exact probability
    values to plug into MA-DTSR Step 2.

    Returns
    -------
    params : dict of calibrated parameters
    """
    pose_dist = stats['pose_distribution']
    poses = ['lying', 'sitting', 'standing', 'running', 'person']

    # Normalise to the four canonical poses used in Step 2
    # If only 'person' exists (single-class dataset), use SAR literature defaults
    has_pose_labels = any(
        p in pose_dist for p in ['lying', 'sitting', 'standing', 'running'])

    if has_pose_labels:
        lying_p    = pose_dist.get('lying',    0.0)
        sitting_p  = pose_dist.get('sitting',  0.0)
        standing_p = pose_dist.get('standing', 0.0)
        running_p  = pose_dist.get('running',  0.0)
        # Redistribute any 'person' mass proportionally
        person_p   = pose_dist.get('person',   0.0)
        total_p    = lying_p + sitting_p + standing_p + running_p
        if total_p < 1e-6:
            total_p = 1.0
        # Normalise the four poses + share person_p proportionally
        lying_p    = (lying_p    + person_p * lying_p    / total_p)
        sitting_p  = (sitting_p  + person_p * sitting_p  / total_p)
        standing_p = (standing_p + person_p * standing_p / total_p)
        running_p  = (running_p  + person_p * running_p  / total_p)
        # Final normalisation
        total = lying_p + sitting_p + standing_p + running_p
        if total < 1e-6:
            total = 1.0
        pose_probs = [
            lying_p    / total,
            sitting_p  / total,
            standing_p / total,
            running_p  / total,
        ]
    else:
        # Single-class or unknown — use SAR operational defaults
        # Based on published SAR casualty statistics
        print("\n  NOTE: No body-position labels found.")
        print("  Using SAR operational defaults for pose distribution.")
        pose_probs = [0.42, 0.28, 0.25, 0.05]

    # Visual/thermal split for dim 6 (platform type)
    visual_frac  = stats['visual_fraction']
    thermal_frac = stats['thermal_fraction']

    # Mobile fraction for dim 5 (mobility type)
    mobile_frac = stats['mobile_fraction']

    params = {
        'pose_labels' : ['lying', 'sitting', 'standing', 'running'],
        'pose_probs'  : [round(p, 4) for p in pose_probs],
        'visual_fraction' : round(visual_frac,  4),
        'thermal_fraction': round(thermal_frac, 4),
        'mobile_fraction' : round(mobile_frac,  4),
        'urgency_ranges'  : {
            'lying'   : [0.70, 1.00],
            'sitting' : [0.40, 0.70],
            'standing': [0.10, 0.40],
            'running' : [0.00, 0.20],
        },
    }

    return params


# ============================================================================
# SECTION 5: Visualisation
# ============================================================================

def plot_wisard_stats(stats, params, save_path='wisard_analysis.png'):
    """
    Generate a four-panel analysis figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('WiSARD Dataset Analysis — MA-DTSR Calibration',
                 fontsize=12, fontweight='bold')

    colours = ['#e63946', '#457b9d', '#2a9d8f', '#f4a261', '#adb5bd']

    # ── Panel 1: Body pose distribution ──────────────────────────────────────
    ax = axes[0, 0]
    pose_counts = stats['pose_counts']
    if pose_counts:
        labels = list(pose_counts.keys())
        values = list(pose_counts.values())
        total  = sum(values)
        bars = ax.bar(range(len(labels)), values,
                      color=colours[:len(labels)],
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Annotation count', fontsize=9)
        ax.set_title('Body Position Distribution\n(WiSARD annotations)', fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + total*0.005,
                    f'{val/total*100:.1f}%',
                    ha='center', va='bottom', fontsize=7)

    # ── Panel 2: Calibrated pose probs for Step 2 ─────────────────────────────
    ax = axes[0, 1]
    pose_labels = params['pose_labels']
    pose_probs  = params['pose_probs']
    bars = ax.bar(pose_labels, pose_probs,
                  color=colours[:4], edgecolor='white',
                  linewidth=0.5, alpha=0.85)
    ax.set_ylabel('Sampling probability', fontsize=9)
    ax.set_title('Calibrated Pose Probabilities\n(used in Step 2)', fontsize=9)
    ax.set_ylim(0, max(pose_probs) * 1.2)
    for bar, prob in zip(bars, pose_probs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=8)

    # ── Panel 3: Modality split ───────────────────────────────────────────────
    ax = axes[1, 0]
    mod_labels = ['Visual (RGB)', 'Thermal (LWIR)']
    mod_vals   = [stats['n_visual_files'], stats['n_thermal_files']]
    if sum(mod_vals) > 0:
        ax.pie(mod_vals, labels=mod_labels,
               colors=['#1d3557', '#e63946'],
               autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 8})
        ax.set_title('Modality Split\n(calibrates dim 6: platform type)',
                     fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No modality data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Modality Split', fontsize=9)

    # ── Panel 4: Urgency ranges from pose mapping ─────────────────────────────
    ax = axes[1, 1]
    urgency = params['urgency_ranges']
    y_pos   = range(len(urgency))
    for i, (pose, (low, high)) in enumerate(urgency.items()):
        ax.barh(i, high - low, left=low,
                color=colours[i], alpha=0.8, height=0.5,
                edgecolor='white', linewidth=0.5)
        ax.text(high + 0.01, i, f'{low:.2f}–{high:.2f}',
                va='center', fontsize=8)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(list(urgency.keys()), fontsize=8)
    ax.set_xlabel('Urgency value (dim 4)', fontsize=9)
    ax.set_title('Urgency Ranges by Body Position\n(calibrates dim 4)',
                 fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.5, color='black', linestyle='--',
               linewidth=0.8, alpha=0.4, label='Midpoint')

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved: {save_path}")


# ============================================================================
# SECTION 6: Output Generator
# ============================================================================

def print_calibrated_params(params, stats):
    """
    Print the exact code block to copy into Step 2.
    """
    print()
    print("=" * 70)
    print("  COPY THIS BLOCK INTO generate_resource_descriptor() IN STEP 2")
    print("=" * 70)
    print()
    print("# ── WiSARD-calibrated parameters ─────────────────────────────")
    print(f"# Source: WiSARD dataset analysis ({stats['total_annotations']:,} annotations)")
    print(f"# Visual files: {stats['n_visual_files']:,}  |  "
          f"Thermal files: {stats['n_thermal_files']:,}")
    print()
    print("WISARD_POSE_PROBS = {")
    for label, prob in zip(params['pose_labels'], params['pose_probs']):
        print(f"    '{label}' : {prob},   "
              f"# {prob*100:.1f}% of WiSARD annotations")
    print("}")
    print()
    print("WISARD_URGENCY_RANGES = {")
    for pose, (lo, hi) in params['urgency_ranges'].items():
        print(f"    '{pose}' : ({lo}, {hi}),")
    print("}")
    print()
    print(f"WISARD_VISUAL_FRACTION  = {params['visual_fraction']}  "
          f"# {params['visual_fraction']*100:.1f}% RGB")
    print(f"WISARD_THERMAL_FRACTION = {params['thermal_fraction']}  "
          f"# {params['thermal_fraction']*100:.1f}% thermal")
    print(f"WISARD_MOBILE_FRACTION  = {params['mobile_fraction']}  "
          f"# {params['mobile_fraction']*100:.1f}% of entities mobile")
    print()
    print("# Replace the dim 4 (urgency) generation in Step 2 with:")
    print("# pose = rng.choice(list(WISARD_POSE_PROBS.keys()),")
    print("#                   p=list(WISARD_POSE_PROBS.values()))")
    print("# lo, hi = WISARD_URGENCY_RANGES[pose]")
    print("# dim4   = rng.uniform(lo, hi)")
    print()
    print("# Replace the dim 5 (mobility) generation with:")
    print("# dim5 = rng.choice([0.0, 1.0],")
    print(f"#                   p=[{1-params['mobile_fraction']:.3f}, "
          f"{params['mobile_fraction']:.3f}])")
    print()
    print("# Replace the dim 6 (platform) sensing branch with:")
    print("# if cat_idx == 2:  # sensing → UAV")
    print(f"#     thermal_flag = rng.choice([0, 1],")
    print(f"#                   p=[{params['visual_fraction']:.3f}, "
          f"{params['thermal_fraction']:.3f}])")
    print("#     plat_idx = 2  # UAV platform regardless of modality")
    print("=" * 70)


def save_stats(stats, params, path='wisard_stats.json'):
    """Save statistics and calibrated params to JSON."""
    output = {
        'statistics'         : stats,
        'calibrated_params'  : params,
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nStats saved: {path}")


# ============================================================================
# SECTION 7: Main
# ============================================================================

def main_wisard_analysis(wisard_root=None):
    """
    Run the full WiSARD analysis pipeline.

    Parameters
    ----------
    wisard_root : str or None — path to extracted WiSARD folder
                                defaults to WISARD_ROOT constant above

    Returns
    -------
    stats  : dict — raw statistics
    params : dict — calibrated parameters for Step 2
    """
    if wisard_root is None:
        wisard_root = WISARD_ROOT

    print("=" * 65)
    print("  WiSARD Dataset Analysis — MA-DTSR Calibration")
    print("=" * 65)
    print(f"\nDataset root: {wisard_root}")

    if not os.path.exists(wisard_root):
        raise FileNotFoundError(
            f"WiSARD root not found: {wisard_root}\n"
            f"Set WISARD_ROOT at the top of this script to your "
            f"WiSARD folder path."
        )

    # ── Step 1: Find files ────────────────────────────────────────────────────
    print("\nStep 1: Discovering annotation files...")
    file_groups = find_annotation_files(wisard_root)

    # ── Step 2: Find class map ────────────────────────────────────────────────
    print("\nStep 2: Loading class definitions...")
    class_map = find_class_file(wisard_root)

    # ── Step 3: Parse annotations ─────────────────────────────────────────────
    print("\nStep 3: Parsing annotations...")
    print("  Parsing visual annotations...")
    vis_records, vis_counts, vis_sizes = parse_annotations(
        file_groups['visual'], class_map)

    print("  Parsing thermal annotations...")
    thm_records, thm_counts, thm_sizes = parse_annotations(
        file_groups['thermal'], class_map)

    # If no modality-specific files found, parse all
    if not vis_records and not thm_records:
        print("  No modality split found — parsing all annotations...")
        all_records, all_counts, all_sizes = parse_annotations(
            file_groups['all'], class_map)
        vis_records = all_records
        vis_counts  = all_counts
    else:
        all_records = vis_records + thm_records
        all_counts  = vis_counts + thm_counts

    if not all_records:
        print("\nWARNING: No annotations parsed.")
        print("Check that WISARD_ROOT points to the correct folder.")
        print("Expected .txt files alongside image files.")
        return None, None

    # ── Step 4: Extract statistics ────────────────────────────────────────────
    print("\nStep 4: Computing statistics...")
    stats, pose_map = extract_statistics(
        all_records, all_counts, class_map,
        vis_records, thm_records)

    # ── Step 5: Generate calibrated parameters ────────────────────────────────
    print("\nStep 5: Generating calibrated parameters...")
    params = generate_calibrated_params(stats)

    # ── Step 6: Print results ──────────────────────────────────────────────────
    print("\n--- WiSARD Statistics ---")
    print(f"  Total annotations  : {stats['total_annotations']:,}")
    print(f"  Visual files       : {stats['n_visual_files']:,}  "
          f"({stats['visual_fraction']*100:.1f}%)")
    print(f"  Thermal files      : {stats['n_thermal_files']:,}  "
          f"({stats['thermal_fraction']*100:.1f}%)")
    print(f"  Mobile fraction    : {stats['mobile_fraction']*100:.1f}%")
    print()
    print("  Body position distribution:")
    total = stats['total_annotations']
    for pose, count in sorted(stats['pose_counts'].items(),
                              key=lambda x: -x[1]):
        print(f"    {pose:15s}: {count:6,}  ({count/total*100:.1f}%)")

    # ── Step 7: Visualise ──────────────────────────────────────────────────────
    print("\nStep 6: Generating analysis figure...")
    plot_wisard_stats(stats, params)

    # ── Step 8: Print code block ───────────────────────────────────────────────
    print_calibrated_params(params, stats)

    # ── Step 9: Save ──────────────────────────────────────────────────────────
    save_stats(stats, params)

    return stats, params


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    stats, params = main_wisard_analysis()
