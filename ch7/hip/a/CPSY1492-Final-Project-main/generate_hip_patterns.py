import numpy as np
import pandas as pd
import os

# =========================
# CONFIG
# =========================
N_UNITS = 40
N_ACTIVE = 10   # choose values that divide nicely into overlap %
N_PATTERNS = 8

OVERLAPS = [0, 20, 40, 50, 60, 80]  # percent overlap
OUTPUT_DIR = "hip_patterns"

VERIFY_OVERLAP = True  # set to False if you don’t want debug prints

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# FUNCTIONS
# =========================

def generate_base_pattern(n_units, n_active):
    pattern = np.zeros(n_units, dtype=int)
    active = np.random.choice(n_units, n_active, replace=False)
    pattern[active] = 1
    return pattern, active


def generate_pattern_from_base(base_pattern, base_active, overlap_pct):
    n_active = len(base_active)
    n_overlap = int((overlap_pct / 100) * n_active)

    # Shared active units
    shared = np.random.choice(base_active, n_overlap, replace=False)

    # New active units
    n_new = n_active - n_overlap
    available = list(set(range(len(base_pattern))) - set(base_active))
    new_units = np.random.choice(available, n_new, replace=False)

    pattern = np.zeros_like(base_pattern)
    pattern[shared] = 1
    pattern[new_units] = 1

    return pattern


def pattern_to_string(pattern):
    return " ".join(map(str, pattern))


def compute_overlap(p1, p2):
    shared = np.sum((p1 == 1) & (p2 == 1))
    total = np.sum(p1 == 1)
    return shared / total if total > 0 else 0


# =========================
# MAIN GENERATION LOOP
# =========================

for overlap in OVERLAPS:
    print(f"\n=== Generating {overlap}% overlap dataset ===")

    rows = []

    base_pattern, base_active = generate_base_pattern(N_UNITS, N_ACTIVE)
    patterns = []

    for i in range(N_PATTERNS):
        if i == 0:
            p = base_pattern
        else:
            p = generate_pattern_from_base(base_pattern, base_active, overlap)

        patterns.append(p)

    # Optional: verify overlaps
    if VERIFY_OVERLAP:
        print("Pairwise overlaps:")
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                ov = compute_overlap(patterns[i], patterns[j])
                print(f"P{i+1} vs P{j+1}: {ov:.2f}")

    # Build TSV rows
    for i, p in enumerate(patterns):
        row = {
            "Name": f"P{i+1}",
            "Input": pattern_to_string(p),
            "ECout": pattern_to_string(p)
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    filename = os.path.join(OUTPUT_DIR, f"patterns_{overlap}.tsv")
    df.to_csv(filename, sep="\t", index=False)

    print(f"Saved: {filename}")

print("\nAll datasets generated successfully.")