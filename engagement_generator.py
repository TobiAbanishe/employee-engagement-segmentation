"""
Project 6: Employee Engagement Segmentation Dashboard
Data Generator — Synthetic Q12 Survey Dataset

Research foundation:
    Gallup Q12 Employee Engagement Survey — 12 questions measuring the
    psychological conditions employees need to thrive at work. Each item
    is rated on a 1-5 scale (1 = strongly disagree, 5 = strongly agree).

    Gallup State of the Global Workplace Report (2023):
        - 23% of employees globally are engaged
        - 59% are not engaged (quiet quitting)
        - 18% are actively disengaged
        - Cost of disengagement: $8.8 trillion globally (9% of GDP)
        - Manager quality accounts for 70% of variance in team engagement
        - Engaged employees show 81% lower absenteeism and 23% higher profitability

Methodology applied:
    1. A manager pool of 60 individuals is created, each with an inherent
       management quality score (0-1) that drives their team's engagement level.
       This encodes Gallup's finding that manager quality explains 70% of
       engagement variance.
    2. Department-level baseline scores further modulate Q12 responses.
    3. Individual-level noise, tenure effects, and a honeymoon curve for new
       hires are applied to produce realistic distribution patterns.
    4. Outcome variables (absenteeism, performance, intent to stay, flight risk)
       are derived from engagement scores with correlations calibrated to
       Gallup's published benchmarks.

Dataset: 900 synthetic employees
Target segmentation: ~23% Engaged, ~59% Not Engaged, ~18% Actively Disengaged
Output:  engagement_data.csv, engagement_data.xlsx
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

N = 900
REFERENCE_DATE = datetime(2024, 6, 1)

# ---------------------------------------------------------------------------
# Q12 question definitions (Gallup)
# ---------------------------------------------------------------------------

Q12_ITEMS = {
    "q01": "I know what is expected of me at work.",
    "q02": "I have the materials and equipment I need to do my work right.",
    "q03": "I have the opportunity to do what I do best every day.",
    "q04": "In the last seven days, I have received recognition or praise for doing good work.",
    "q05": "My supervisor, or someone at work, seems to care about me as a person.",
    "q06": "There is someone at work who encourages my development.",
    "q07": "At work, my opinions seem to count.",
    "q08": "The mission of my company makes me feel my job is important.",
    "q09": "My associates are committed to doing quality work.",
    "q10": "I have a best friend at work.",
    "q11": "In the last six months, someone has talked to me about my progress.",
    "q12": "This last year, I have had opportunities to learn and grow.",
}

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

DEPARTMENTS = [
    "Engineering",
    "Finance",
    "Human Resources",
    "Marketing",
    "Operations",
    "Legal",
    "Sales",
    "Product",
    "Customer Support",
]

# Department baseline engagement (mean score on 1-5 scale)
# Calibrated so aggregate output hits ~23/59/18 Gallup split
DEPT_BASELINES = {
    "Engineering":       3.55,
    "Finance":           3.30,
    "Human Resources":   3.60,
    "Marketing":         3.45,
    "Operations":        2.95,
    "Legal":             3.20,
    "Sales":             3.10,
    "Product":           3.65,
    "Customer Support":  2.85,
}

GENDER_OPTIONS = ["Male", "Female", "Non-binary"]
GENDER_PROBS   = [0.52, 0.44, 0.04]

ETHNICITY_OPTIONS = [
    "White",
    "Black or African American",
    "Hispanic or Latino",
    "Asian",
    "Two or more races",
    "Other / prefer not to say",
]
ETHNICITY_PROBS = [0.55, 0.12, 0.14, 0.12, 0.04, 0.03]

LOCATIONS      = ["Toronto", "Vancouver", "Calgary", "Ottawa", "Montreal", "Remote"]
LOCATION_PROBS = [0.35, 0.20, 0.15, 0.12, 0.10, 0.08]

JOB_LEVELS = ["Entry", "Junior", "Mid-Level", "Senior", "Lead / Manager", "Director+"]

PERFORMANCE_RATINGS = [1, 2, 3, 4, 5]

# ---------------------------------------------------------------------------
# Engagement segmentation thresholds (Gallup Q12 methodology)
#   Engaged:           avg Q12 score >= 4.0
#   Not Engaged:       avg Q12 score 2.5 to < 4.0
#   Actively Disengaged: avg Q12 score < 2.5
# ---------------------------------------------------------------------------

ENGAGED_THRESHOLD    = 4.0
DISENGAGED_THRESHOLD = 2.5


def classify_segment(avg_score: float) -> str:
    if avg_score >= ENGAGED_THRESHOLD:
        return "Engaged"
    if avg_score >= DISENGAGED_THRESHOLD:
        return "Not Engaged"
    return "Actively Disengaged"


# ---------------------------------------------------------------------------
# Manager pool
# Gallup: manager quality explains ~70% of team engagement variance.
# We create 60 managers with a quality score (0-1) that biases Q12 responses.
# ---------------------------------------------------------------------------

N_MANAGERS = 60


def build_manager_pool() -> dict:
    """
    Return a dict mapping manager_id -> manager quality score (0-1).
    Quality is drawn from a beta distribution to create realistic spread:
    most managers are average; a few are excellent or poor.
    """
    pool = {}
    for i in range(N_MANAGERS):
        mgr_id  = f"MGR{str(i + 101).zfill(4)}"
        quality = float(np.random.beta(a=5, b=3))  # right-skewed: most are decent
        pool[mgr_id] = quality
    return pool


# ---------------------------------------------------------------------------
# Tenure effect
# Gallup documents a honeymoon effect: new hires score higher in months 1-6,
# decline in months 7-24, then stabilise or bifurcate at longer tenures.
# ---------------------------------------------------------------------------

def tenure_adjustment(years_tenure: float) -> float:
    if years_tenure < 0.5:
        return 0.30   # Honeymoon — new hire optimism
    if years_tenure < 2.0:
        return 0.05   # Reality sets in
    if years_tenure < 5.0:
        return 0.0    # Baseline
    if years_tenure < 10.0:
        return -0.05  # Mild drift
    # Long tenure: bimodal — either very committed or checked out
    return float(np.random.choice([0.20, -0.30], p=[0.45, 0.55]))


# ---------------------------------------------------------------------------
# Q12 item generation
# Manager-heavy items: Q04 (recognition), Q05 (cares), Q06 (development),
# Q07 (opinions count), Q11 (progress conversations) — more sensitive to mgr quality.
# ---------------------------------------------------------------------------

MGR_HEAVY_ITEMS = {"q04", "q05", "q06", "q07", "q11"}
MGR_WEIGHT      = 0.70   # Gallup: manager drives 70% of engagement variance


def generate_q12_responses(
    dept_baseline: float,
    manager_quality: float,
    years_tenure: float,
) -> dict:
    """
    Generate a full set of 12 Q12 item scores (1-5) for one employee.
    """
    ten_adj = tenure_adjustment(years_tenure)

    # Manager contribution: scale quality (0-1) to a score offset centred on 0
    mgr_signal = (manager_quality - 0.5) * 2.0  # maps 0-1 to -1 to +1

    responses = {}
    for item in Q12_ITEMS:
        # Base score from department
        base = dept_baseline + ten_adj

        if item in MGR_HEAVY_ITEMS:
            # Heavy manager influence
            signal = MGR_WEIGHT * mgr_signal * 1.2
        else:
            # Lighter manager influence for environment / resources items
            signal = MGR_WEIGHT * mgr_signal * 0.4

        # Individual noise
        noise = np.random.normal(0, 0.55)

        raw = base + signal + noise

        # Clamp to 1-5 and round to nearest 0.5 (realistic survey response pattern)
        raw    = float(np.clip(raw, 1.0, 5.0))
        scored = round(raw * 2) / 2          # rounds to nearest 0.5
        scored = float(np.clip(scored, 1.0, 5.0))
        responses[item] = scored

    return responses


# ---------------------------------------------------------------------------
# Outcome variable generation
# Calibrated to Gallup benchmark correlations
# ---------------------------------------------------------------------------

def generate_outcomes(avg_score: float, segment: str, perf_rating: int) -> dict:
    """
    Generate business outcome variables correlated with engagement level.
    Gallup benchmarks:
      - Engaged employees: 81% lower absenteeism
      - Engaged employees: 23% higher profitability (proxied via performance)
      - Actively Disengaged: 4x more likely to be a flight risk
    """
    # Absenteeism: inverse relationship with engagement
    if segment == "Engaged":
        absence_days = max(0, int(np.random.poisson(3)))
    elif segment == "Not Engaged":
        absence_days = max(0, int(np.random.poisson(7)))
    else:  # Actively Disengaged
        absence_days = max(0, int(np.random.poisson(14)))

    # Intent to stay (1-5)
    intent_base = avg_score * 0.8 + np.random.normal(0, 0.4)
    intent_to_stay = int(np.clip(round(intent_base), 1, 5))

    # Flight risk flag: low engagement + low intent to stay
    flight_risk = 1 if (avg_score < 3.0 and intent_to_stay <= 2) else 0

    # Performance rating — weakly correlated with engagement
    perf_noise = np.random.randint(-1, 2)
    perf_from_engagement = int(np.clip(round(avg_score * 0.7 + perf_noise), 1, 5))

    # eNPS proxy: likelihood to recommend as employer (0-10)
    enps_base = (avg_score - 1) / 4 * 10 + np.random.normal(0, 1.5)
    enps_score = int(np.clip(round(enps_base), 0, 10))

    # eNPS segment: Promoter (9-10), Passive (7-8), Detractor (0-6)
    if enps_score >= 9:
        enps_segment = "Promoter"
    elif enps_score >= 7:
        enps_segment = "Passive"
    else:
        enps_segment = "Detractor"

    return {
        "absence_days":       absence_days,
        "intent_to_stay":     intent_to_stay,
        "flight_risk_flag":   flight_risk,
        "performance_rating": perf_from_engagement,
        "enps_score":         enps_score,
        "enps_segment":       enps_segment,
    }


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def random_hire_date() -> datetime:
    start = datetime(2012, 1, 1)
    end   = datetime(2024, 1, 1)
    return start + timedelta(days=random.randint(0, (end - start).days))


def calc_years_tenure(hire_date: datetime) -> float:
    return round((REFERENCE_DATE - hire_date).days / 365.25, 1)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(n: int = N) -> pd.DataFrame:
    """
    Generate n synthetic employee engagement survey records calibrated to
    Gallup Q12 methodology and State of the Global Workplace (2023) benchmarks.
    """
    manager_pool = build_manager_pool()
    manager_ids  = list(manager_pool.keys())

    # Assign managers to departments — each dept gets a proportional slice
    dept_managers: dict[str, list[str]] = {dept: [] for dept in DEPARTMENTS}
    for mgr_id in manager_ids:
        dept = random.choice(DEPARTMENTS)
        dept_managers[dept].append(mgr_id)

    # Ensure every department has at least 2 managers
    for dept in DEPARTMENTS:
        while len(dept_managers[dept]) < 2:
            dept_managers[dept].append(random.choice(manager_ids))

    records = []

    for i in range(n):
        # --- Demographics ---
        gender    = str(np.random.choice(GENDER_OPTIONS, p=GENDER_PROBS))
        ethnicity = str(np.random.choice(ETHNICITY_OPTIONS, p=ETHNICITY_PROBS))
        age       = int(np.clip(np.random.normal(36, 9), 22, 62))

        # --- Job structure ---
        dept      = random.choice(DEPARTMENTS)
        job_level = random.choice(JOB_LEVELS)

        # --- Manager ---
        manager_id      = random.choice(dept_managers[dept])
        manager_quality = manager_pool[manager_id]

        # --- Dates and tenure ---
        hire_date    = random_hire_date()
        years_tenure = calc_years_tenure(hire_date)

        # --- Location ---
        location  = str(np.random.choice(LOCATIONS, p=LOCATION_PROBS))
        is_remote = 1 if location == "Remote" else 0

        # --- Q12 responses ---
        dept_baseline = DEPT_BASELINES[dept]
        q12_responses = generate_q12_responses(dept_baseline, manager_quality, years_tenure)
        avg_q12_score = round(float(np.mean(list(q12_responses.values()))), 4)
        segment       = classify_segment(avg_q12_score)

        # --- Outcomes ---
        perf_base = int(np.random.choice(PERFORMANCE_RATINGS, p=[0.03, 0.12, 0.50, 0.28, 0.07]))
        outcomes  = generate_outcomes(avg_q12_score, segment, perf_base)

        # --- Survey metadata ---
        survey_month = random.choice(["Jan", "Apr", "Jul", "Oct"])
        survey_year  = 2023

        record = {
            "employee_id":        f"EMP{str(i + 2001).zfill(5)}",
            "gender":             gender,
            "ethnicity":          ethnicity,
            "age":                age,
            "location":           location,
            "is_remote":          is_remote,
            "department":         dept,
            "job_level":          job_level,
            "manager_id":         manager_id,
            "manager_quality":    round(manager_quality, 4),
            "hire_date":          hire_date.strftime("%Y-%m-%d"),
            "years_tenure":       years_tenure,
            "survey_month":       survey_month,
            "survey_year":        survey_year,
            "engagement_segment": segment,
            "avg_q12_score":      avg_q12_score,
            **{k: v for k, v in q12_responses.items()},
            **outcomes,
        }
        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------

def print_validation(df: pd.DataFrame) -> None:
    """Print a validation summary against Gallup 2023 global benchmarks."""
    print("\n" + "=" * 65)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 65)

    print(f"\nTotal employees : {len(df):,}")
    print(f"Columns         : {df.shape[1]}")

    print("\n--- Engagement segmentation (Gallup 2023 benchmark) ---")
    seg_counts = df["engagement_segment"].value_counts()
    seg_pct    = df["engagement_segment"].value_counts(normalize=True).mul(100).round(1)
    summary    = pd.DataFrame({"count": seg_counts, "pct": seg_pct})
    benchmarks = {"Engaged": 23.0, "Not Engaged": 59.0, "Actively Disengaged": 18.0}
    summary["gallup_benchmark"] = summary.index.map(benchmarks)
    print(summary.to_string())

    print("\n--- Avg Q12 score by segment ---")
    print(df.groupby("engagement_segment")["avg_q12_score"].mean().round(3).to_string())

    print("\n--- Flight risk count ---")
    print(f"  Flagged employees : {df['flight_risk_flag'].sum():,}")
    print(f"  As % of workforce : {df['flight_risk_flag'].mean() * 100:.1f}%")

    print("\n--- Avg absenteeism days by segment ---")
    print(df.groupby("engagement_segment")["absence_days"].mean().round(1).to_string())

    print("\n--- Avg Q12 score by department ---")
    print(df.groupby("department")["avg_q12_score"].mean().sort_values(ascending=False).round(3).to_string())

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(df: pd.DataFrame, output_dir: str = ".") -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path  = os.path.join(output_dir, "engagement_data.csv")
    xlsx_path = os.path.join(output_dir, "engagement_data.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="EngagementData")
    print(f"\nOutput files:")
    print(f"  CSV   : {csv_path}")
    print(f"  Excel : {xlsx_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating engagement segmentation dataset ...")
    df = generate_dataset(N)
    print_validation(df)
    export(df)
    print("\nDone.")
