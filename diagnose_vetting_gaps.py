"""
Diagnose scenarios that pass SCI 2025 vetting (Vetting|SCI 2025 == 'ok')
but fail the hard quality vetting criteria.

For each failing scenario, reports which criterion/year caused the failure
and the actual vs expected values.
"""

import pandas as pd
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
meta = pd.read_csv("inputs/sci_meta_data.csv")
quality_data = pd.read_csv("inputs/quality_data_sci.csv")

# ── 1. Scenarios that are marked as "ok" in metadata ──────────────────────
ok_meta = meta[meta["Vetting|SCI 2025"].str.strip().str.lower() == "ok"]
ok_set = set(zip(ok_meta["Model"], ok_meta["Scenario"]))
print(f"Scenarios with Vetting|SCI 2025 == 'ok': {len(ok_set)}")

# ── 2. Vetting criteria (mirroring VETTING_CRITERIA_SCI from constants.py) ─
VETTING_CRITERIA_SCI = {
    "Historical EIP emissions": {
        "Variables": ["Emissions|CO2|Energy and Industrial Processes"],
        "Value": [33460.1, 35627.3, 36315.3, 39383.5],
        "Range": [0.25, 0.25, 0.27, 0.25],
        "Year": [2010, 2015, 2020, 2025],
    },
    "Final Energy": {
        "Variables": ["Final Energy"],
        "Value": [365.074, 389.56, 395.78, 443.09],
        "Range": [0.25, 0.25, 0.40, 0.25],
        "Year": [2010, 2015, 2020, 2025],
    },
    "Primary Energy - Coal": {
        "Variables": ["Primary Energy|Coal"],
        "Value": [153.51, 161.99, 160.00, 200.43],
        "Range": [0.25, 0.25, 0.40, 0.25],
        "Year": [2010, 2015, 2020, 2025],
    },
    "Primary Energy - Oil": {
        "Variables": ["Primary Energy|Oil"],
        "Value": [167.72, 180.60, 173.40, 197.75],
        "Range": [0.25, 0.25, 0.40, 0.25],
        "Year": [2010, 2015, 2020, 2025],
    },
    "Primary Energy - Gas": {
        "Variables": ["Primary Energy|Gas"],
        "Value": [126.22, 138.32, 155.11, 164.66],
        "Range": [0.25, 0.25, 0.40, 0.25],
        "Year": [2010, 2015, 2020, 2025],
    },
    "Primary Energy - Nuclear": {
        "Variables": ["Primary Energy|Nuclear"],
        "Value": [9.92, 9.25, 9.63, 8.97],
        "Range": [0.25, 0.25, 0.40, 0.25],
        "Year": [2010, 2015, 2020, 2025],
    },
}

# ── 3. Run hard vetting logic on quality data ─────────────────────────────
# For each criterion/year, determine pass/fail/not-reporting per scenario
quality_data_ok = quality_data[
    quality_data.apply(
        lambda r: (r["Model"], r["Scenario"]) in ok_set, axis=1
    )
]

all_scenarios = quality_data_ok[["Model", "Scenario"]].drop_duplicates()
all_sm = set(zip(all_scenarios["Model"], all_scenarios["Scenario"]))
print(f"'ok' scenarios with quality data rows: {len(all_sm)}")

# Track per-criterion results
failure_details = []  # list of dicts for each scenario ✕ criterion ✕ year

for criteria_name, cfg in VETTING_CRITERIA_SCI.items():
    variables = cfg["Variables"]
    values = cfg["Value"]
    ranges = cfg["Range"]
    years = cfg["Year"]

    crit_data = quality_data_ok[quality_data_ok["Variable"].isin(variables)]
    grouped = crit_data.groupby(["Model", "Scenario"])

    for year, target_val, vetting_range in zip(years, values, ranges):
        ycol = str(year)
        yr_sums = grouped[ycol].sum(min_count=1).reset_index()
        yr_sums = yr_sums.set_index(["Model", "Scenario"])

        lower = target_val * (1 - vetting_range)
        upper = target_val * (1 + vetting_range)

        for model, scenario in all_sm:
            if (model, scenario) in yr_sums.index:
                actual = yr_sums.loc[(model, scenario), ycol]
                if pd.isna(actual):
                    status = "no_data_for_year"
                    pct_dist = np.nan
                elif actual < lower or actual > upper:
                    status = "fail"
                    pct_dist = abs(actual - target_val) / abs(target_val)
                else:
                    status = "pass"
                    pct_dist = abs(actual - target_val) / abs(target_val)
            else:
                actual = np.nan
                status = "variable_not_reported"
                pct_dist = np.nan

            failure_details.append({
                "Model": model,
                "Scenario": scenario,
                "Criterion": criteria_name,
                "Variable": ", ".join(variables),
                "Year": year,
                "Target": target_val,
                "Range_pct": vetting_range,
                "Lower_bound": round(lower, 2),
                "Upper_bound": round(upper, 2),
                "Actual": actual if not isinstance(actual, pd.Series) else actual.iloc[0],
                "Pct_distance": pct_dist,
                "Status": status,
            })

details_df = pd.DataFrame(failure_details)

# ── 4. Determine which scenarios pass ALL criteria/years ──────────────────
# A scenario passes hard vetting if it has status == "pass" for EVERY
# criterion/year combination.
pivot = details_df.pivot_table(
    index=["Model", "Scenario"],
    columns=["Criterion", "Year"],
    values="Status",
    aggfunc="first",
)

passes_all = (pivot == "pass").all(axis=1)
passing_scenarios = set(passes_all[passes_all].index)
failing_scenarios = set(passes_all[~passes_all].index)

print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")
print(f"Scenarios marked 'ok' in Vetting|SCI 2025:        {len(ok_set)}")
print(f"  of which found in quality data:                  {len(all_sm)}")
print(f"  PASS hard vetting:                               {len(passing_scenarios)}")
print(f"  FAIL hard vetting:                               {len(failing_scenarios)}")

# ── 5. Scenarios in meta 'ok' but missing from quality data entirely ──────
ok_not_in_quality = ok_set - all_sm
if ok_not_in_quality:
    print(f"  Missing from quality data entirely:              {len(ok_not_in_quality)}")
    missing_df = pd.DataFrame(
        sorted(ok_not_in_quality), columns=["Model", "Scenario"]
    )
    print("\n  Missing scenarios:")
    print(missing_df.to_string(index=False))

# ── 6. For failing scenarios, show WHY they fail ──────────────────────────
if failing_scenarios:
    print(f"\n{'='*70}")
    print("FAILURE DETAILS")
    print(f"{'='*70}")

    fail_details = details_df[
        details_df.apply(
            lambda r: (r["Model"], r["Scenario"]) in failing_scenarios, axis=1
        )
    ]
    # Only show non-pass rows for failing scenarios
    fail_reasons = fail_details[fail_details["Status"] != "pass"].sort_values(
        ["Model", "Scenario", "Criterion", "Year"]
    )

    for (model, scenario), group in fail_reasons.groupby(["Model", "Scenario"]):
        print(f"\n  {model} | {scenario}")
        for _, row in group.iterrows():
            if row["Status"] == "fail":
                print(
                    f"    FAIL  {row['Criterion']} ({row['Year']}): "
                    f"actual={row['Actual']:.2f}, "
                    f"target={row['Target']:.2f}, "
                    f"allowed=[{row['Lower_bound']:.2f}, {row['Upper_bound']:.2f}], "
                    f"pct_distance={row['Pct_distance']:.1%}"
                )
            elif row["Status"] == "no_data_for_year":
                print(
                    f"    NO DATA  {row['Criterion']} ({row['Year']}): "
                    f"variable exists but year {row['Year']} is NaN"
                )
            elif row["Status"] == "variable_not_reported":
                print(
                    f"    NOT REPORTED  {row['Criterion']} ({row['Year']}): "
                    f"variable '{row['Variable']}' not in quality data"
                )

# ── 7. Save full details to CSV for manual inspection ────────────────────
fail_output = details_df[
    details_df.apply(
        lambda r: (r["Model"], r["Scenario"]) in failing_scenarios, axis=1
    )
].sort_values(["Model", "Scenario", "Criterion", "Year"])

fail_output.to_csv("outputs/vetting_gap_diagnosis.csv", index=False)
print(f"\nFull details saved to outputs/vetting_gap_diagnosis.csv")
