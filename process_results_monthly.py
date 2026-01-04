"""
Monthly-focused processor for LLM test results.
Loads a results CSV, summarizes per-month accuracy, and produces a compact monthly plot.
"""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


# =============================================================================
# Data loading
# =============================================================================

def load_results_from_csv(csv_file: str) -> List[Dict]:
    """Load results data from CSV file and normalize boolean fields."""
    results: List[Dict] = []
    with open(csv_file, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            knows_str = row.get("llm_knows_death", "")
            if knows_str == "True":
                row["llm_knows_death"] = True
            elif knows_str == "False":
                row["llm_knows_death"] = False
            else:
                row["llm_knows_death"] = None
            results.append(row)
    return results


# =============================================================================
# Statistics
# =============================================================================

def calculate_accuracy_by_month(results: List[Dict]) -> Dict[str, Dict]:
    """Aggregate accuracy statistics grouped by YYYY-MM."""
    stats_by_month: Dict[str, Dict] = defaultdict(
        lambda: {"correct": 0, "incorrect": 0, "unknown": 0, "total": 0}
    )

    for result in results:
        month = result["death_date"][:7]  # YYYY-MM
        knows = result["llm_knows_death"]

        stats_by_month[month]["total"] += 1
        if knows is True:
            stats_by_month[month]["correct"] += 1
        elif knows is False:
            stats_by_month[month]["incorrect"] += 1
        else:
            stats_by_month[month]["unknown"] += 1

    for month, stats in stats_by_month.items():
        known_total = stats["correct"] + stats["incorrect"]
        if known_total > 0:
            stats["accuracy"] = stats["correct"] / known_total * 100
        else:
            stats["accuracy"] = None

    return dict(stats_by_month)


# =============================================================================
# Reporting
# =============================================================================

def print_summary(results: List[Dict], stats_by_month: Dict[str, Dict], min_samples: int) -> None:
    """Print overall and per-month summaries."""
    total_correct = sum(1 for r in results if r["llm_knows_death"] is True)
    total_incorrect = sum(1 for r in results if r["llm_knows_death"] is False)
    total_unknown = sum(1 for r in results if r["llm_knows_death"] is None)

    total = len(results)
    print("\n" + "=" * 60)
    print("Overall")
    print("=" * 60)
    print(f"Total tested: {total}")
    print(f"  Knows death: {total_correct} ({total_correct/total*100:.1f}%)")
    print(f"  Doesn't know: {total_incorrect} ({total_incorrect/total*100:.1f}%)")
    if total_unknown:
        print(f"  Errors: {total_unknown} ({total_unknown/total*100:.1f}%)")

    print("\nPer month (known >= {min_samples}):")
    header = f"{'Month':<10}{'Accuracy':>10}{'Known':>10}{'Total':>10}"
    print(header)
    print("-" * len(header))

    for month in sorted(stats_by_month.keys()):
        stats = stats_by_month[month]
        known_total = stats["correct"] + stats["incorrect"]
        if stats["accuracy"] is None or known_total < min_samples:
            continue
        acc_str = f"{stats['accuracy']:.1f}%" if stats["accuracy"] is not None else "N/A"
        print(f"{month:<10}{acc_str:>10}{known_total:>10}{stats['total']:>10}")


# =============================================================================
# Plotting
# =============================================================================

def plot_monthly_accuracy(
    stats_by_month: Dict[str, Dict],
    model: str,
    output_file: str,
    min_samples: int,
) -> None:
    """Plot monthly accuracy without per-bar labels so many months fit."""
    months: List[str] = []
    accuracies: List[float] = []

    for month in sorted(stats_by_month.keys()):
        stats = stats_by_month[month]
        known_total = stats["correct"] + stats["incorrect"]
        if stats["accuracy"] is None or known_total < min_samples:
            continue
        months.append(month)
        accuracies.append(stats["accuracy"])

    if not months:
        print("No monthly data meets the sample threshold; plot skipped.")
        return

    # Expand width slightly as months grow; keeps labels readable.
    fig_width = max(12, len(months) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.bar(range(len(months)), accuracies, color="#4d7fcd", edgecolor="#1f3f75", width=0.8)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=60, ha="right")

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Death month (YYYY-MM)", fontsize=11)
    ax.set_title(f"{model} knowledge of deaths by month", fontsize=13)

    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.35)

    ax.text(
        0.01,
        0.97,
        f"Months with at least {min_samples} known samples",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        alpha=0.75,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    print(f"Monthly plot saved to {output_file}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LLM results by month and produce a compact plot."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="test_results.csv",
        help="Input CSV file with test results (default: test_results.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LLM",
        help="Model name used for plot titles (default: LLM)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="cutoff_plot_monthly.png",
        help="Output PNG for the monthly plot (default: cutoff_plot_monthly.png)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum known samples required for a month to be shown (default: 1)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the monthly plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading results from {args.input}")
    results = load_results_from_csv(args.input)
    print(f"Loaded {len(results)} rows")

    if not results:
        print("No results to analyze; exiting.")
        return

    stats_by_month = calculate_accuracy_by_month(results)

    print_summary(results, stats_by_month, args.min_samples)

    if not args.no_plot:
        plot_monthly_accuracy(stats_by_month, args.model, args.plot, args.min_samples)


if __name__ == "__main__":
    main()
