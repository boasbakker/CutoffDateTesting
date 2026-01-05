import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# =============================================================================
# Data loading
# =============================================================================

def load_results_from_csv(csv_file: str) -> List[Dict]:
    """Load results data from CSV file and normalize boolean fields."""
    results: List[Dict] = []
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return []
        
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

def calculate_moving_average(data: List[float], window_size: int = 3) -> List[float]:
    """Calculate simple moving average to smooth the trend line."""
    moving_averages = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window = data[start_idx : i + 1]
        moving_averages.append(sum(window) / len(window))
    return moving_averages

# =============================================================================
# String Formatting & Parsing
# =============================================================================

def format_number(num: int) -> str:
    """Formats large numbers into '500 thousand', '1.5 million', etc."""
    if num >= 1_000_000:
        val = num / 1_000_000
        return f"{val:.1f} million".replace(".0 million", " million")
    elif num >= 1_000:
        val = num / 1_000
        return f"{val:.0f} thousand"
    return str(num)

def parse_title_info(filename: str) -> Tuple[str, str]:
    """
    Parses the filename to extract Model Name and View count.
    Input: gpt-5.2_2020-01-01_to_2025-12-31_topmonth-100_minviews-500000.csv
    Returns: ("GPT 5.2", "All deaths with minimum 500 thousand views")
    """
    base = os.path.basename(filename)
    
    # 1. Extract Model Name (everything before the first underscore)
    parts = base.split('_')
    raw_model = parts[0]
    
    # Capitalize logic: "gpt-5.2" -> "GPT 5.2"
    if "gpt" in raw_model.lower():
        model_name = raw_model.upper().replace("-", " ")
    else:
        # Fallback for other models: capitalize first letter
        model_name = raw_model.replace("-", " ").title()

    # 2. Extract Min Views using Regex
    views_match = re.search(r"minviews-(\d+)", base)
    subtitle = ""
    if views_match:
        views_count = int(views_match.group(1))
        readable_views = format_number(views_count)
        subtitle = f"All deaths with minimum {readable_views} views"
    else:
        subtitle = "Knowledge of deaths by month"

    return model_name, subtitle

# =============================================================================
# Reporting
# =============================================================================

def print_summary(results: List[Dict], stats_by_month: Dict[str, Dict], min_samples: int) -> None:
    total = len(results)
    if total == 0: return

    total_correct = sum(1 for r in results if r["llm_knows_death"] is True)
    total_incorrect = sum(1 for r in results if r["llm_knows_death"] is False)
    
    print("\n" + "=" * 60)
    print("Overall")
    print("=" * 60)
    print(f"Total tested: {total}")
    print(f"  Knows death: {total_correct} ({total_correct/total*100:.1f}%)")
    print(f"  Doesn't know: {total_incorrect} ({total_incorrect/total*100:.1f}%)")

# =============================================================================
# Plotting
# =============================================================================

def plot_monthly_accuracy(
    stats_by_month: Dict[str, Dict],
    input_filename: str,
    output_file: str,
    min_samples: int,
) -> None:
    months: List[str] = []
    accuracies: List[float] = []

    # Sort and filter data
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

    # Calculate Moving Average (Window size 3 months)
    moving_avg = calculate_moving_average(accuracies, window_size=3)

    # Determine figure width based on data points
    fig_width = max(14, len(months) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # Set background color for better contrast
    ax.set_facecolor('#f8f9fa')

    # 1. Plot Bars
    bars = ax.bar(range(len(months)), accuracies, color="#4d7fcd", edgecolor="#3b5b95", width=0.8, label="Monthly Accuracy", zorder=2)
    
    # 2. Plot Moving Average Line
    ax.plot(range(len(months)), moving_avg, color="#d9534f", linewidth=2.5, marker='o', markersize=3, label="3-Month Moving Avg", zorder=3)

    # 3. Handle X-Axis Labels (Show every 3rd label to prevent crowding)
    tick_step = 3
    ax.set_xticks(range(0, len(months), tick_step))
    ax.set_xticklabels([months[i] for i in range(0, len(months), tick_step)], rotation=45, ha="right", fontsize=10)

    # 4. Generate Dynamic Titles
    model_name, subtitle = parse_title_info(input_filename)
    
    plt.title(f"{model_name}\n{subtitle}", fontsize=16, fontweight='bold', pad=20)
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Death month", fontsize=12)

    # 5. Grid and Limits
    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    # 6. Legend
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=1)

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
        help="Input CSV file with test results",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="cutoff_plot_monthly.png",
        help="Output PNG for the monthly plot",
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
        return

    stats_by_month = calculate_accuracy_by_month(results)

    print_summary(results, stats_by_month, args.min_samples)

    if not args.no_plot:
        # If the user didn't supply a custom --plot filename (left the default),
        # generate one from the input filename so it contains model and date info.
        default_plot_name = "cutoff_plot_monthly.png"
        if not args.plot or args.plot == default_plot_name:
            input_base = os.path.splitext(os.path.basename(args.input))[0]
            # Save the plot next to the input file for convenience
            input_dir = os.path.dirname(os.path.abspath(args.input)) or os.getcwd()
            plot_file = os.path.join(input_dir, f"{input_base}_monthly.png")
        else:
            plot_file = args.plot

        plot_monthly_accuracy(stats_by_month, args.input, plot_file, args.min_samples)


if __name__ == "__main__":
    main()