"""
Script 2: Process LLM test results and generate statistics/plots.
Reads results CSV file and produces analysis.
"""

import csv
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
import argparse

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from statistics import mean, median


# ============================================================================
# Data Loading
# ============================================================================

def load_results_from_csv(csv_file: str) -> List[Dict]:
    """Load results data from CSV file."""
    results = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert llm_knows_death string to bool/None
            knows_str = row.get('llm_knows_death', '')
            if knows_str == 'True':
                row['llm_knows_death'] = True
            elif knows_str == 'False':
                row['llm_knows_death'] = False
            else:
                row['llm_knows_death'] = None
            results.append(row)
    return results


# ============================================================================
# Statistics Calculation
# ============================================================================

def calculate_pageview_stats(deaths: List[Dict]) -> Dict:
    """
    Calculate pageview statistics (min, max, mean, median) for a list of deaths.
    """
    pageviews = [int(d.get('pageviews', 0)) for d in deaths]
    if not pageviews:
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'count': 0}
    
    return {
        'min': min(pageviews),
        'max': max(pageviews),
        'mean': mean(pageviews),
        'median': median(pageviews),
        'count': len(pageviews)
    }


def calculate_accuracy_by_date(results: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate accuracy statistics grouped by date.
    """
    stats_by_date = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'unknown': 0, 'total': 0})
    
    for result in results:
        date = result['death_date']
        knows = result['llm_knows_death']
        
        stats_by_date[date]['total'] += 1
        
        if knows is True:
            stats_by_date[date]['correct'] += 1
        elif knows is False:
            stats_by_date[date]['incorrect'] += 1
        else:
            stats_by_date[date]['unknown'] += 1
    
    # Calculate percentages
    for date, stats in stats_by_date.items():
        known_total = stats['correct'] + stats['incorrect']
        if known_total > 0:
            stats['accuracy'] = stats['correct'] / known_total * 100
        else:
            stats['accuracy'] = None
    
    return dict(stats_by_date)


def calculate_accuracy_by_month(results: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate accuracy statistics grouped by month.
    """
    stats_by_month = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'unknown': 0, 'total': 0})
    
    for result in results:
        month = result['death_date'][:7]  # YYYY-MM
        knows = result['llm_knows_death']
        
        stats_by_month[month]['total'] += 1
        
        if knows is True:
            stats_by_month[month]['correct'] += 1
        elif knows is False:
            stats_by_month[month]['incorrect'] += 1
        else:
            stats_by_month[month]['unknown'] += 1
    
    # Calculate percentages
    for month, stats in stats_by_month.items():
        known_total = stats['correct'] + stats['incorrect']
        if known_total > 0:
            stats['accuracy'] = stats['correct'] / known_total * 100
        else:
            stats['accuracy'] = None
    
    return dict(stats_by_month)


def calculate_accuracy_by_pageviews(results: List[Dict], num_bins: int = 10) -> List[Dict]:
    """
    Calculate accuracy statistics grouped by pageview ranges using logarithmic binning.
    Returns list of dicts with bin info and accuracy.
    """
    # Filter to results with valid pageviews and known outcome
    valid_results = [
        r for r in results 
        if r['llm_knows_death'] is not None and int(r.get('pageviews', 0)) > 0
    ]
    
    if not valid_results:
        return []
    
    # Get pageview range
    pageviews = [int(r['pageviews']) for r in valid_results]
    min_pv = min(pageviews)
    max_pv = max(pageviews)
    
    # Create logarithmic bin edges
    log_min = np.log10(min_pv)
    log_max = np.log10(max_pv)
    bin_edges = np.logspace(log_min, log_max, num_bins + 1)
    
    # Assign results to bins and calculate accuracy
    bins = []
    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        
        # Get results in this bin
        bin_results = [
            r for r in valid_results
            if low <= int(r['pageviews']) < high or (i == len(bin_edges) - 2 and int(r['pageviews']) == high)
        ]
        
        if bin_results:
            correct = sum(1 for r in bin_results if r['llm_knows_death'] is True)
            total = len(bin_results)
            accuracy = correct / total * 100
            
            bins.append({
                'low': low,
                'high': high,
                'mid': np.sqrt(low * high),  # Geometric mean for log scale
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            })
    
    return bins


# ============================================================================
# Plotting
# ============================================================================

def plot_results(stats_by_date: Dict[str, Dict], model: str, output_file: str = "cutoff_plot.png"):
    """
    Plot the accuracy over time.
    """
    # Filter out dates with no valid accuracy
    dates = []
    accuracies = []
    
    for date_str in sorted(stats_by_date.keys()):
        stats = stats_by_date[date_str]
        if stats['accuracy'] is not None:
            dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
            accuracies.append(stats['accuracy'])
    
    if not dates:
        print("No data to plot!")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot raw data points
    ax.scatter(dates, accuracies, alpha=0.5, s=30, label='Daily accuracy')
    
    # Add rolling average if we have enough points
    if len(dates) > 7:
        # Convert to numpy for rolling average
        dates_num = mdates.date2num(dates)
        window = min(7, len(dates) // 3)
        
        # Simple moving average
        rolling_acc = np.convolve(accuracies, np.ones(window)/window, mode='valid')
        rolling_dates = dates[window-1:]
        
        ax.plot(rolling_dates, rolling_acc, 'r-', linewidth=2, label=f'{window}-day moving average')
    
    # Formatting
    ax.set_xlabel('Death Date', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{model} Knowledge of Deaths Over Time\n(Higher = knows more deaths from that date)', fontsize=14)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha='right')
    
    ax.set_ylim(-5, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")
    plt.show()


def plot_monthly_results(stats_by_month: Dict[str, Dict], model: str, output_file: str = "cutoff_plot_monthly.png"):
    """
    Plot the accuracy by month (cleaner view).
    """
    months = []
    accuracies = []
    totals = []
    
    for month_str in sorted(stats_by_month.keys()):
        stats = stats_by_month[month_str]
        if stats['accuracy'] is not None:
            months.append(datetime.strptime(month_str + '-15', '%Y-%m-%d'))
            accuracies.append(stats['accuracy'])
            totals.append(stats['correct'] + stats['incorrect'])
    
    if not months:
        print("No data to plot!")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Bar plot
    bar_width = 20  # days
    bars = ax.bar(months, accuracies, width=bar_width, alpha=0.7, color='steelblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, acc, total in zip(bars, accuracies, totals):
        height = bar.get_height()
        ax.annotate(f'{acc:.0f}%\n(n={total})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Formatting
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{model} Knowledge of Deaths by Month\n(Percentage of deaths the model knows about)', fontsize=14)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha='right')
    
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Monthly plot saved to {output_file}")
    plt.show()


def plot_pageviews_vs_accuracy(results: List[Dict], model: str, output_file: str = "cutoff_plot_pageviews.png"):
    """
    Plot accuracy vs pageviews using logarithmic binning.
    Shows how model knowledge correlates with person's notability (pageviews).
    """
    bins = calculate_accuracy_by_pageviews(results, num_bins=10)
    
    if not bins:
        print("No valid data for pageviews plot!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract data
    mids = [b['mid'] for b in bins]
    accuracies = [b['accuracy'] for b in bins]
    totals = [b['total'] for b in bins]
    
    # Scatter plot with size proportional to sample count
    sizes = [max(50, min(500, t * 5)) for t in totals]  # Scale sizes
    scatter = ax.scatter(mids, accuracies, s=sizes, alpha=0.7, c='steelblue', edgecolors='navy')
    
    # Connect points with line
    ax.plot(mids, accuracies, 'b-', alpha=0.5, linewidth=1.5)
    
    # Add labels showing sample size
    for mid, acc, total in zip(mids, accuracies, totals):
        ax.annotate(f'n={total}', (mid, acc), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=8, alpha=0.8)
    
    # Add individual points as rug plot (small ticks at top/bottom)
    valid_results = [
        r for r in results 
        if r['llm_knows_death'] is not None and int(r.get('pageviews', 0)) > 0
    ]
    knows_pv = [int(r['pageviews']) for r in valid_results if r['llm_knows_death'] is True]
    doesnt_know_pv = [int(r['pageviews']) for r in valid_results if r['llm_knows_death'] is False]
    
    # Rug plot at top (knows) and bottom (doesn't know)
    ax.scatter(knows_pv, [102] * len(knows_pv), marker='|', s=30, c='green', alpha=0.3, label='Knows (individual)')
    ax.scatter(doesnt_know_pv, [-2] * len(doesnt_know_pv), marker='|', s=30, c='red', alpha=0.3, label="Doesn't know (individual)")
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Pageviews (60 days after death)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{model} Knowledge vs Pageviews (Notability)\n(Higher pageviews = more notable person)', fontsize=14)
    
    ax.set_ylim(-5, 110)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    
    # Add legend with explanation
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add text explaining bubble size
    ax.text(0.02, 0.98, 'Bubble size = sample count', transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Pageviews plot saved to {output_file}")
    plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process LLM test results and generate statistics/plots.'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='test_results.csv',
        help='Input CSV file with test results (default: test_results.csv)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='LLM',
        help='Model name for plot titles (default: LLM)'
    )
    parser.add_argument(
        '--plot', 
        type=str, 
        default='cutoff_plot.png',
        help='Output plot file (default: cutoff_plot.png)'
    )
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip generating plots, only print statistics'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}")
    results = load_results_from_csv(args.input)
    print(f"Loaded {len(results)} results")
    
    if not results:
        print("No results to analyze!")
        return
    
    # Calculate statistics
    stats_by_date = calculate_accuracy_by_date(results)
    stats_by_month = calculate_accuracy_by_month(results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    total_correct = sum(1 for r in results if r['llm_knows_death'] is True)
    total_incorrect = sum(1 for r in results if r['llm_knows_death'] is False)
    total_unknown = sum(1 for r in results if r['llm_knows_death'] is None)
    
    print(f"Total tested: {len(results)}")
    print(f"  Knows death: {total_correct} ({total_correct/len(results)*100:.1f}%)")
    print(f"  Doesn't know: {total_incorrect} ({total_incorrect/len(results)*100:.1f}%)")
    if total_unknown > 0:
        print(f"  Errors: {total_unknown} ({total_unknown/len(results)*100:.1f}%)")
    
    # Pageview statistics for known vs unknown deaths
    known_deaths = [r for r in results if r['llm_knows_death'] is True]
    unknown_deaths = [r for r in results if r['llm_knows_death'] is False]
    
    print("\nPageview Statistics (60 days after death):")
    if known_deaths:
        known_stats = calculate_pageview_stats(known_deaths)
        print(f"  Deaths model KNOWS (n={known_stats['count']}):")
        print(f"    Min: {known_stats['min']:,}  Max: {known_stats['max']:,}")
        print(f"    Mean: {known_stats['mean']:,.0f}  Median: {known_stats['median']:,.0f}")
    if unknown_deaths:
        unknown_stats = calculate_pageview_stats(unknown_deaths)
        print(f"  Deaths model DOESN'T KNOW (n={unknown_stats['count']}):")
        print(f"    Min: {unknown_stats['min']:,}  Max: {unknown_stats['max']:,}")
        print(f"    Mean: {unknown_stats['mean']:,.0f}  Median: {unknown_stats['median']:,.0f}")
    
    print("\nBy month:")
    for month in sorted(stats_by_month.keys()):
        stats = stats_by_month[month]
        acc = stats['accuracy']
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        print(f"  {month}: {acc_str} ({stats['correct']}/{stats['correct']+stats['incorrect']} known)")
    
    # Plot results (unless --no-plots)
    if not args.no_plots:
        plot_results(stats_by_date, args.model, args.plot)
        
        # Also create monthly plot
        monthly_plot = args.plot.replace('.png', '_monthly.png')
        plot_monthly_results(stats_by_month, args.model, monthly_plot)
        
        # Also create pageviews vs accuracy plot
        pageviews_plot = args.plot.replace('.png', '_pageviews.png')
        plot_pageviews_vs_accuracy(results, args.model, pageviews_plot)


if __name__ == "__main__":
    main()
