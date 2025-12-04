# Filename: analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tabulate import tabulate

# Set professional style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

def load_data(filepath):
    try:
        df = pd.read_csv(filepath, parse_dates=['Start Time', 'End Time'])
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}. Please run resource_monitor.py first.")
        exit(1)

def refine_task_names(df):
    # Helper to make pod names readable task names
    def get_task_name(row):
        name = row['Pod Name']
        if row['Pipeline Type'] == 'Monolithic':
            return 'Monolithic Execution'

        # Heuristics for Distributed Pipeline
        if 'download-model' in name: return '1. Download Model'
        if 'download-dataset' in name: return '2. Download Data'
        if 'ptj-worker-synthetic' in name: return '3. Training (Worker)'
        if 'launch-training-job' in name: return '3.1 Training (Orchestrator)'
        if 'merge-adapter' in name: return '4. Merge Adapter'
        if 'run-inference' in name: return '5. Inference'
        return name.split('-')[0] # Fallback

    df['Task Name'] = df.apply(get_task_name, axis=1)
    return df

def generate_summary_table(df):
    summary = df.groupby('Pipeline Type').agg(
        Total_CPU_Core_Hours=('CPU Core-Hours', 'sum'),
        Total_Memory_GiB_Hours=('Memory GiB-Hours', 'sum'),
        Total_GPU_Hours=('GPU-Hours', 'sum'),
        Total_Pod_Duration_Hours=('Duration (Hours)', 'sum')
    )

    # Calculate Wall-Clock Time (Make-span)
    def calculate_wall_clock(x):
        if x['Start Time'].empty or x['End Time'].empty:
            return 0
        return (x['End Time'].max() - x['Start Time'].min()).total_seconds() / 3600.0

    wall_clock = df.groupby('Pipeline Type').apply(calculate_wall_clock)
    summary['Wall_Clock_Time_Hours'] = wall_clock

    # Reorder
    summary = summary[['Wall_Clock_Time_Hours', 'Total_CPU_Core_Hours',
                       'Total_Memory_GiB_Hours', 'Total_GPU_Hours', 'Total_Pod_Duration_Hours']]

    # Calculate Differences
    summary_t = summary.T
    dist_vals = summary_t.get('Distributed', pd.Series(dtype=float))
    mono_vals = summary_t.get('Monolithic', pd.Series(dtype=float))

    if not dist_vals.empty and not mono_vals.empty:
        diff = mono_vals - dist_vals
        # Handle division by zero if a resource wasn't used
        pct_change = (diff / dist_vals).fillna(0) * 100
        summary_t['Difference (M-D)'] = diff
        summary_t['% Change'] = pct_change

    print("\n" + "="*90)
    print("RESOURCE CONSUMPTION COMPARISON SUMMARY".center(90))
    print("="*90)
    print(tabulate(summary_t, headers='keys', tablefmt="grid", floatfmt=".3f"))
    return summary

def plot_resource_comparison(summary):
    plot_data = summary.reset_index().melt(id_vars=['Pipeline Type'],
                             value_vars=['Total_CPU_Core_Hours', 'Total_Memory_GiB_Hours', 'Total_GPU_Hours'],
                             var_name='Resource Type', value_name='Total Resource-Hours')

    plot_data['Resource Type'] = plot_data['Resource Type'].map({
        'Total_CPU_Core_Hours': 'CPU (Core-Hours)',
        'Total_Memory_GiB_Hours': 'Memory (GiB-Hours)',
        'Total_GPU_Hours': 'GPU (Hours)'
    })

    plt.figure(figsize=(12, 7))
    g = sns.barplot(data=plot_data, x='Resource Type', y='Total Resource-Hours', hue='Pipeline Type', palette="colorblind")
    plt.title('Total Resource Allocation Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Total Allocated Resource-Hours')
    plt.xlabel('')

    # Add annotations
    for p in g.patches:
        height = p.get_height()
        if height > 0:
            g.annotate(format(height, '.2f'),
                       (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("resource_comparison.png")
    print("\nSaved visualization: resource_comparison.png")

def plot_gantt(df, pipeline_type, ax):
    """Helper function to plot the Gantt chart for a specific pipeline."""
    df_plot = df[df['Pipeline Type'] == pipeline_type].copy()
    if df_plot.empty: return

    # Calculate relative times in minutes
    start_time = df_plot['Start Time'].min()
    df_plot['Relative Start'] = (df_plot['Start Time'] - start_time).dt.total_seconds() / 60.0
    df_plot['Duration Mins'] = df_plot['Duration (Hours)'] * 60.0

    # Sort by start time for plotting order
    df_plot = df_plot.sort_values(by=['Relative Start'], ascending=True)

    # Color based on resource intensity
    def get_color(row):
        if row['GPUs Requested'] > 0:
            return '#e74c3c' # Red for GPU
        elif row['Memory GiB Requested'] > 10:
            return '#f39c12' # Orange for High Memory
        else:
            return '#3498db' # Blue for CPU/Low resource

    colors = df_plot.apply(get_color, axis=1)

    # Plot bars against Task Name
    ax.barh(df_plot['Task Name'], df_plot['Duration Mins'], left=df_plot['Relative Start'], color=colors, edgecolor='black')

    # Add resource labels inside the bars
    for i, row in df_plot.iterrows():
        label = f"CPU: {row['CPU Cores Requested']:.1f} | Mem: {row['Memory GiB Requested']:.1f}G | GPU: {int(row['GPUs Requested'])}"
        text_x = row['Relative Start'] + row['Duration Mins'] / 2
        # Only add text if the bar is long enough
        if row['Duration Mins'] > 5:
             ax.text(text_x, row['Task Name'], label, ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    ax.set_title(f'{pipeline_type} Pipeline Timeline', fontsize=16)
    ax.invert_yaxis() # Put the first task on top
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    return get_color

def plot_time_efficiency(summary):
    """Visualizes Wall Clock Time vs Total Pod Duration to show overhead."""
    data = summary[['Wall_Clock_Time_Hours', 'Total_Pod_Duration_Hours']].copy()

    # Overhead = Wall Clock - Total Pod Duration.
    # If negative (due to parallelism), we clamp it at 0 for visualization purposes.
    data['Overhead (Scheduling/Provisioning)'] = data['Wall_Clock_Time_Hours'] - data['Total_Pod_Duration_Hours']
    data['Overhead (Scheduling/Provisioning)'] = data['Overhead (Scheduling/Provisioning)'].apply(lambda x: max(0, x))
    data['Execution Time (Total Pod Duration)'] = data['Total_Pod_Duration_Hours']

    # Prepare data for stacked bar chart
    plot_data = data[['Execution Time (Total Pod Duration)', 'Overhead (Scheduling/Provisioning)']]

    plt.figure(figsize=(8, 6))

    # Stacked bar chart
    plot_data.plot(kind='bar', stacked=True, color=sns.color_palette("viridis", 2))

    plt.title("Time Efficiency and Orchestration Overhead", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Time (Hours)")
    plt.xlabel("Pipeline Type")
    plt.xticks(rotation=0)

    # Add total labels
    for i, (index, row) in enumerate(data.iterrows()):
        total = row['Wall_Clock_Time_Hours']
        plt.text(i, total + 0.05, f'Total: {total:.2f}h', ha='center')

    plt.tight_layout()
    plt.savefig("time_efficiency.png")
    print("Saved visualization: time_efficiency.png")

def main():
    parser = argparse.ArgumentParser(description="Analyze KFP Resource Monitoring Results")
    parser.add_argument("--input_csv", default="comparison_results_all.csv", help="The combined CSV file generated by resource_monitor.py")
    args = parser.parse_args()

    df = load_data(args.input_csv)
    if df.empty:
        print("Input CSV is empty. Cannot perform analysis."); return

    df = refine_task_names(df)
    summary = generate_summary_table(df)

    if len(summary) > 1:
        plot_resource_comparison(summary)

        # Generate Gantt Chart (Side-by-side or stacked timelines)
        fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

        # Create custom legend for Gantt chart
        gpu_patch = plt.Rectangle((0,0),1,1,fc='#e74c3c', edgecolor='black')
        mem_patch = plt.Rectangle((0,0),1,1,fc='#f39c12', edgecolor='black')
        cpu_patch = plt.Rectangle((0,0),1,1,fc='#3498db', edgecolor='black')

        plot_gantt(df, 'Distributed', axes[0])
        plot_gantt(df, 'Monolithic', axes[1])

        fig.legend([gpu_patch, mem_patch, cpu_patch], ['GPU Intensive', 'High Memory (CPU)', 'Low Resource/CPU'], loc='upper right')

        fig.supxlabel('Timeline (Minutes)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle('Execution Timeline and Resource Allocation', fontsize=20, fontweight='bold')
        plt.savefig("gantt_chart.png")
        print("Saved visualization: gantt_chart.png")

        plot_time_efficiency(summary)

if __name__ == '__main__':
    main()