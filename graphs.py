import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
def load_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def filter_data(df, benchmark=None, bit=None, category=None, simd=None, result=None):
    query_df = df.copy()

    if benchmark:
        query_df = query_df[query_df['Benchmark'] == benchmark]
    if bit is not None:
        query_df = query_df[query_df['bit'] == bit]
    if category:
        query_df = query_df[query_df['arithmetic_type'] == category]
    if simd:
        query_df = query_df[query_df['simd_type'] == simd]
    if result:
        query_df = query_df[query_df['result'] == result]
    
    return query_df

def plot_sdc_percentage_by_bit(df, benchmarks):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 6))

    for benchmark in benchmarks:
        benchmark_df = df[df["Benchmark"] == benchmark]
        total = len(benchmark_df)
        if total == 0:
            continue

        bit_groups = benchmark_df.groupby("bit")
        bit_percentages = {
            bit: len(group[group["result"] == "SDC"]) / len(group)
            for bit, group in bit_groups
        }

        bits = sorted(bit_percentages.keys())
        values = [bit_percentages[bit] for bit in bits]
        plt.plot(bits, values, marker="o", label=benchmark)

    plt.xlabel("Bit")
    plt.ylabel("SDC (%)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/sdc_percentage_by_bit.pdf")
    plt.close()
def plot_sdc_percentage_by_arithmetic_type_per_bit(df, benchmark, output_dir="plots"):
    os.makedirs("plots/percentages_arithmetic", exist_ok=True)
    benchmark_df = df[df["Benchmark"] == benchmark]
    if benchmark_df.empty:
        return

    categories = benchmark_df["arithmetic_type"].unique()
    plt.figure(figsize=(12, 6))

    for category in categories:
        category_df = benchmark_df[benchmark_df["arithmetic_type"] == category]
        bit_values = sorted(category_df["bit"].unique())
        percentages = []

        for bit in bit_values:
            group = category_df[category_df["bit"] == bit]
            total = len(group)
            mismatches = len(group[group["result"] == "SDC"])
            percentage = mismatches / total if total > 0 else 0
            percentages.append(percentage)

        plt.plot(bit_values, percentages, marker="o", label=category)

    plt.xlabel("Bit")
    plt.ylabel("SDC (%)")
   
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/percentages_arithmetic/{benchmark}_sdc_arithmetic.pdf")
    plt.close()


def plot_sdc_percentage_by_simd_type_per_bit(df, benchmark):
    os.makedirs("plots/percentages_simd", exist_ok=True)
    benchmark_df = df[df["Benchmark"] == benchmark]
    if benchmark_df.empty:
        return

    simd_types = benchmark_df["simd_type"].unique()
    plt.figure(figsize=(12, 6))

    for simd in simd_types:
        simd_df = benchmark_df[benchmark_df["simd_type"] == simd]
        bit_values = sorted(simd_df["bit"].unique())
        percentages = []

        for bit in bit_values:
            group = simd_df[simd_df["bit"] == bit]
            total = len(group)
            mismatches = len(group[group["result"] == "SDC"])
            percentage = mismatches / total if total > 0 else 0
            percentages.append(percentage)

        plt.plot(bit_values, percentages, marker="o", label=simd)

    plt.xlabel("Bit Flipped")
    plt.ylabel("SDC (%)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/percentages_simd/{benchmark}_sdc_simd.pdf")
    plt.close()

def plot_avg_sdc_percentage_per_arithmetic(df, benchmark):
    os.makedirs("plots/average_arithmetic", exist_ok=True)
    benchmark_df = df[df["Benchmark"] == benchmark]
    if benchmark_df.empty:
        return

    category_avg_sdc = {}
    for category in ["add", "mul", "sub", "div"]:
        cat_df = benchmark_df[benchmark_df["arithmetic_type"] == category]
        if cat_df.empty:
            category_avg_sdc[category] = 0
            continue

        bit_values = cat_df["bit"].unique()
        bit_sdc_percentages = []
        for bit in bit_values:
            group = cat_df[cat_df["bit"] == bit]
            sdc = len(group[group["result"] == "SDC"])
            percentage = sdc / len(group) if len(group) > 0 else 0
            bit_sdc_percentages.append(percentage)

        category_avg_sdc[category] = (
            sum(bit_sdc_percentages) / len(bit_sdc_percentages)
            if bit_sdc_percentages
            else 0
        )

    categories = list(category_avg_sdc.keys())
    values = [category_avg_sdc[cat]*100 for cat in categories]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, values)
    plt.xlabel("Arithmetic Type")
    plt.ylabel("Average SDC Percent")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/average_arithmetic/{benchmark}_avg_sdc_arithmetic.pdf")
    plt.close()

def plot_avg_sdc_percentage_per_simd(df, benchmark):
    os.makedirs("plots/average_simd", exist_ok=True)
    benchmark_df = df[df["Benchmark"] == benchmark]
    if benchmark_df.empty:
        return

    simd_avg_sdc = {}
    for simd in ["ss", "sd", "ps", "pd"]:
        simd_df = benchmark_df[benchmark_df["simd_type"] == simd]
        if simd_df.empty:
            simd_avg_sdc[simd] = 0
            continue

        bit_values = simd_df["bit"].unique()
        bit_sdc_percentages = []
        for bit in bit_values:
            group = simd_df[simd_df["bit"] == bit]
            sdc = len(group[group["result"] == "SDC"])
            percentage = sdc / len(group) if len(group) > 0 else 0
            bit_sdc_percentages.append(percentage)

        simd_avg_sdc[simd] = (
            sum(bit_sdc_percentages) / len(bit_sdc_percentages)
            if bit_sdc_percentages
            else 0
        )

    simds = list(simd_avg_sdc.keys())
    values = [simd_avg_sdc[s]*100 for s in simds]

    plt.figure(figsize=(10, 6))
    plt.bar(simds, values)
    plt.xlabel("SIMD Type")
    plt.ylabel("Average SDC Percentage")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/average_simd/{benchmark}_avg_sdc_simd.pdf")
    plt.close()

def plot_avg_sdc_percentage_per_arithmetic_all(df, benchmarks):
    os.makedirs("plots/average_arithmetic", exist_ok=True)
    categories = ["add", "mul", "sub", "div"]
    benchmark_data = {b: [] for b in benchmarks}

    for benchmark in benchmarks:
        benchmark_df = df[df["Benchmark"] == benchmark]
        for category in categories:
            cat_df = benchmark_df[benchmark_df["arithmetic_type"] == category]
            if cat_df.empty:
                benchmark_data[benchmark].append(0)
                continue
            bit_values = cat_df["bit"].unique()
            bit_sdc_percentages = []
            for bit in bit_values:
                group = cat_df[cat_df["bit"] == bit]
                sdc = len(group[group["result"] == "SDC"])
                percentage = sdc / len(group) if len(group) > 0 else 0
                bit_sdc_percentages.append(percentage)
            avg = sum(bit_sdc_percentages) / len(bit_sdc_percentages) if bit_sdc_percentages else 0
            benchmark_data[benchmark].append(avg*100)

    x = np.arange(len(categories))
    width = 0.8 / len(benchmarks)  # space out bars

    plt.figure(figsize=(12, 6))
    for idx, benchmark in enumerate(benchmarks):
        plt.bar(x + idx * width, benchmark_data[benchmark], width=width, label=benchmark)

    plt.xlabel("Arithmetic Type")
    plt.ylabel("Average SDC (%)")
    plt.xticks(x + width * (len(benchmarks) - 1) / 2, categories)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/average_arithmetic/all_benchmarks_avg_sdc_arithmetic.pdf")
    plt.close()


def plot_avg_sdc_percentage_per_simd_all(df, benchmarks):
    os.makedirs("plots/average_simd", exist_ok=True)
    simd_types = ["ss", "sd", "ps", "pd"]
    benchmark_data = {b: [] for b in benchmarks}

    for benchmark in benchmarks:
        benchmark_df = df[df["Benchmark"] == benchmark]
        for simd in simd_types:
            simd_df = benchmark_df[benchmark_df["simd_type"] == simd]
            if simd_df.empty:
                benchmark_data[benchmark].append(0)
                continue
            bit_values = simd_df["bit"].unique()
            bit_sdc_percentages = []
            for bit in bit_values:
                group = simd_df[simd_df["bit"] == bit]
                sdc = len(group[group["result"] == "SDC"])
                percentage = sdc / len(group) if len(group) > 0 else 0
                bit_sdc_percentages.append(percentage)
            avg = sum(bit_sdc_percentages) / len(bit_sdc_percentages) if bit_sdc_percentages else 0
            benchmark_data[benchmark].append(avg*100)

    x = np.arange(len(simd_types))
    width = 0.8 / len(benchmarks)

    plt.figure(figsize=(12, 6))
    for idx, benchmark in enumerate(benchmarks):
        plt.bar(x + idx * width, benchmark_data[benchmark], width=width, label=benchmark)

    plt.xlabel("SIMD Type")
    plt.ylabel("Average SDC Percentage per Bit")
    plt.xticks(x + width * (len(benchmarks) - 1) / 2, simd_types)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/average_simd/all_benchmarks_avg_sdc_simd.pdf")
    plt.close()


def plot_avg_sdc_per_category_by_benchmark(df, benchmarks):
    os.makedirs("plots/average_arithmetic", exist_ok=True)
    categories = ["add", "mul", "sub", "div"]
    category_data = {cat: [] for cat in categories}

    for benchmark in benchmarks:
        benchmark_df = df[df["Benchmark"] == benchmark]
        for cat in categories:
            cat_df = benchmark_df[benchmark_df["arithmetic_type"] == cat]
            if cat_df.empty:
                category_data[cat].append(0)
                continue
            bit_values = cat_df["bit"].unique()
            bit_sdc_percentages = []
            for bit in bit_values:
                group = cat_df[cat_df["bit"] == bit]
                sdc = len(group[group["result"] == "SDC"])
                percentage = sdc / len(group) if len(group) > 0 else 0
                bit_sdc_percentages.append(percentage)
            avg = sum(bit_sdc_percentages) / len(bit_sdc_percentages) if bit_sdc_percentages else 0
            category_data[cat].append(avg*100)

    x = np.arange(len(benchmarks))
    width = 0.8 / len(categories)

    plt.figure(figsize=(12, 6))
    for idx, cat in enumerate(categories):
        plt.bar(x + idx * width, category_data[cat], width=width, label=cat)

    plt.xlabel("Application")
    plt.ylabel("Average SDC (%)")
    plt.xticks(x + width * (len(categories) - 1) / 2, benchmarks, rotation=45)
    plt.legend(title="Arithmetic Type")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/average_arithmetic/benchmarks_grouped_by_category.pdf")
    plt.close()


def plot_avg_sdc_per_simd_by_benchmark(df, benchmarks):
    os.makedirs("plots/average_simd", exist_ok=True)
    simd_types = ["ss", "sd", "ps", "pd"]
    simd_data = {simd: [] for simd in simd_types}

    for benchmark in benchmarks:
        benchmark_df = df[df["Benchmark"] == benchmark]
        for simd in simd_types:
            simd_df = benchmark_df[benchmark_df["simd_type"] == simd]
            if simd_df.empty:
                simd_data[simd].append(0)
                continue
            bit_values = simd_df["bit"].unique()
            bit_sdc_percentages = []
            for bit in bit_values:
                group = simd_df[simd_df["bit"] == bit]
                sdc = len(group[group["result"] == "SDC"])
                percentage = sdc / len(group) if len(group) > 0 else 0
                bit_sdc_percentages.append(percentage)
            avg = sum(bit_sdc_percentages) / len(bit_sdc_percentages) if bit_sdc_percentages else 0
            simd_data[simd].append(avg*100)

    x = np.arange(len(benchmarks))
    width = 0.8 / len(simd_types)

    plt.figure(figsize=(12, 6))
    for idx, simd in enumerate(simd_types):
        plt.bar(x + idx * width, simd_data[simd], width=width, label=simd)

    plt.xlabel("Application")
    plt.ylabel("Average SDC (%)")
    plt.xticks(x + width * (len(simd_types) - 1) / 2, benchmarks, rotation=45)
    plt.legend(title="SIMD Type")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/average_simd/benchmarks_grouped_by_simd.pdf")
    plt.close()


def plot_avg_sdc_per_category_by_benchmark_weight(df, benchmarks):
    os.makedirs("plots/average_arithmetic", exist_ok=True)
    categories = ["add", "mul", "sub", "div"]
    category_data = {cat: [] for cat in categories}

    for benchmark in benchmarks:
        benchmark_df = df[df["Benchmark"] == benchmark]
        for cat in categories:
            cat_df = benchmark_df[benchmark_df["arithmetic_type"] == cat]
            if cat_df.empty:
                category_data[cat].append(0)
                continue
            bit_values = cat_df["bit"].unique()
            bit_sdc_percentages = []
            for bit in bit_values:
                group = cat_df[cat_df["bit"] == bit]
                sdc = len(group[group["result"] == "SDC"])
                percentage = sdc / len(group) if len(group) > 0 else 0
                bit_sdc_percentages.append(percentage)
            avg = sum(bit_sdc_percentages) / len(bit_sdc_percentages) if bit_sdc_percentages else 0
            weight=1/(len(cat_df) / len(benchmark_df))
            category_data[cat].append(avg*100*weight)

    x = np.arange(len(benchmarks))
    width = 0.8 / len(categories)

    plt.figure(figsize=(12, 6))
    for idx, cat in enumerate(categories):
        plt.bar(x + idx * width, category_data[cat], width=width, label=cat)

    plt.xlabel("Application")
    plt.ylabel("Weighted Average SDC")
    plt.xticks(x + width * (len(categories) - 1) / 2, benchmarks, rotation=45)
    plt.legend(title="Arithmetic Type")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/average_arithmetic/benchmarks_grouped_by_category_weighted.pdf")
    plt.close()


def plot_avg_sdc_per_simd_by_benchmark_weight(df, benchmarks):
    os.makedirs("plots/average_simd", exist_ok=True)
    simd_types = ["ss", "sd", "ps", "pd"]
    simd_data = {simd: [] for simd in simd_types}

    for benchmark in benchmarks:
        benchmark_df = df[df["Benchmark"] == benchmark]
        for simd in simd_types:
            simd_df = benchmark_df[benchmark_df["simd_type"] == simd]
            if simd_df.empty:
                simd_data[simd].append(0)
                continue
            bit_values = simd_df["bit"].unique()
            bit_sdc_percentages = []
            for bit in bit_values:
                group = simd_df[simd_df["bit"] == bit]
                sdc = len(group[group["result"] == "SDC"])
                percentage = sdc / len(group) if len(group) > 0 else 0
                bit_sdc_percentages.append(percentage)
            avg = sum(bit_sdc_percentages) / len(bit_sdc_percentages) if bit_sdc_percentages else 0
            weight=1/(len(simd_df) / len(benchmark_df))
            simd_data[simd].append(avg*100*weight)

    x = np.arange(len(benchmarks))
    width = 0.8 / len(simd_types)

    plt.figure(figsize=(12, 6))
    for idx, simd in enumerate(simd_types):
        plt.bar(x + idx * width, simd_data[simd], width=width, label=simd)

    plt.xlabel("Application")
    plt.ylabel("Average SDC (%)")
    plt.xticks(x + width * (len(simd_types) - 1) / 2, benchmarks, rotation=45)
    plt.legend(title="SIMD Type")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/average_simd/benchmarks_grouped_by_simd_weighted.pdf")
    plt.close()




def plot_general_summary(df, output_folder="plots/general_view", colors=None):
    os.makedirs(output_folder, exist_ok=True)
    result_types = ["No Issues", "Crashed", "SDC"]
    result_counts = [len(df[df["result"] == res]) for res in result_types]

    if not colors:
        colors = {"No Issues": "#4CAF50", "Crashed": "#F44336", "SDC": "#FFC107"}

    plt.figure(figsize=(8, 6))
    plt.bar(result_types, result_counts, color=[colors[res] for res in result_types])
    plt.ylabel("Total Count")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "general_summary.pdf"))
    plt.close()

def plot_normalized_result_distribution_by_category(df, benchmarks, category_col, folder_name, colors):
    output_folder = os.path.join("plots", folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for benchmark in benchmarks:
        subset = df[df["Benchmark"] == benchmark]
        grouped = subset.groupby([category_col, "result"]).size().unstack(fill_value=0)

        # Normalize to percentages
        normalized = grouped.div(grouped.sum(axis=1), axis=0) * 100
          # Ensure consistent order

        ax = normalized.plot(kind="bar", stacked=False, color=[colors.get(col, "#000000") for col in normalized.columns])
        
        plt.xlabel(category_col.capitalize())
        plt.ylabel("Percentage")
        plt.xticks(rotation=45)
        plt.legend(title="Result")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{benchmark}_{category_col}.pdf"))
        plt.close()


def plot_stacked_sdc_distribution_by_benchmark(df, benchmarks, colors=None):
    os.makedirs("plots/stacked_distribution", exist_ok=True)

    # Default color map if none provided
    if colors is None:
        colors = {
            "SDC": "#e74c3c",
            "No Issues": "#2ecc71",
            "Crashed": "#3498db"
        }

    result_types = ["SDC", "No Issues", "Crashed"]
    all_bits = sorted(df["bit"].unique())

    for benchmark in benchmarks:
        bit_data = {result: [] for result in result_types}

        for bit in all_bits:
            subset = df[(df["Benchmark"] == benchmark) & (df["bit"] == bit)]
            total = len(subset)
            for result in result_types:
                count = len(subset[subset["result"] == result])
                bit_data[result].append(count / total if total > 0 else 0)

        x = np.arange(len(all_bits))
        bottom = np.zeros(len(all_bits))

        plt.figure(figsize=(12, 6))
        for result in result_types:
            plt.bar(x, bit_data[result], bottom=bottom, label=result, color=colors.get(result, None))
            bottom += np.array(bit_data[result])

        plt.xlabel("Bit Position")
        plt.ylabel("Normalized Result Count")
        plt.xticks(x, all_bits, rotation=45)
        plt.legend(title="Result Type")
        plt.tight_layout()
        plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
        plt.savefig(f"plots/stacked_distribution/{benchmark}_stacked_distribution.pdf")
        plt.close()

def main():
    full_sweeps_path = "full_sweeps.csv"
    df_sweeps = load_csv(full_sweeps_path)
    initial_path = "initial_results.csv"
    initial_res = load_csv(initial_path)
    
    colors = {'No Issues': '#b2d8d8', 'Crashed': '#008080', 'SDC': '#004c4c'}
    org_benchmarks = ["namd_r","nab_r","lbm_r","roms_r","wrf_r"]
    benchmarks = ["namd_r","nab_r","lbm_r","roms_r","wrf_r","povray_r","imagik_r","fotonik3d_r"]
    plot_sdc_percentage_by_bit(df_sweeps, benchmarks)
    plot_avg_sdc_per_category_by_benchmark(df_sweeps, benchmarks)
    plot_avg_sdc_per_simd_by_benchmark(df_sweeps, benchmarks)
    plot_avg_sdc_per_category_by_benchmark_weight(df_sweeps, benchmarks)
    plot_avg_sdc_per_simd_by_benchmark_weight(df_sweeps, benchmarks)
    plot_general_summary(initial_res,colors=colors)
    plot_general_summary(df_sweeps,"plots/general_view_full",colors=colors)
    plot_stacked_sdc_distribution_by_benchmark(initial_res, benchmarks, colors)
    for benchmark in benchmarks:
        plot_sdc_percentage_by_arithmetic_type_per_bit(df_sweeps, benchmark)
        plot_sdc_percentage_by_simd_type_per_bit(df_sweeps, benchmark)
        plot_avg_sdc_percentage_per_arithmetic(df_sweeps, benchmark)
        plot_avg_sdc_percentage_per_simd(df_sweeps, benchmark)


    plot_normalized_result_distribution_by_category(initial_res, org_benchmarks, "arithmetic_type", "arithmetic", colors)
    plot_normalized_result_distribution_by_category(initial_res, org_benchmarks, "simd_type", "SIMD", colors)
    # Add more queries or export filtered results
    # filtered_df = filter_data(df, category='add')
    # filtered_df.to_csv("filtered_output.csv", index=False)

if __name__ == "__main__":
    main()