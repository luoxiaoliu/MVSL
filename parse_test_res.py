"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        if "summary" in subdir:
            continue
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
        output_results['std'] = std
    print("===")

    return output_results

def find_seed_parent_dirs(base_path, subdir="seed"):
    """Find all unique parent directories containing subdir subdirectories.

    Args:
        base_path (str): The base directory to start searching.

    Returns:
        list: A sorted list of unique parent directories containing subdir subdirectories.
    """
    seed_parent_dirs = set()  # Use a set to store unique directories

    for root, dirs, files in os.walk(base_path):
        if any(d.startswith(subdir) for d in dirs):
            seed_parent_dirs.add(root)

    return sorted(seed_parent_dirs)  # Convert to a sorted list before returning

def parse_summary_log_simple(file_path: str, dataset_index: int = 4) -> tuple:
    """
    从日志中提取各数据集的 accuracy（按路径中 '/' 分割，取 dataset_index 位置为数据集名）
    返回: ({dataset: acc}, avg_acc, avg_std)
    """
    dataset_accs = {}
    avg_acc = avg_std = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Summary of directory:"):
            # 提取路径部分
            path = line.replace("Summary of directory:", "").strip()
            parts = path.split('/')
            if len(parts) > dataset_index:
                dataset = parts[dataset_index]
            else:
                i += 1
                continue  # 跳过格式异常行

            # 下一行应为 accuracy
            if i + 1 < len(lines):
                acc_line = lines[i + 1].strip()
                # 格式: * accuracy: 52.20% +- 3.11%
                if "accuracy:" in acc_line:
                    # 提取数字
                    try:
                        acc_str = acc_line.split(':')[1].split('%')[0].strip()
                        std_str = acc_line.split('+-')[1].split('%')[0].strip()
                        acc = float(acc_str)
                        std = float(std_str)
                        dataset_accs[dataset] = acc
                    except:
                        pass  # 解析失败跳过
            i += 2  # 跳过下一行
        elif line.startswith("Average performance"):
            # 接下来两行是 accuracy 和 std
            if i + 2 < len(lines):
                acc_line = lines[i + 1].strip()  # * accuracy: 72.48%
                std_line = lines[i + 2].strip()  # * std: 6.27%
                try:
                    avg_acc = float(acc_line.split(':')[1].split('%')[0].strip())
                    avg_std = float(std_line.split(':')[1].split('%')[0].strip())
                except:
                    pass
            i += 3
        else:
            i += 1

    return dataset_accs, avg_acc, avg_std


def harmonic_mean(a, b):
    return 2 * a * b / (a + b) if a and b and a + b != 0 else 0.0


def compute_harmonic_averages_simple(base_path, novel_path, dataset_index=4):
    b_accs, b_avg, b_std = parse_summary_log_simple(base_path, dataset_index)
    n_accs, n_avg, n_std = parse_summary_log_simple(novel_path, dataset_index)

    # 数据集调和平均
    common = set(b_accs.keys()) & set(n_accs.keys())
    hm_dataset = {d: round(harmonic_mean(b_accs[d], n_accs[d]), 2) for d in common}

    # 整体调和平均
    hm_overall = round(harmonic_mean(b_avg, n_avg), 2) if b_avg and n_avg else None

    result = {
        "dataset_harmonic": hm_dataset,
        "overall_harmonic_accuracy": hm_overall,
    }

    if b_std and n_std:
        result["overall_harmonic_std"] = round(harmonic_mean(b_std, n_std), 2)

    return result

def compute_harmonic_mean(dir):
    base_summary = osp.join(dir, "train_base", "summary.txt")
    novel_summary = osp.join(dir, "test_new", "summary.txt")
    dataset_index = len(dir.split("/")) + 1

    b_accs, b_avg, b_std = parse_summary_log_simple(base_summary, dataset_index)
    n_accs, n_avg, n_std = parse_summary_log_simple(novel_summary, dataset_index)

    common = set(b_accs.keys()) & set(n_accs.keys())
    datasets_sorted = sorted(common)

    # 找出最长数据集名长度，用于对齐
    max_name_len = max(len(d) for d in datasets_sorted) if datasets_sorted else 6
    # 确保 "Average" 也能对齐
    max_name_len = max(max_name_len, len("Average"))

    for d in datasets_sorted:
        base_val = b_accs[d]
        novel_val = n_accs[d]
        hm_val = round(harmonic_mean(base_val, novel_val), 2)
        print(f"{d:<{max_name_len}}: base={base_val:>6.2f}%  novel={novel_val:>6.2f}%  hm={hm_val:>6.2f}%")

    if b_avg and n_avg:
        hm_overall = round(harmonic_mean(b_avg, n_avg), 2)
        print(f"{'Average':<{max_name_len}}: base={b_avg:>6.2f}%  novel={n_avg:>6.2f}%  hm={hm_overall:>6.2f}%")
    else:
        print(f"{'Average':<{max_name_len}}: N/A")

def compute_few_shot_results(dir):
    print("Average performance")

    # 收集所有 summary.txt 中的 accuracy 和 std
    accuracies = []
    stds = []

    # 遍历 dir 下的所有子目录
    subdirs = listdir_nohidden(dir, sort=True)

    for subdir in subdirs:
        summary_path = osp.join(dir, subdir, "summary.txt")
        if osp.exists(summary_path):
            with open(summary_path, "r") as f:
                content = f.read()
                # 直接解析 "* accuracy: xx% +- yy%" 行
                import re

                match = re.search(r"\* accuracy: ([\d.]+)% \+- ([\d.]+)%", content)
                if match:
                    acc = float(match.group(1))
                    std = float(match.group(2))
                    accuracies.append(acc)
                    stds.append(std)

    if accuracies:
        avg_accuracy = np.mean(accuracies)
        print(f"* accuracy: {avg_accuracy:.2f}%")

    if stds:
        avg_std = np.mean(stds)
        print(f"* std: {avg_std:.2f}%")

    if not accuracies and not stds:
        print("No valid data found in summary.txt files.")

def compute_all_few_shot_results(dir):
    print("All few-shot performance")
    
    # 遍历 dir 下的所有子目录
    subdirs = listdir_nohidden(dir, sort=True)
    
    # 过滤并按自然顺序排序 shots 子目录
    shots_subdirs = [s for s in subdirs if s.startswith("shots_")]
    shots_subdirs.sort(key=lambda x: int(x.split("_")[1]))
    
    for subdir in shots_subdirs:
        # 提取 shot 数量，如 shots_1 -> 1
        shot_num = subdir.split("_")[1]
        
        summary_path = osp.join(dir, subdir, "summary.txt")
        if osp.exists(summary_path):
            with open(summary_path, "r") as f:
                content = f.read()
                # 解析 accuracy 和 std
                match_acc = re.search(r'\* accuracy: ([\d.]+)%', content)
                match_std = re.search(r'\* std: ([\d.]+)%', content)
                if match_acc and match_std:
                    acc = float(match_acc.group(1))
                    std = float(match_std.group(1))
                    print(f"{shot_num}-shot: * accuracy: {acc:.2f}% * std: {std:.2f}%")
                else:
                    print(f"{shot_num}-shot: No valid data found.")
        else:
            print(f"{shot_num}-shot: summary.txt not found.")


def main(args, end_signal):
    if args.hm:
        compute_harmonic_mean(args.directory)
        return
    if args.multi_ds:
        compute_few_shot_results(args.directory)
        return
    if args.all_res:
        compute_all_few_shot_results(args.directory)
        return
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.multi_exp:
        final_results = defaultdict(list)

        # for directory in listdir_nohidden(args.directory, sort=True): # >> 默认只能读取到一层目录
            # directory = osp.join(args.directory, directory)
        for directory in find_seed_parent_dirs(args.directory): # >> 读取到多层目录
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

        print("Average performance")
        for key, values in final_results.items():
            val = np.mean(values)
            print(f"* {key}: {val:.2f}%")

    else:
        parse_function(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, help="path to directory")
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )
    parser.add_argument("--hm", action="store_true", help="compute harmonic mean")
    parser.add_argument("--multi-ds", action="store_true", help="compute few-shot results")
    parser.add_argument("--all-res", action="store_true", help="compute all few-shot results")
    args = parser.parse_args()

    end_signal = "Finish training"
    if args.test_log:
        end_signal = "=> result"

    main(args, end_signal)