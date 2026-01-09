import os
import json
import argparse
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# リポジトリ設定
REPO_ID = "gitpullpull/Introspective_Temperature_benchmark"
REPO_TYPE = "dataset"

def parse_args():
    parser = argparse.ArgumentParser(description="HF Benchmark Merger & Analyzer")
    
    # デフォルト値をユーザー指定のものに変更
    parser.add_argument("--path", type=str, default="Qwen3-8B_base_NP_Templock_0shot", 
                        help="リポジトリ内の対象フォルダ (デフォルト: Qwen3-8B_base_NP_Templock_0shot)")
    
    parser.add_argument("--output", type=str, default="MMLU-pro",
                        help="ローカルの保存先フォルダ (デフォルト: MMLU-pro)")
    
    return parser.parse_args()

def analyze_and_save(experiment_id, merged_details, output_dir):
    """
    Pandasを使って詳細な分析を行い、CSVとして保存する
    """
    if not merged_details:
        print(f"[{experiment_id}] データが空のため分析をスキップします。")
        return

    df = pd.DataFrame(merged_details)

    # 必須カラムのガード
    if "category" not in df.columns: df["category"] = "unknown"
    if "total_tokens" not in df.columns: df["total_tokens"] = 0
    if "is_correct" not in df.columns: return

    # --- カテゴリー別分析 ---
    # 平均正答率、平均トークン数、問題数
    analysis = df.groupby("category").agg(
        accuracy=("is_correct", "mean"),
        avg_tokens=("total_tokens", "mean"),
        count=("is_correct", "count")
    ).reset_index()

    # ソート (問題数が多い順 -> 正答率が高い順)
    analysis = analysis.sort_values(by=["count", "accuracy"], ascending=[False, False])

    # --- 全体平均行 ---
    total_acc = df["is_correct"].mean()
    total_tokens = df["total_tokens"].mean()
    total_count = len(df)
    
    summary_row = pd.DataFrame([{
        "category": "ALL_TOTAL",
        "accuracy": total_acc,
        "avg_tokens": total_tokens,
        "count": total_count
    }])
    
    final_df = pd.concat([analysis, summary_row], ignore_index=True)

    # CSV保存
    csv_filename = os.path.join(output_dir, f"{experiment_id}_analysis.csv")
    final_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"  -> 分析CSV: {csv_filename}")
    print(f"     (Overall Acc: {total_acc:.4f}, Count: {total_count})")

def main():
    args = parse_args()
    
    target_path = args.path.strip("/")
    local_output_dir = Path(args.output)
    local_output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    api = HfApi(token=hf_token)
    
    print(f"ターゲットリポジトリパス: {REPO_ID}/{target_path}")
    print(f"ローカル保存先: {local_output_dir}/")

    # ファイル探索
    try:
        files_info = api.list_repo_tree(
            repo_id=REPO_ID, 
            repo_type=REPO_TYPE, 
            path_in_repo=target_path, 
            recursive=True
        )
    except Exception as e:
        print(f"エラー: パスが見つかりません。\n{e}")
        return

    # 実験IDの決定 (target_path のフォルダ名自体を実験IDとする)
    # 例: path="Qwen3-8B..." なら experiment_id="Qwen3-8B..."
    # もし指定パス内にさらにサブフォルダがある場合は、それらを別々の実験として扱う
    
    experiments = {}
    
    for file_info in files_info:
        path = file_info.path
        if path.endswith(".json"):
            # ファイル名自体にタイムスタンプ等が入っていてもOK
            # 親ディレクトリ単位でまとめる
            parent_dir = os.path.dirname(path) 
            
            # target_pathそのものが指定されている場合、その中にあるjsonは全て同一実験とみなす
            # target_path/subdir/file.json の場合は subdir を実験IDにする
            
            # target_path と file_path の関係を整理
            # 例: target="Qwen...", file="Qwen.../chunk.json" -> rel="chunk.json" -> dir=""
            rel_path = path.replace(f"{target_path}/", "")
            sub_dir = os.path.dirname(rel_path)
            
            if sub_dir == "":
                # target_path直下のファイル
                exp_id = os.path.basename(target_path)
            else:
                # サブディレクトリがある場合
                exp_id = sub_dir

            if exp_id not in experiments:
                experiments[exp_id] = []
            experiments[exp_id].append(path)

    print(f"検出された実験セット: {list(experiments.keys())}")

    # 各実験セットごとに結合処理
    for exp_id, file_paths in experiments.items():
        print(f"\n--- 処理中: {exp_id} ({len(file_paths)} files) ---")
        
        loaded_chunks = []
        
        for file_path in tqdm(file_paths, desc="Downloading"):
            local_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename=file_path, 
                repo_type=REPO_TYPE,
                token=hf_token
            )
            
            with open(local_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # ここが重要：ファイル名ではなく、JSON内の config.chunk_start を見てソート順を決める
                chunk_start = data.get("config", {}).get("chunk_start", 0)
                loaded_chunks.append({"start": chunk_start, "data": data})
        
        # chunk_start の数値順にソート (これでタイムスタンプ等のファイル名は無視して正しく並ぶ)
        loaded_chunks.sort(key=lambda x: x["start"])
        
        merged_output = {
            "config": None,
            "metrics": {"accuracy": 0.0, "correct": 0, "total": 0},
            "details": []
        }
        
        for item in loaded_chunks:
            chunk = item["data"]
            if merged_output["config"] is None:
                merged_output["config"] = chunk.get("config", {})
            
            if "metrics" in chunk:
                merged_output["metrics"]["correct"] += chunk["metrics"].get("correct", 0)
                merged_output["metrics"]["total"] += chunk["metrics"].get("total", 0)
            
            if "details" in chunk:
                merged_output["details"].extend(chunk.get("details", []))
        
        # Accuracy再計算
        total = merged_output["metrics"]["total"]
        correct = merged_output["metrics"]["correct"]
        merged_output["metrics"]["accuracy"] = correct / total if total > 0 else 0.0
        
        if merged_output["config"]:
            merged_output["config"]["chunk_start"] = 0
            merged_output["config"]["chunk_end"] = total
            merged_output["config"]["experiment_id"] = exp_id

        # 1. 結合JSON保存
        json_filename = os.path.join(local_output_dir, f"{exp_id}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(merged_output, f, indent=4, ensure_ascii=False)
        print(f"  -> 結合JSON: {json_filename}")

        # 2. 分析CSV保存
        analyze_and_save(exp_id, merged_output["details"], local_output_dir)

if __name__ == "__main__":
    main()
