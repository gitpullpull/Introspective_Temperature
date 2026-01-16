import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# リポジトリ設定
REPO_ID = "gitpullpull/Introspective_Temperature_benchmark"
REPO_TYPE = "dataset"

def parse_args():
    parser = argparse.ArgumentParser(description="GPQA Benchmark Analyzer")
    
    # デフォルトを 'gpqa-diamond-results/base-it-up' に変更
    parser.add_argument("--path", type=str, default="gpqa-diamond-results/base-it-up", 
                        help="リポジトリ内の対象フォルダ (デフォルト: gpqa-diamond-results/base-it-up)")
    
    parser.add_argument("--output", type=str, default="GPQA_Analysis",
                        help="ローカルの保存先フォルダ (デフォルト: GPQA_Analysis)")
    
    return parser.parse_args()

def analyze_and_save(experiment_id, merged_details, output_dir):
    """
    GPQA用の分析を行い、CSVとして保存する。
    問題ごとの正答率（Pass Rate）を集計する。
    """
    if not merged_details:
        print(f"[{experiment_id}] データが空のため分析をスキップします。")
        return

    df = pd.DataFrame(merged_details)

    if "is_correct" not in df.columns: return
    if "total_tokens" not in df.columns: df["total_tokens"] = np.nan
    if "question" not in df.columns: return

    # --- 問題ごとの集計 (Per-Question Analysis) ---
    # pass_rate: 5 seed中、何回正解したかの割合 (平均正答率)
    # tokens: Seed 46のトークン数 (他はNaNなのでmaxを取ればSeed 46の値になる)
    # ground_truth: 代表値を採用
    # ※ 質問文のシャッフル等で正解記号が変わる場合があるため、Questionでグルーピング
    
    # シャッフル対策: 質問文だけでまとめる（ground_truthは集計から外すか、代表値をとる）
    analysis = df.groupby(["question"]).agg(
        pass_rate=("is_correct", "mean"),
        tokens=("total_tokens", "max"),
        correct_count=("is_correct", "sum"),
        ground_truth=("ground_truth", "first")
    ).reset_index()

    # Pass Rateが高い順 -> トークン数が少ない順 にソート
    analysis = analysis.sort_values(by=["pass_rate", "tokens"], ascending=[False, True])

    # --- 全体サマリー行 ---
    # 全体の平均正答率 (Micro Average)
    total_acc = df["is_correct"].mean()
    # 全体の平均トークン数 (Seed 46が存在する行のみの平均)
    total_tokens = df["total_tokens"].mean()
    
    print(f"  -> 集計結果: Overall Acc = {total_acc:.2%}, Avg Tokens (Seed 46) = {total_tokens:.1f}")

    summary_row = pd.DataFrame([{
        "question": "== TOTAL SUMMARY ==",
        "ground_truth": "",
        "pass_rate": total_acc,
        "tokens": total_tokens,
        "correct_count": df["is_correct"].sum()
    }])
    
    final_df = pd.concat([analysis, summary_row], ignore_index=True)

    # CSV保存
    csv_filename = os.path.join(output_dir, f"{experiment_id}_analysis.csv")
    final_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"  -> 分析CSV保存完了: {csv_filename}")


def process_file_data(data):
    """
    JSONデータから必要な情報を抽出して正規化する
    """
    config = data.get("config", {})
    details = data.get("details", [])

    # Seedの特定: configにない場合は42とする
    seed = config.get("seed", 42)
    
    normalized_details = []
    for item in details:
        is_correct = item.get("correct", False)
        
        # トークン数の処理: Seed 46のみ取得し、それ以外はNaN
        token_count = np.nan
        if seed == 46:
            # 新フォーマット対応
            if "response_length" in item:
                token_count = item["response_length"]
            elif "total_tokens" in item:
                token_count = item["total_tokens"]
        
        new_item = {
            "question": item.get("question", "").strip(),
            "ground_truth": item.get("ground_truth", ""),
            "is_correct": is_correct,
            "seed": seed,
            "total_tokens": token_count
        }
        normalized_details.append(new_item)

    return normalized_details, seed

def main():
    args = parse_args()
    
    target_path = args.path.strip("/")
    local_output_dir = Path(args.output)
    local_output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    api = HfApi(token=hf_token)
    
    print(f"ターゲットパス: {REPO_ID}/{target_path}")
    print(f"保存先: {local_output_dir}/")

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

    # 実験IDごとにファイルをグループ化
    experiments = {}
    for file_info in files_info:
        path = file_info.path
        if path.endswith(".json"):
            # target_path からの相対パスを取得
            rel_path = path.replace(f"{target_path}/", "")
            sub_dir = os.path.dirname(rel_path)
            
            if sub_dir == "":
                # ターゲットフォルダ直下の場合、そのターゲットフォルダ名を実験IDとする
                exp_id = os.path.basename(target_path)
            else:
                # サブフォルダがある場合、そのサブフォルダ名を実験IDとする
                exp_id = sub_dir

            if exp_id not in experiments:
                experiments[exp_id] = []
            experiments[exp_id].append(path)

    print(f"検出された実験セット: {list(experiments.keys())}")

    # 各実験セットごとに処理
    for exp_id, file_paths in experiments.items():
        print(f"\n--- 処理中: {exp_id} ---")
        
        all_details = []
        seeds_processed = set()
        
        # ターゲットSeed (42〜46)
        target_seeds = [42, 43, 44, 45, 46]

        for file_path in tqdm(file_paths, desc="Loading Seeds"):
            # ローカルへのダウンロード
            local_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename=file_path, 
                repo_type=REPO_TYPE,
                token=hf_token
            )
            
            with open(local_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            details, seed = process_file_data(raw_data)
            
            if seed in target_seeds:
                all_details.extend(details)
                seeds_processed.add(seed)

        if not all_details:
            print(f"  -> 対象Seed(42-46)のデータなし。スキップします。")
            continue

        print(f"  -> 集計対象Seed: {sorted(list(seeds_processed))}")

        # 分析とCSV保存
        analyze_and_save(exp_id, all_details, local_output_dir)

if __name__ == "__main__":
    main()
