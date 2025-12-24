import torch
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import ORPOTrainer, ORPOConfig
from datasets import load_dataset
from huggingface_hub import HfApi, login

# =================================================================
# 0. 環境設定・認証・タイムスタンプ生成
# =================================================================
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run Timestamp: {run_timestamp}")

# 環境変数 HF_TOKEN を読み込み
hf_token = os.getenv("HF_TOKEN")
upload_repo_id = "gitpullpull/Introspective_Temperature_test"

if hf_token:
    login(token=hf_token)
    print(f"Hugging Face logged in. Target Repo: {upload_repo_id}")
else:
    print("WARNING: HF_TOKEN not found. Upload will fail.")

# =================================================================
# 1. ロギング設定
# =================================================================
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# =================================================================
# 2. モデル読み込み
# =================================================================
model_name = "Unsloth/qwen3-8b"
max_seq_length = 4096
output_dir = "./output_orpo"
os.makedirs(output_dir, exist_ok=True)

print(f"Loading model: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    bias = "none",
    lora_dropout = 0,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# =================================================================
# 3. データセット
# =================================================================
dataset_file = "clean_orpo5k_sanitized.jsonl" 
dataset = load_dataset("json", data_files=dataset_file, split="train")

def filter_bad_rows(example):
    try:
        row = example.get("data", example)
        if row is None: return False
        if not isinstance(row.get("user_prompt"), str): return False
        if row.get("chosen_response") == row.get("rejected_response"): return False
        return True
    except: return False

dataset = dataset.filter(filter_bad_rows)

system_instruction = """You allow dynamic control of generation temperature. Always use <TEMP_LOW>, <TEMP_MID>, or <TEMP_HIGH> tags to indicate your thought process and response style."""

def format_orpo_data(example):
    row = example.get("data", example)
    formatted_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": row["user_prompt"]}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    # ORPOもDPOと同じ形式(prompt, chosen, rejected)を使用
    formatted_chosen = f"<think>\n{row['thought_chosen']}\n</think>\n{row['chosen_response']}"
    formatted_rejected = f"<think>\n{row['thought_rejected']}\n</think>\n{row['rejected_response']}"
    return {
        "prompt": formatted_prompt,
        "chosen": formatted_chosen,
        "rejected": formatted_rejected,
    }

formatted_dataset = dataset.map(format_orpo_data)


# =================================================================
# 4. ORPOトレーニング設定
# =================================================================

orpo_config = ORPOConfig(
    output_dir = output_dir,
    remove_unused_columns = False,
    
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,

    # --- Lion設定 ---
    optim = "paged_lion_8bit",
    learning_rate = 4e-6, 
    weight_decay = 0.01,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    
    # --- ORPO固有設定 ---
    beta = 0.1,
    # --------------------
    
    num_train_epochs = 2,
    max_steps = -1,
    warmup_ratio = 0.1,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    lr_scheduler_type = "cosine",
    seed = 3407,
    report_to = "tensorboard",
    max_length = max_seq_length,
    max_prompt_length = max_seq_length // 2,
)

trainer = ORPOTrainer(
    model = model,
    args = orpo_config,
    train_dataset = formatted_dataset,
    tokenizer = tokenizer,
)

# =================================================================
# 5. 実行 & ローカル保存
# =================================================================
print(f"Lion 8bit ORPOトレーニングを開始します (LR: {orpo_config.learning_rate})...")
trainer.train()

print("ローカル保存中...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# ログ保存（ファイル名にタイムスタンプを付与）
log_filename = f"training_log_{run_timestamp}.csv"
log_file_path = os.path.join(output_dir, log_filename)
log_df = pd.DataFrame(trainer.state.log_history)
log_df.to_csv(log_file_path, index=False)

# 可視化保存（ファイル名にタイムスタンプを付与）
plot_filename = f"training_plots_{run_timestamp}.png"
plot_file_path = os.path.join(output_dir, plot_filename)
try:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    loss_data = log_df.dropna(subset=["loss"])
    if not loss_data.empty:
        plt.plot(loss_data["step"], loss_data["loss"], label="Total Loss", color="blue", alpha=0.7)
        plt.title("ORPO Total Loss")
        plt.grid(True, linestyle='--')
    
    plt.subplot(1, 2, 2)
    if "log_odds_ratio" in log_df.columns:
        odds_data = log_df.dropna(subset=["log_odds_ratio"])
        if not odds_data.empty:
            plt.plot(odds_data["step"], odds_data["log_odds_ratio"], label="Log Odds Ratio", color="purple")
            plt.title("Log Odds Ratio")
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(plot_file_path)
    print("Plots saved.")
except Exception as e:
    print(f"Plotting failed: {e}")

# =================================================================
# 6. Hugging Faceへのアップロード (フォルダ分け)
# =================================================================
if hf_token:
    # タイムスタンプごとのフォルダ名を定義 (例: run_20251218_183000)
    upload_subfolder = f"run_{run_timestamp}"
    
    print(f"Uploading to Hugging Face: {upload_repo_id}")
    print(f"Destination folder: {upload_subfolder}")
    
    try:
        api = HfApi()
        
        # output_dir の中身を丸ごと指定したサブフォルダへアップロード
        api.upload_folder(
            folder_path=output_dir,
            repo_id=upload_repo_id,
            repo_type="model",
            path_in_repo=upload_subfolder, # これにより競合を防ぐ
            token=hf_token
        )
            
        print("Upload Job Finished Successfully!")
        print(f"Saved to: https://huggingface.co/{upload_repo_id}/tree/main/{upload_subfolder}")
        
    except Exception as e:
        print(f"Upload failed: {e}")
else:
    print("Skip upload (HF_TOKEN not set).")

print("完了。")
