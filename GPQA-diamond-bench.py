import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import os
import json
import random
import argparse
import time
import csv
from datetime import datetime
from huggingface_hub import HfApi, login

# ==============================================================================
# 1. 設定
# ==============================================================================
MAX_SEQ_LENGTH = 16384
MAX_NEW_TOKENS = 8192
LOAD_IN_4BIT = False

# HF設定
hf_token = os.getenv("HF_TOKEN")
upload_repo_id = "gitpullpull/Introspective_Temperature_benchmark"

if hf_token:
    login(token=hf_token)
    print(f"Hugging Face logged in. Target Repo: {upload_repo_id}")
else:
    print("WARNING: HF_TOKEN not found. Upload will fail.")

# ==============================================================================
# 2. バッチ対応の手動生成ループ
# ==============================================================================
class BatchManualGenerator:
    """バッチ対応の手動生成クラス（KVキャッシュ使用）"""
    def __init__(self, model, tokenizer, temp_map, initial_temp=0.6):
        self.model = model
        self.tokenizer = tokenizer
        self.temp_map = temp_map
        self.initial_temp = initial_temp
        
    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask, max_new_tokens=8192, 
                       top_k=20, top_p=0.95, check_window=20):
        """バッチ単位での手動生成"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 各サンプルの温度を初期化
        temperatures = torch.full((batch_size,), self.initial_temp, device=device, dtype=torch.float)
        
        # 生成完了フラグ
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 生成IDの初期化
        generated_ids = input_ids.clone()
        
        # 初回推論でKVキャッシュを生成
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            use_cache=True
        )
        
        if hasattr(outputs, "logits"):
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
        else:
            next_token_logits = outputs[0][:, -1, :]
            past_key_values = outputs[1]
        
        # トークン生成ループ
        for step in range(max_new_tokens):
            # 温度更新チェック（最近のトークンから）
            if step > 0 and step % 5 == 0:  # 5トークンごとにチェック
                self._update_temperatures(generated_ids, temperatures, check_window)
            
            # Top-Kフィルタリング
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
            
            # 温度適用（バッチごとに異なる温度）
            safe_temps = torch.clamp(temperatures, min=0.1, max=2.0).unsqueeze(-1)
            scaled_logits = next_token_logits / safe_temps
            
            # Softmax
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Top-Pフィルタリング
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # サンプリング
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # 終了トークンチェック
            is_eos = (next_tokens.squeeze(-1) == self.tokenizer.eos_token_id)
            finished = finished | is_eos
            
            # 全て終了したら抜ける
            if finished.all():
                break
            
            # まだ生成中のサンプルのみ次のトークンを追加
            next_tokens = next_tokens.masked_fill(finished.unsqueeze(-1), self.tokenizer.pad_token_id)
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # attention_maskの更新
            attention_mask = torch.cat([
                attention_mask, 
                (~finished).unsqueeze(-1).long()
            ], dim=-1)
            
            # 次のステップの準備
            cache_length = past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                cache_length, 
                cache_length + 1, 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
            
            # 次の推論
            outputs = self.model(
                input_ids=next_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            if hasattr(outputs, "logits"):
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                next_token_logits = outputs[0][:, -1, :]
                past_key_values = outputs[1]
        
        return generated_ids
    
    def _update_temperatures(self, generated_ids, temperatures, check_window):
        """最近生成されたトークンから温度タグを検出して更新"""
        batch_size = generated_ids.shape[0]
        seq_len = generated_ids.shape[1]
        start_pos = max(0, seq_len - check_window)
        
        for i in range(batch_size):
            recent_ids = generated_ids[i, start_pos:]
            recent_text = self.tokenizer.decode(recent_ids, skip_special_tokens=True)
            
            # タグの検出
            for tag, temp in self.temp_map.items():
                if tag in recent_text:
                    temperatures[i] = temp
                    break

# ==============================================================================
# 3. 回答抽出ロジック
# ==============================================================================
def extract_answer(response):
    """Thinking対応の回答抽出"""
    # </think> 以降から回答を抽出
    if "</think>" in response:
        final_part = response.split("</think>")[-1].strip()
    else:
        final_part = response[-15000:] if len(response) > 2000 else response

    # Answer: X 形式（公式推奨）
    official_pattern = r"Answer:\s*([A-D])"
    matches = re.findall(official_pattern, final_part, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # その他のパターン
    patterns = [
        r'"answer"\s*:\s*"([A-D])"',
        r"(?:choice|option) is\s*:?\s*(?:\[|\()?([A-D])(?:\]|\))?",
        r"Therefore,?\s*(?:the answer is\s*)?([A-D])",
        r"\*\*([A-D])\*\*",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, final_part, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None

def parse_thinking_response(response):
    """Thinkingコンテンツと回答を分離"""
    if "</think>" in response:
        parts = response.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        content = parts[1].strip() if len(parts) > 1 else ""
    else:
        thinking = ""
        content = response.strip()
    return thinking, content

# ==============================================================================
# 4. データセット処理
# ==============================================================================
def load_gpqa_dataset():
    """GPQAデータセットの読み込み"""
    print("Loading GPQA dataset...")
    dataset_source = "unknown"
    
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        dataset_source = "idavidrein"
        print("Loaded Idavidrein/gpqa (gpqa_diamond)")
    except Exception as e:
        print(f"Failed to load Idavidrein: {e}")
        try:
            dataset = load_dataset("fingertap/GPQA-Diamond", split="test")
            dataset_source = "fingertap"
            print("Loaded fingertap/GPQA-Diamond")
        except Exception as e2:
            raise RuntimeError(f"Could not load any GPQA dataset: {e2}")
    
    return dataset, dataset_source

def format_item(item, tokenizer, use_custom_prompt=False):
    """データアイテムをプロンプト形式に変換"""
    # Idavidrein形式
    if "Correct Answer" in item and "Incorrect Answer 1" in item:
        q_text = item["Question"]
        correct_answer = item["Correct Answer"]
        incorrect_answers = [item[f"Incorrect Answer {i}"] for i in range(1, 4) 
                            if item.get(f"Incorrect Answer {i}")]

        choices = [correct_answer] + incorrect_answers
        random.shuffle(choices)

        try:
            gt_idx = choices.index(correct_answer)
            gt_label = chr(ord('A') + gt_idx)
        except:
            return None

        options_str = ""
        for i, opt in enumerate(choices):
            options_str += f"{chr(ord('A') + i)}) {opt}\n"

        final_user_content = f"{q_text}\n\n{options_str}"

    # fingertap形式
    elif "question" in item and "answer" in item:
        q_text = item["question"]
        gt_label = item["answer"]
        final_user_content = q_text
    else:
        return None

    # システムプロンプト（動的温度タグ付き）
    if use_custom_prompt:
        system_prompt = (
            "You allow dynamic control of generation temperature. Always use <TEMP_LOW>, <TEMP_MID>, or <TEMP_HIGH> tags to indicate your thought process and response style.\n\n"
            "Answer the following multiple choice question. "
            "The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
            "Think step by step before answering."
        )
    else:
        system_prompt = (
            "Answer the following multiple choice question. "
            "The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
            "Think step by step before answering."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_user_content}
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    return {
        "text": full_prompt,
        "ground_truth": gt_label,
        "question_raw": q_text
    }

def custom_collate_fn(batch):
    """カスタムコレート関数"""
    return {
        "text": [b["text"] for b in batch],
        "ground_truth": [b["ground_truth"] for b in batch],
        "question_raw": [b["question_raw"] for b in batch]
    }

# ==============================================================================
# 5. モデル読み込み
# ==============================================================================
def load_model(args):
    """モデルとトークナイザーの読み込み"""
    print(f"Loading Model: {args.model}")
    print(f"Load in 4bit: {LOAD_IN_4BIT}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # LoRAアダプターの読み込み
    if args.lora_path and os.path.exists(args.lora_path):
        print(f"Loading LoRA Adapter from: {args.lora_path}")
        try:
            model = PeftModel.from_pretrained(model, args.lora_path)
            print("LoRA loaded successfully.")
        except Exception as e:
            print(f"Failed to load LoRA: {e}. Proceeding with base model.")
    else:
        print("Proceeding without LoRA adapter.")
    
    FastLanguageModel.for_inference(model)
    
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        
    return model, tokenizer

# ==============================================================================
# 6. メイン評価ループ
# ==============================================================================
@torch.no_grad()
def eval_gpqa(model, tokenizer, dataset, args):
    """GPQA評価の実行"""
    BATCH_SIZE = args.batch_size
    print(f"Starting Evaluation. Batch Size: {BATCH_SIZE}")
    
    # データセット処理
    print("Preparing prompts...")
    processed_items = []
    for item in tqdm(dataset, desc="Formatting Prompts"):
        formatted = format_item(item, tokenizer, args.use_custom_prompt)
        if formatted is not None:
            processed_items.append(formatted)
    
    print(f"Total valid samples: {len(processed_items)}")
    
    # サンプル数制限
    if args.num_samples > 0:
        if len(processed_items) > args.num_samples:
            print(f"Randomly sampling {args.num_samples} items from {len(processed_items)} total items.")
            # シード固定（データ選択の再現性のため、引数のシードとは別に固定するか、引数を使うか）
            # ここでは引数のシードを使用している状態になります
            processed_items = random.sample(processed_items, args.num_samples)
        else:
            print(f"Requested {args.num_samples} items, but only {len(processed_items)} available. Using all.")
    
    dataloader = DataLoader(
        processed_items, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )
    
    # 温度マップの設定
    if args.use_Introspective_Temperature:
        TEMP_MAP = {
            "<TEMP_LOW>": 0.4,
            "<TEMPERATURE_LOW>": 0.4,
            "<TEMP_MID>": 0.6,
            "<TEMPERATURE_MID>": 0.6,
            "<TEMP_HIGH>": 1.0,
            "<TEMPERATURE_HIGH>": 1.0,
        }
    else:
        TEMP_MAP = {}
    
    # 手動生成器の初期化
    generator = BatchManualGenerator(
        model=model,
        tokenizer=tokenizer,
        temp_map=TEMP_MAP,
        initial_temp=0.6
    )

    all_results = []
    correct_count = 0
    total_count = 0

    for batch in tqdm(dataloader, desc="Inference"):
        texts = batch["text"]
        ground_truths = batch["ground_truth"]
        questions_raw = batch["question_raw"]
        
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to("cuda")
        
        # 手動生成を実行
        outputs = generator.generate_batch(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            top_k=20,
            top_p=0.95,
            check_window=20
        )

        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, response in enumerate(responses):
            thinking_content, final_content = parse_thinking_response(response)
            predicted = extract_answer(response)
            
            gt = ground_truths[i]
            is_correct = (predicted == gt)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            all_results.append({
                "question": questions_raw[i][:100],
                "predicted": predicted,
                "ground_truth": gt,
                "correct": is_correct,
                "thinking_length": len(thinking_content),
                "output_sample": response  # サンプルのみ保存
            })

    return all_results, correct_count, total_count

# ==============================================================================
# 7. メイン実行部
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="GPQA Evaluation with Manual Generation")
    parser.add_argument("--model", "-m", type=str, default="unsloth/Qwen3-8B",
                       help="Base model path")
    parser.add_argument("--lora_path", "-lp", type=str, nargs='?', const="./Introspective_Temperature_test/run_20251219_040313", default=None,
                       help="LoRA adapter path")
    parser.add_argument("--save_dir", "-s", type=str, default="./benchmark_results",
                       help="Results save directory")
    parser.add_argument("--batch_size", "-bs", type=int, default=3,
                       help="Batch size for inference")
    parser.add_argument("--num_samples", "-ns", type=int, default=-1,
                       help="Number of random samples to evaluate.")
    parser.add_argument("--use_custom_prompt", "-up", action='store_true',
                       help="Use custom prompt with temperature control tags.")
    parser.add_argument("--use_Introspective_Temperature", "-ui", action='store_true',
                       help="Use Introspective_Temperature control tags.")
    parser.add_argument("--seed", "-sd", type=int, default=42, 
                       help="Random seed for reproducibility")

    args = parser.parse_args()
    
    # ▼▼▼ 変更: シード設定をmain内で実行 ▼▼▼
    print(f"Setting random seed to {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # モデル読み込み
    model, tokenizer = load_model(args)
    
    # データセット読み込み
    dataset, dataset_source = load_gpqa_dataset()
    
    # 評価実行
    results, correct, total = eval_gpqa(model, tokenizer, dataset, args)
    
    # 精度計算
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\nTotal Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # 結果保存
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    model_name = args.model.split("/")[-1]
    lora_name = "base" if args.lora_path is None else args.lora_path.split("/")[-1]
    prompt_str = "UP" if args.use_custom_prompt else "NP"
    Introspective_Temperature_str = "IT" if args.use_Introspective_Temperature is True else "Templock"
    sample_str = f"n{args.num_samples}" if args.num_samples > 0 else "full"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ▼▼▼ 変更: ファイル名にシードを含める ▼▼▼
    filename = f"gpqa_{dataset_source}_{model_name}_{lora_name}_{prompt_str}_{Introspective_Temperature_str}_{sample_str}_seed{args.seed}_{timestamp}.json"
    save_path = os.path.join(args.save_dir, filename)
    
    output_data = {
        "config": {
            "model": args.model,
            "lora": args.lora_path,
            "dataset_source": dataset_source,
            "load_in_4bit": LOAD_IN_4BIT,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "use_custom_prompt": args.use_custom_prompt,
            "use_Introspective_Temperature": args.use_Introspective_Temperature,
            # ▼▼▼ 追加: Configにシードを記録 ▼▼▼
            "seed": args.seed
        },
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        },
        "details": results
    }
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"Results saved to {save_path}")

    # HFアップロード処理
    if hf_token:
        try:
            api = HfApi()
            # フォルダ分け用のIDを作成
            experiment_id = f"GPQA_{model_name}_{lora_name}_{prompt_str}_{Introspective_Temperature_str}"
            path_in_repo = f"{experiment_id}/{filename}"
            
            api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=path_in_repo,
                repo_id=upload_repo_id,
                repo_type="dataset"
            )
            print(f"Uploaded to HF: {upload_repo_id}/{path_in_repo}")
        except Exception as e:
            print(f"Failed to upload to Hugging Face: {e}")

if __name__ == "__main__":
    main()