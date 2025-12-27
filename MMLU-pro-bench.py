from unsloth import FastLanguageModel
import torch
import csv
import json
import argparse
import os
import random
import time
import re
import logging
import sys
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import PeftModel
from huggingface_hub import HfApi, login

# ==============================================================================
# 1. 設定とフラグ
# ==============================================================================
MAX_SEQ_LENGTH = 16384
MAX_NEW_TOKENS = 2048
LOAD_IN_4BIT = False

hf_token = os.getenv("HF_TOKEN")
upload_repo_id = "gitpullpull/Introspective_Temperature_benchmark"

if hf_token:
    login(token=hf_token)
    print(f"Hugging Face logged in. Target Repo: {upload_repo_id}")
else:
    print("WARNING: HF_TOKEN not found. Upload will fail.")

# 選択肢リスト (MMLU-Proは10択: A-J)
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# 再現性
random.seed(12345)
torch.manual_seed(12345)

# ==============================================================================
# 2. 改善版：バッチ対応の手動生成ループ
# ==============================================================================
class BatchManualGenerator:
    """バッチ対応の手動生成クラス（KVキャッシュ使用）"""
    def __init__(self, model, tokenizer, temp_map, initial_temp=0.6):
        self.model = model
        self.tokenizer = tokenizer
        self.temp_map = temp_map
        self.initial_temp = initial_temp
        
    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask, max_new_tokens=2048, 
                       top_k=20, top_p=0.95, check_window=15):
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
            # 安全な温度範囲を確保
            safe_temps = torch.clamp(temperatures, min=0.1, max=2.0).unsqueeze(-1)
            scaled_logits = next_token_logits / safe_temps
            
            # Softmax
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Top-Pフィルタリング
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Top-P閾値を超えるトークンをマスク
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                # 元のインデックスに戻してマスク
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
# 3. 公式準拠: 回答抽出ロジック (Regex) - 10択(A-J)対応版
# ==============================================================================
def extract_answer(text):
    """公式コードのロジック: パターン1"""
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)

def extract_again(text):
    """公式コードのロジック: パターン2"""
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    """公式コードのロジック: パターン3 (最終手段)"""
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

# ==============================================================================
# 4. データ処理とプロンプト作成
# ==============================================================================
def preprocess_dataset_to_list(dataset):
    """Datasetオブジェクトを扱いやすい辞書のリストに変換"""
    res = []
    for each in dataset:
        options = [opt for opt in each["options"] if opt != "N/A"]
        if len(options) > 10:
            options = options[:10]
            
        item = {
            "question": each["question"],
            "options": options,
            "answer": each["answer"],
            "answer_index": each["answer_index"],
            "cot_content": each.get("cot_content", ""),
            "category": each["category"],
            "src": each.get("src", "")
        }
        res.append(item)
    return res

def format_cot_example(example, including_answer=True):
    """公式フォーマット準拠"""
    prompt = "Question:\n" + example["question"] + "\nOptions:\n"
    for i, opt in enumerate(example["options"]):
        if i < len(CHOICES):
            prompt += "{}. {}\n".format(CHOICES[i], opt)
    
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt

def create_prompt_text(curr, val_df, k, use_custom_prompt=False):
    """プロンプト作成"""
    if use_custom_prompt:
        prompt = "You can control the output style using <TEMP_LOW>, <TEMP_MID>, or <TEMP_HIGH> tags.\n\n"
        prompt += "The following are multiple choice questions (with answers) about {}.\n\n".format(curr["category"].replace("_", " "))
    else:
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(curr["category"].replace("_", " "))
    
    # Few-shot examples
    if k > 0:
        subject = curr["category"]
        relevant_examples = [x for x in val_df if x["category"] == subject]
        shots = relevant_examples[:k]
        for example in shots:
            prompt += format_cot_example(example, including_answer=True)
    
    # Target Question
    prompt += format_cot_example(curr, including_answer=False)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    return messages

def custom_collate_fn(batch):
    return {
        "text": [b["text"] for b in batch],
        "original_item": [b["original_item"] for b in batch]
    }

# ==============================================================================
# 5. モデル読み込み
# ==============================================================================
def load_model(args):
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
# 6. メイン評価ループ（改善版）
# ==============================================================================
@torch.no_grad()
def eval_mmlu_pro(model, tokenizer, test_data, val_data, args):
    BATCH_SIZE = args.batch_size
    print(f"Starting Evaluation. Batch Size: {BATCH_SIZE}")
    
    processed_items = []
    print("Preparing prompts...")
    k = args.nshot
    
    for item in tqdm(test_data, desc="Formatting Prompts"):
        messages = create_prompt_text(item, val_data, k, args.use_custom_prompt)
        
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=True
        )
        
        processed_items.append({
            "text": full_text,
            "original_item": item
        })

    dataloader = DataLoader(processed_items, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    # 温度マップの設定
    if args.use_Introspective_Temperature:
        TEMP_MAP = {
            "<TEMP_LOW>": 0.4,
            "<TEMPERATURE_LOW>": 0.4,
            "<TEMP_MID>": 0.6,
            "<TEMPERATURE_MID>": 0.6,
            "<TEMP_HIGH>": 0.8,
            "<TEMPERATURE_HIGH>": 0.8,
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
        original_items = batch["original_item"]
        
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
            item = original_items[i]
            category = item["category"]
            question_id = item["question"][:50]
            gt = item["answer"]
            
            pred = extract_answer(response)
            
            if pred is None:
                is_correct = False
            else:
                is_correct = (pred == gt)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            all_results.append({
                "category": category,
                "question_head": question_id,
                "model_output": response,
                "predicted": pred,
                "ground_truth": gt,
                "is_correct": is_correct
            })

    return all_results, correct_count, total_count

# ==============================================================================
# 7. メイン実行部
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nshot", "-k", type=int, default=0)
    parser.add_argument("--batch_size", "-bs", type=int, default=3)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="./benchmark_results")
    parser.add_argument("--global_record_file", "-grf", type=str, default="eval_record.csv")
    parser.add_argument("--model", "-m", type=str, default="unsloth/Qwen3-8B")
    parser.add_argument("--lora_path", "-lp", type=str, nargs='?', const="./Introspective_Temperature_test/run_20251224_170022", default=None)
    parser.add_argument("--num_samples", "-ns", type=int, default=-1, help="Number of random samples to evaluate.")
    parser.add_argument("--use_custom_prompt", "-up", action='store_true', help="Use custom prompt with temperature control tags.")
    parser.add_argument("--use_Introspective_Temperature", "-ui", action='store_true', help="Use Introspective_Temperature control tags.")

    args = parser.parse_args()
    
    if args.nshot > 0:
        print(f"Mode: {args.nshot}-shot Evaluation")
    else:
        print("Mode: 0-shot Evaluation")

    model, tokenizer = load_model(args)
    
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    
    test_df = preprocess_dataset_to_list(dataset["test"])
    val_df = preprocess_dataset_to_list(dataset["validation"])
    
    if args.selected_subjects != "all":
        targets = args.selected_subjects.split(",")
        test_df = [x for x in test_df if any(t in x["category"] for t in targets)]
        print(f"Filtered to {len(test_df)} questions based on subjects.")

    if args.num_samples > 0:
        if len(test_df) > args.num_samples:
            print(f"Randomly sampling {args.num_samples} items from {len(test_df)} total items.")
            random.seed(12345)
            test_df = random.sample(test_df, args.num_samples)
        else:
            print(f"Requested {args.num_samples} items, but only {len(test_df)} available. Using all.")

    results, corr, total = eval_mmlu_pro(model, tokenizer, test_df, val_df, args)
    
    accuracy = corr / total if total > 0 else 0
    print(f"\nTotal Accuracy: {accuracy:.4f} ({corr}/{total})")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    model_name = args.model.split("/")[-1]
    lora_name = "base" if args.lora_path is None else args.lora_path.split("/")[-1]
    prompt_str ="UP" if args.use_custom_prompt is True else "NP"
    Introspective_Temperature_str = "IT" if args.use_Introspective_Temperature is True else "Templock"
    shot_str = "5shot" if args.nshot > 0 else "0shot"
    sample_str = f"n{args.num_samples}" if args.num_samples > 0 else "full"
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    filename = f"{model_name}_{lora_name}_{prompt_str}_{Introspective_Temperature_str}_{shot_str}_{sample_str}_{timestamp}.json"
    save_path = os.path.join(args.save_dir, filename)
    
    output_data = {
        "config": {
            "model": args.model,
            "lora": args.lora_path,
            "shots": args.nshot,
            "load_in_4bit": LOAD_IN_4BIT,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "use_custom_prompt": args.use_custom_prompt,
            "use_Introspective_Temperature": args.use_Introspective_Temperature
            },
        "metrics": {
            "accuracy": accuracy,
            "correct": corr,
            "total": total
        },
        "details": results
    }
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"Results saved to {save_path}")

    if hf_token:
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=filename,
                repo_id=upload_repo_id,
                repo_type="dataset"
            )
            print(f"Successfully uploaded {filename} to {upload_repo_id}")
        except Exception as e:
            print(f"Failed to upload to Hugging Face: {e}")

    if args.global_record_file:
        csv_filename = os.path.basename(args.global_record_file)
        csv_path = os.path.join(args.save_dir, csv_filename)
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, "CoT", args.selected_subjects, shot_str, f"{sample_str}_{timestamp}", accuracy])
        print(f"Record appended to {csv_path}")



if __name__ == "__main__":
    main()
