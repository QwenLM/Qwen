import os
import pandas as pd
import numpy as np
import argparse
import datasets
import torch
from collections import defaultdict

from typing import List
from tqdm import tqdm
from transformers.trainer_utils import set_seed


"""
wget https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip
mkdir data/cmmlu
mv cmmlu_v1_0_1.zip data/cmmlu
cd data/cmmlu; unzip cmmlu_v1_0_1.zip
cd ../../
python evaluate_cmmlu.py -d data/cmmlu/
"""


def load_models_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True
    )
    return model, tokenizer


def format_example(line, include_answer=True):
    example = "问题：" + line["Question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\n答案：" + line["Answer"] + "\n\n"
    else:
        example += "\n答案："
    return example


def generate_few_shot_prompt(k, subject, dev_df):
    prompt = ""
    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True,
        )
    return prompt


def get_logits(tokenizer, model, inputs: List[str]):
    input_ids = tokenizer(inputs, padding='longest')["input_ids"]
    input_ids = torch.tensor(input_ids, device=model.device)
    tokens = {"input_ids": input_ids}
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    outputs = model(input_ids, attention_mask=attention_mask)["logits"]
    logits = outputs[:, -1, :]
    log_probs = torch.nn.functional.softmax(logits, dim=-1)
    return log_probs, {"tokens": tokens}


@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    k=5,
    dev_df=None,
    few_shot=False,
    save_result_dir=None,
    batch_size=1,
    **kwargs,
):
    result = []
    score = []

    few_shot_prompt = (
        generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else []
    )
    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
    if args.debug:
        print(f"few_shot_prompt: {few_shot_prompt}")

    choices_ids = torch.tensor(
        tokenizer("A")["input_ids"] + tokenizer("B")["input_ids"] +
        tokenizer("C")["input_ids"] + tokenizer("D")["input_ids"]
    ).unsqueeze(0).to(model.device)

    idx_list = list(range(0, len(test_df), batch_size))
    for i in tqdm(idx_list):
        full_prompt_list = []
        answer_list = []
        for row in test_df.iloc[i:i+batch_size].to_dict(orient='records'):
            question = format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            full_prompt_list.append(full_prompt)
            if 'Answer' in row:
                answer_list.append(row['Answer'])

        logits, input_info = get_logits(tokenizer, model, full_prompt_list)
        softval = logits.gather(1, choices_ids.expand(logits.size(0), -1)).softmax(1)
        if softval.dtype in {torch.bfloat16, torch.float16}:
            softval = softval.to(dtype=torch.float32)
        probs = softval.detach().cpu().numpy()

        for i in range(len(probs)):
            for j, choice in enumerate(choices):
                all_probs[f"prob_{choice}"].append(probs[i][j])
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs[i])]

            if answer_list != []:
                correct = 1 if pred == answer_list[i] else 0
                score.append(correct)
                if args.debug:
                    print(f'{question} pred: {pred} ref: {answer_list[i]}')
            result.append(pred)

    if score:
        correct_ratio = 100 * sum(score) / len(score)
        if args.debug:
            print(subject_name, correct_ratio)
    else:
        correct_ratio = 0
    if save_result_dir:
        test_df["model_output"] = result
        for i, choice in enumerate(choices):
            test_df[f"prob_{choice}"] = all_probs[f"prob_{choice}"]
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    return correct_ratio


def cal_cmmlu(res):
    print("\n\n\n")
    res = {k.split("-")[-1]: float(v) for k, v in res.items()}
    for k, v in TASK_NAME_MAPPING.items():
        avg_acc = np.mean(list(map(lambda x: res[x], v)))
        print(f"{k} acc: {avg_acc:.2f}")
    avg_all_acc = np.mean(list(res.values()))
    print(f"AVERAGE acc: {avg_all_acc:.2f}")


subcategories = {
    "agronomy": ["other"],
    "anatomy": ["biology"],
    "ancient_chinese": ["linguistics", "china specific"],
    "arts": ["arts"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "chinese_civil_service_exam": ["politics", "china specific"],
    "chinese_driving_rule": ["other", "china specific"],
    "chinese_food_culture": ["culture", "china specific"],
    "chinese_foreign_policy": ["politics", "china specific"],
    "chinese_history": ["history", "china specific"],
    "chinese_literature": ["literature", "china specific"],
    "chinese_teacher_qualification": ["education", "china specific"],
    "college_actuarial_science": ["math"],
    "college_education": ["education"],
    "college_engineering_hydrology": ["engineering"],
    "college_law": ["law"],
    "college_mathematics": ["math"],
    "college_medical_statistics": ["statistics"],
    "clinical_knowledge": ["other"],
    "college_medicine": ["other"],
    "computer_science": ["computer science"],
    "computer_security": ["other"],
    "conceptual_physics": ["physics"],
    "construction_project_management": ["other", "china specific"],
    "economics": ["economics"],
    "education": ["education"],
    "elementary_chinese": ["linguistics", "china specific"],
    "elementary_commonsense": ["other", "china specific"],
    "elementary_information_and_technology": ["other"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "ethnology": ["culture", "china specific"],
    "food_science": ["other"],
    "genetics": ["biology"],
    "global_facts": ["global"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_geography": ["geography"],
    "high_school_mathematics": ["math"],
    "high_school_physics": ["physics"],
    "high_school_politics": ["politics", "china specific"],
    "human_sexuality": ["other"],
    "international_law": ["law"],
    "journalism": ["sociology"],
    "jurisprudence": ["law"],
    "legal_and_moral_basis": ["other"],
    "logical": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "marxist_theory": ["philosophy"],
    "modern_chinese": ["linguistics", "china specific"],
    "nutrition": ["other"],
    "philosophy": ["philosophy"],
    "professional_accounting": ["business"],
    "professional_law": ["law"],
    "professional_medicine": ["other"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_study": ["politics"],
    "sociology": ["culture"],
    "sports_science": ["other"],
    "traditional_chinese_medicine": ["other", "china specific"],
    "virology": ["biology"],
    "world_history": ["history"],
    "world_religions": ["global"],
}

categories = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
        "statistics",
    ],
    "Humanities": ["history", "philosophy", "law", "arts", "literature", "global"],
    "Social Science": [
        "linguistics",
        "business",
        "politics",
        "culture",
        "economics",
        "geography",
        "psychology",
        "education",
        "sociology",
    ],
    "Other": ["other"],
    "China specific": ["china specific"],
}

TASK_NAME_MAPPING = defaultdict(list)
for k, v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                TASK_NAME_MAPPING[k].append(subject)


choices = ["A", "B", "C", "D"]


def main(args):
    model, tokenizer = load_models_tokenizer(args)

    test_result = {}
    for subject_name in tqdm(subcategories.keys()):
        dev_file_path = os.path.join(args.eval_data_path, "dev", f"{subject_name}.csv")
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}.csv"
        )
        dev_df = pd.read_csv(dev_file_path)
        test_df = pd.read_csv(test_file_path)

        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            dev_df=dev_df,
            test_df=test_df,
            k=5,
            few_shot=True,
            save_result_dir=f"outs/cmmlu_eval_result",
            batch_size=args.batch_size
        )
        test_result[subject_name] = score
    cal_cmmlu(test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen-7B",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")

    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument(
        "-d", "--eval_data_path", type=str, required=True, help="Path to eval data"
    )
    group.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--debug", action="store_true", default=False, help="Print infos."
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
