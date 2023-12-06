import argparse
import json
import os
import pprint

import json5
import jsonlines
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import Agent, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.tools.evaluate_agent import evaluate_agent
from transformers.trainer_utils import set_seed

data_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def is_callable(response, golden):
    return response["action"].strip().lower() == golden["action"].strip().lower()


def process_res(response):
    # parse response
    response += "\n"  # fix not-find bug
    thought = response[: response.find("Action:")].strip()
    action = response[
        response.find("Action:") + len("Action:") : response.find("Action Input:")
    ].strip()
    action_input = response[
        response.find("Action Input:")
        + len("Action Input:") : response.find("Observation:")
    ].strip()
    # TODO: This parsing result is incorrect if the response contains multiple Actions. To be fixed in the future.
    observation = response[
        response.find("Observation:") + len("Observation:") : response.rfind("Thought:")
    ].strip()
    thought_last = response[
        response.rfind("Thought:") + len("Thought:") : response.find("Final Answer:")
    ].strip()
    final_answer = response[
        response.find("Final Answer:") + len("Final Answer:") :
    ].strip()
    try:
        action_input = json.dumps(
            json5.loads(action_input), ensure_ascii=False, sort_keys=True
        )
    except:
        # print("JSON Load Error:", action_input)
        action_input = ""
    res_dict = {
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "observation": observation,
        "thought_last": thought_last,
        "final_answer": final_answer,
    }
    return res_dict


class _DummyTokenizer:
    def tokenize(self, text: str):
        return text.split()


def _get_tokenized_string(tokenizer, text_list):
    token_ids_list, tokenized_string_list = [], []
    for text in text_list:
        assert tokenizer is not None
        token_ids = tokenizer.encode(text)
        tokens_bytes = tokenizer.convert_ids_to_tokens(token_ids)
        tokens = [token.decode("utf-8", errors="replace") for token in tokens_bytes]
        tokenized_string = " ".join(tokens)
        token_ids_list.append(token_ids)
        tokenized_string_list.append(tokenized_string)
    return token_ids_list, tokenized_string_list


def eval_action(job):
    response = job["gen"][0]
    golden = job["response"]

    if "\nAction: " in response:
        response, golden = process_res(response), process_res(golden)
        if is_callable(response, golden):
            return True
    return False


def eval_action_input(job, tokenizer):
    response = job["gen"][0]
    golden = job["response"]
    response, golden = process_res(response), process_res(golden)
    query = job["prompt"]

    job = {}
    job["prompt"] = query
    job["gen"] = response["action_input"]
    job["response"] = golden["action_input"]

    job["_gen_tok"], job["_gen_tok_str"] = _get_tokenized_string(
        tokenizer, [response["action_input"]]
    )
    job["_reference_tok"], job["_reference_tok_str"] = _get_tokenized_string(
        tokenizer, [golden["action_input"]]
    )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], tokenizer=_DummyTokenizer()
    )
    score = scorer.score(job["_reference_tok_str"][0], job["_gen_tok_str"][0])

    rouge = score["rougeL"].fmeasure

    return rouge


class QWenAgent(Agent):
    """
    Agent that uses QWen model and tokenizer to generate code.

    Example:

    ```py
    agent = QWenAgent()
    agent.run("Draw me a picture of rivers and lakes.")
    ```
    """

    def __init__(
        self,
        chat_prompt_template=None,
        run_prompt_template=None,
        additional_tools=None,
        tokenizer=None,
        model=None,
    ):
        if tokenizer and model:
            self.tokenizer = tokenizer
            self.model = model
        else:
            checkpoint = "Qwen/Qwen-7B-Chat"
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, trust_remote_code=True
            )
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    checkpoint, device_map="auto", trust_remote_code=True
                )
                .cuda()
                .eval()
            )
            self.model.generation_config = GenerationConfig.from_pretrained(
                checkpoint, trust_remote_code=True
            )  # 可指定不同的生成长度、top_p等相关超参
            self.model.generation_config.do_sample = False  # greedy

        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_one(self, prompt, stop):
        # "Human:" 和 "Assistant:" 曾为通义千问的特殊保留字，需要替换为 "_HUMAN_:" 和 "_ASSISTANT_:"。这一问题将在未来版本修复。
        prompt = prompt.replace("Human:", "_HUMAN_:").replace(
            "Assistant:", "_ASSISTANT_:"
        )
        stop = [
            item.replace("Human:", "_HUMAN_:").replace("Assistant:", "_ASSISTANT_:")
            for item in stop
        ]

        result, _ = self.model.chat(self.tokenizer, prompt, history=None)
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]

        result = result.replace("_HUMAN_:", "Human:").replace(
            "_ASSISTANT_:", "Assistant:"
        )
        return result


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="auto",
        trust_remote_code=True,
        bf16=True,
        use_flash_attn=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False  # use greedy decoding
    return model, tokenizer


def load_jobs(filename):
    jobs = []
    with jsonlines.open(os.path.join(data_root_path, filename), mode="r") as reader:
        for job in reader:
            jobs.append(job)
    return jobs


def react_inference(filename, model, tokenizer):
    filename_cache = filename + ".cache"
    if os.path.exists(os.path.join(data_root_path, filename_cache)):
        jobs = load_jobs(filename=filename_cache)
        print("Loaded from", filename_cache)
    else:
        with open(os.path.join(data_root_path, filename_cache), "w") as f:
            jobs = load_jobs(filename=filename)
            print("Inference:", filename)
            for job in tqdm(jobs):
                response, history = model.chat(tokenizer, job["prompt"], history=None)
                job["gen"] = [response]
                f.writelines(json.dumps(job, ensure_ascii=False) + "\n")
        print(filename_cache, "is saved.")
    return jobs


def main(args):
    print("loading model weights")
    if args.checkpoint_path is not None:
        model, tokenizer = load_models_tokenizer(args)
    else:
        model, tokenizer = None, None
    print("model loaded")

    result = {}
    # eval react positive
    if args.eval_react_positive:
        print("eval react positive ...")
        acc_count = 0
        rouge_mean = 0
        jobs = react_inference(
            filename=args.eval_react_positive_filename, model=model, tokenizer=tokenizer
        )
        for job in jobs:
            if eval_action(job):
                acc_count += 1
            rouge = eval_action_input(job, tokenizer)
            rouge_mean += rouge / len(jobs)

        scores = {
            "action_right_rate": acc_count / len(jobs),
            "action_input_rouge": rouge_mean,
        }

        result.update({"react_positive": scores})

    # eval react negative
    if args.eval_react_negative:
        print("eval react negative ...")
        bad_count = 0
        jobs = react_inference(
            filename=args.eval_react_negative_filename, model=model, tokenizer=tokenizer
        )
        for job in jobs:
            if "\nAction: " in job["gen"][0]:
                bad_count += 1
        scores = {"bad_rate": bad_count / len(jobs)}
        result.update({"react_negative": scores})

    # eval hfagent
    if args.eval_hfagent:
        print("eval hfagent ...")
        agent = QWenAgent(model=model, tokenizer=tokenizer)
        scores = evaluate_agent(agent, verbose=False, return_errors=False)
        result.update({"hfagent": scores})

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen-7B-Chat",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument(
        "--eval-react-positive",
        action="store_true",
        default=False,
        help="Eval react positive.",
    )
    group.add_argument(
        "--eval-react-positive-filename",
        type=str,
        default="exam_plugin_v1_react_positive.jsonl",
        help="Eval react positive filename.",
    )
    group.add_argument(
        "--eval-react-negative",
        action="store_true",
        default=False,
        help="Eval react negative.",
    )
    group.add_argument(
        "--eval-react-negative-filename",
        type=str,
        default="exam_plugin_v1_react_negative.jsonl",
        help="Eval react negative filename.",
    )
    group.add_argument(
        "--eval-hfagent", action="store_true", default=False, help="Eval hfagent."
    )

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
