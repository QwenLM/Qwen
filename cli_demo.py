# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple command-line interactive chat demo."""

import argparse
import os
import platform
import shutil
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed

DEFAULT_CKPT_PATH = 'Qwen/Qwen-7B-Chat'

_WELCOME_MSG = '''\
Welcome to use Qwen-Chat model, type text to start chat, type :h to show command help.
(欢迎使用 Qwen-Chat 模型，输入内容即可进行对话，:h 显示命令帮助。)

Note: This demo is governed by the original license of Qwen.
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc.
(注：本演示受Qwen的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)
'''
_HELP_MSG = '''\
Commands:
    :help / :h          Show this help message              显示帮助信息
    :exit / :quit / :q  Exit the demo                       退出Demo
    :clear / :cl        Clear screen                        清屏
    :clear-his / :clh   Clear history                       清除对话历史
    :history / :his     Show history                        显示对话历史
    :seed               Show current random seed            显示当前随机种子
    :seed <N>           Set random seed to <N>              设置随机种子
    :conf               Show current generation config      显示生成配置
    :conf <key>=<value> Change generation config            修改生成配置
    :reset-conf         Reset generation config             重置生成配置
'''


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config


def _clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def _print_history(history):
    terminal_width = shutil.get_terminal_size()[0]
    print(f'History ({len(history)})'.center(terminal_width, '='))
    for index, (query, response) in enumerate(history):
        print(f'User[{index}]: {query}')
        print(f'QWen[{index}]: {response}')
    print('=' * terminal_width)


def _get_input() -> str:
    while True:
        try:
            message = input('User> ').strip()
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print('[ERROR] Query is empty')


def main():
    parser = argparse.ArgumentParser(
        description='QWen-Chat command-line interactive chat demo.')
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    args = parser.parse_args()

    history, response = [], ''

    model, tokenizer, config = _load_model_tokenizer(args)
    orig_gen_config = deepcopy(model.generation_config)

    _clear_screen()
    print(_WELCOME_MSG)

    seed = args.seed

    while True:
        query = _get_input()

        # Process commands.
        if query.startswith(':'):
            command_words = query[1:].strip().split()
            if not command_words:
                command = ''
            else:
                command = command_words[0]

            if command in ['exit', 'quit', 'q']:
                break
            elif command in ['clear', 'cl']:
                _clear_screen()
                print(_WELCOME_MSG)
                continue
            elif command in ['clear-history', 'clh']:
                print(f'[INFO] All {len(history)} history cleared')
                history.clear()
                continue
            elif command in ['help', 'h']:
                print(_HELP_MSG)
                continue
            elif command in ['history', 'his']:
                _print_history(history)
                continue
            elif command in ['seed']:
                if len(command_words) == 1:
                    print(f'[INFO] Current random seed: {seed}')
                    continue
                else:
                    new_seed_s = command_words[1]
                    try:
                        new_seed = int(new_seed_s)
                    except ValueError:
                        print(f'[WARNING] Fail to change random seed: {new_seed_s!r} is not a valid number')
                    else:
                        print(f'[INFO] Random seed changed to {new_seed}')
                        seed = new_seed
                    continue
            elif command in ['conf']:
                if len(command_words) == 1:
                    print(model.generation_config)
                else:
                    for key_value_pairs_str in command_words[1:]:
                        eq_idx = key_value_pairs_str.find('=')
                        if eq_idx == -1:
                            print('[WARNING] format: <key>=<value>')
                            continue
                        conf_key, conf_value_str = key_value_pairs_str[:eq_idx], key_value_pairs_str[eq_idx + 1:]
                        try:
                            conf_value = eval(conf_value_str)
                        except Exception as e:
                            print(e)
                            continue
                        else:
                            print(f'[INFO] Change config: model.generation_config.{conf_key} = {conf_value}')
                            setattr(model.generation_config, conf_key, conf_value)
                continue
            elif command in ['reset-conf']:
                print('[INFO] Reset generation config')
                model.generation_config = deepcopy(orig_gen_config)
                print(model.generation_config)
                continue
            else:
                # As normal query.
                pass

        # Run chat.
        set_seed(seed)
        try:
            for response in model.chat_stream(tokenizer, query, history=history, generation_config=config):
                _clear_screen()
                print(f"\nUser: {query}")
                print(f"\nQwen-Chat: {response}")
        except KeyboardInterrupt:
            print('[WARNING] Generation interrupted')
            continue

        history.append((query, response))


if __name__ == "__main__":
    main()
