import argparse
import base64
import collections
import logging
import unicodedata
from pathlib import Path

import regex as re
from tqdm.contrib.logging import tqdm_logging_redirect

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def load_tiktoken_bpe(tiktoken_bpe_file: str) -> "dict[bytes, int]":
    contents = open(tiktoken_bpe_file, "rb").read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }


def dump_tiktoken_bpe(bpe_ranks: "dict[bytes, int]", tiktoken_bpe_file: str) -> None:
    with open(tiktoken_bpe_file, "wb") as f:
        for token, rank in sorted(bpe_ranks.items(), key=lambda x: x[1]):
            f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")


def bytes_to_pieces(the_bytes: bytes) -> "tuple[bytes]":
    return tuple(bytes([byte]) for byte in the_bytes)


def get_pairs(pieces: "tuple[bytes]") -> "set[tuple[bytes, bytes]]":
    return set(zip(pieces[:-1], pieces[1:]))


def get_stats(
    vocab: "dict[tuple[bytes, ...], int]",
) -> "dict[tuple[bytes, bytes], int]":
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_vocab(
    pair: "tuple[bytes, bytes]", vocab: "dict[tuple[bytes, ...], int]"
) -> "dict[tuple[bytes, ...], int]":
    return {apply_bp(pieces, pair): freq for pieces, freq in vocab.items()}


def apply_bp(
    pieces: "tuple[bytes, ...]", pair: "tuple[bytes, bytes]"
) -> "tuple[bytes, ...]":
    new_pieces = []
    first, second = pair
    i = 0
    while i < len(pieces):
        try:
            j = pieces.index(first, i)
            new_pieces.extend(pieces[i:j])
            i = j
        except:
            new_pieces.extend(pieces[i:])
            break

        if pieces[i] == first and i < len(pieces) - 1 and pieces[i + 1] == second:
            new_pieces.append(first + second)
            i += 2
        else:
            new_pieces.append(pieces[i])
            i += 1

    return tuple(new_pieces)


def bpe(word: bytes, merges: "dict[bytes,int]") -> "tuple[bytes, ...]":
    pieces = bytes_to_pieces(word)
    while len(pieces) > 1:
        pairs = get_pairs(pieces)
        pair = min(pairs, key=lambda pair: merges.get(pair[0] + pair[1], float("inf")))

        if pair[0] + pair[1] not in merges:
            break
        pieces = apply_bp(pieces, pair)
        # logger.debug(f"{[(p, p.decode('utf8', errors='replace')) for p in pieces]} {pair} {pieces}")
    return pieces


def best_pair_sort_key(
    item: "tuple[dict[bytes, bytes], int]",
) -> "tuple[int, int, int, str, bytes]":
    # prefer to use the highest frequency or shortest length or lexi sort, sligtly slower
    pair, freq = item
    pair_bytes = pair[0] + pair[1]
    pair_byte_length = len(pair_bytes)
    pair_str = pair_bytes.decode("utf-8", errors="replace")
    pair_str_length = len(pair_str)
    return -freq, pair_str_length, pair_byte_length, pair_str, pair_bytes


def learn_bpe(
    freqs: "dict[str,int]", existing: "dict[bytes, int]"
) -> "tuple[bytes, bytes]":
    vocab = {bpe(k.encode("utf-8"), existing): v for k, v in freqs.items()}
    vocab = {key: value for key, value in vocab.items() if len(key) > 1}
    new_merges = []
    with tqdm_logging_redirect() as bar:
        while vocab:
            pairs = get_stats(vocab)

            best, freq = min(pairs.items(), key=best_pair_sort_key)

            logger.debug(
                f'{best} ({(best[0]+best[1]).decode("utf-8", errors="replace")}) is selected as the next merge with freq {freq}'
            )
            new_merges.append(best)

            vocab = merge_vocab(best, vocab)
            vocab = {key: value for key, value in vocab.items() if len(key) > 1}
            bar.update()

    return new_merges


def load_expand_vocab(path: Path) -> "dict[str, int]":
    freqs = {}
    with open(path, "r", encoding="utf8") as fin:
        for line in fin:
            if not line.strip():
                continue
            word, freq = line.strip().split("\t")
            word = unicodedata.normalize("NFC", word)
            parts = re.findall(PAT_STR, word)
            if len(parts) > 1:
                logger.warning(
                    f"{word} would be pre-tokenized to {parts}, and thus cannot be added to vocabulary"
                )
                continue
            try:
                freq = int(freq)
            except ValueError as _:
                freq = 1
            if word in freqs:
                logger.warning(
                    f"{word} is repeated, the frequency is increased by this much"
                )
                freqs[word] += freq
            else:
                freqs[word] = freq
    return freqs


def make_new_merges_by_bpe(
    input_path: Path, output_path: Path, expand_path: Path, start_id: int
) -> None:
    mergeable_ranks = load_tiktoken_bpe(input_path)

    if not start_id or start_id == -1:
        start_id = len(mergeable_ranks)
    elif start_id < len(mergeable_ranks):
        logger.warning(
            f"start_id {start_id} is too small, existing merges will be overridden, DONOT DO THIS. changed to {len(mergeable_ranks)}"
        )
        start_id = len(mergeable_ranks)
    else:
        start_id = start_id

    expand_vocab_freqs = load_expand_vocab(expand_path)
    for word in list(expand_vocab_freqs):
        token = word.encode("utf-8")
        if token in mergeable_ranks:
            logger.warning(f"word {word} is already a token {token}, skipping")
            del expand_vocab_freqs[word]

    logger.info(f"number of existing merges: {len(mergeable_ranks)}")
    logger.info(f"number of words for expanding: {len(expand_vocab_freqs)}")

    new_merges = learn_bpe(expand_vocab_freqs, mergeable_ranks)
    logger.info(f"number of newly learned merges: {len(new_merges)}")

    extra_merges = {p[0] + p[1]: i for i, p in enumerate(new_merges, start=start_id)}

    dump_tiktoken_bpe(extra_merges, output_path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", type=str, help="Path for input tiktoken file")
    parser.add_argument(
        "output_path",
        type=str,
        help="Path for output tiktoken file, containing only the new merges",
    )
    parser.add_argument(
        "vocab_path",
        type=str,
        help="Path for words needed adding, each line is a word and its frequency separated by \\t",
    )
    # if the extended vocabulary is for fine-tuning, you better set those correctly (the default is for qwen.tiktoken)
    # if the extended vocabulary is for pretraining from the start, no need
    parser.add_argument(
        "--start_id",
        type=int,
        default=151851,
        help="The start id for new merges. For Qwen tokenizer, this should be 151851 (skipping the existing special tokens)",
    )

    args = parser.parse_args()

    make_new_merges_by_bpe(
        args.input_path, args.output_path, args.vocab_path, args.start_id
    )


if __name__ == "__main__":
    main()
