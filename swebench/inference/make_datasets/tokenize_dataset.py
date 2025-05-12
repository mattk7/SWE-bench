#!/usr/bin/env python3

"""Provided a source (raw) directory and the final (eval) directory, create a training split by removing all instances that are in the final directory from the source directory."""

import os
import logging
from argparse import ArgumentParser
from pathlib import Path

import tiktoken
from datasets import disable_caching, load_from_disk, load_dataset
from tqdm.auto import tqdm
from transformers import LlamaTokenizer, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.warning("Disabling caching")
disable_caching()


def cl100k(text, tokenizer):
    return tokenizer.encode(text, disallowed_special=())


def llama(text, tokenizer):
    return tokenizer(text, add_special_tokens=False, return_attention_mask=False)[
        "input_ids"
    ]

def qwen(text, tokenizer):
    return tokenizer(text, add_special_tokens=False, return_attention_mask=False)[
        "input_ids"
    ]
"""[31mSignature:[39m     
tokenizer(
    text: Union[str, List[str], List[List[str]], NoneType] = [38;5;28;01mNone[39;00m,
    text_pair: Union[str, List[str], List[List[str]], NoneType] = [38;5;28;01mNone[39;00m,
    text_target: Union[str, List[str], List[List[str]], NoneType] = [38;5;28;01mNone[39;00m,
    text_pair_target: Union[str, List[str], List[List[str]], NoneType] = [38;5;28;01mNone[39;00m,
    add_special_tokens: bool = [38;5;28;01mTrue[39;00m,
    padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = [38;5;28;01mFalse[39;00m,
    truncation: Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = [38;5;28;01mNone[39;00m,
    max_length: Optional[int] = [38;5;28;01mNone[39;00m,
    stride: int = [32m0[39m,
    is_split_into_words: bool = [38;5;28;01mFalse[39;00m,
    pad_to_multiple_of: Optional[int] = [38;5;28;01mNone[39;00m,
    padding_side: Optional[str] = [38;5;28;01mNone[39;00m,
    return_tensors: Union[str, transformers.utils.generic.TensorType, NoneType] = [38;5;28;01mNone[39;00m,
    return_token_type_ids: Optional[bool] = [38;5;28;01mNone[39;00m,
    return_attention_mask: Optional[bool] = [38;5;28;01mNone[39;00m,
    return_overflowing_tokens: bool = [38;5;28;01mFalse[39;00m,
    return_special_tokens_mask: bool = [38;5;28;01mFalse[39;00m,
    return_offsets_mapping: bool = [38;5;28;01mFalse[39;00m,
    return_length: bool = [38;5;28;01mFalse[39;00m,
    verbose: bool = [38;5;28;01mTrue[39;00m,
    **kwargs,
) -> transformers.tokenization_utils_base.BatchEncoding
[31mType:[39m           Qwen2TokenizerFast
[31mString form:[39m   
Qwen2TokenizerFast(name_or_path='Qwen/Qwen3-0.6B', vocab_size=151643, model_max_length=131072, is <...> ("</think>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
           }
           )
[31mLength:[39m         151669
[31mFile:[39m           ~/SWE-bench/lib/python3.12/site-packages/transformers/models/qwen2/tokenization_qwen2_fast.py
[31mDocstring:[39m     
Construct a "fast" Qwen2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
Byte-Pair-Encoding.

Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
be encoded differently whether it is at the beginning of the sentence (without space) or not:

```python
>>> from transformers import Qwen2TokenizerFast

>>> tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
>>> tokenizer("Hello world")["input_ids"]
[9707, 1879]

>>> tokenizer(" Hello world")["input_ids"]
[21927, 1879]
```
This is expected.

This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

Args:
    vocab_file (`str`, *optional*):
        Path to the vocabulary file.
    merges_file (`str`, *optional*):
        Path to the merges file.
    tokenizer_file (`str`, *optional*):
        Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
        contains everything needed to load the tokenizer.
    unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
        The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
        token instead. Not applicable to this tokenizer.
    bos_token (`str`, *optional*):
        The beginning of sequence token. Not applicable for this tokenizer.
    eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
        The end of sequence token.
    pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
        The token used for padding, for example when batching sequences of different lengths.
[31mCall docstring:[39m
Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

Args:
    text (`str`, `List[str]`, `List[List[str]]`, *optional*):
        The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
        (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
        `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
    text_pair (`str`, `List[str]`, `List[List[str]]`, *optional*):
        The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
        (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
        `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
    text_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
        The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
        list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
        you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
    text_pair_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
        The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
        list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
        you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

    add_special_tokens (`bool`, *optional*, defaults to `True`):
        Whether or not to add special tokens when encoding the sequences. This will use the underlying
        `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
        automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
        automatically.
    padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
        Activates and controls padding. Accepts the following values:

        - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
          sequence if provided).
        - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
          acceptable input length for the model if that argument is not provided.
        - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
          lengths).
    truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
        Activates and controls truncation. Accepts the following values:

        - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
          to the maximum acceptable input length for the model if that argument is not provided. This will
          truncate token by token, removing a token from the longest sequence in the pair if a pair of
          sequences (or a batch of pairs) is provided.
        - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
          maximum acceptable input length for the model if that argument is not provided. This will only
          truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
        - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
          maximum acceptable input length for the model if that argument is not provided. This will only
          truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
        - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
          greater than the model maximum admissible input size).
    max_length (`int`, *optional*):
        Controls the maximum length to use by one of the truncation/padding parameters.

        If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
        is required by one of the truncation/padding parameters. If the model has no specific maximum input
        length (like XLNet) truncation/padding to a maximum length will be deactivated.
    stride (`int`, *optional*, defaults to 0):
        If set to a number along with `max_length`, the overflowing tokens returned when
        `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
        returned to provide some overlap between truncated and overflowing sequences. The value of this
        argument defines the number of overlapping tokens.
    is_split_into_words (`bool`, *optional*, defaults to `False`):
        Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
        tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
        which it will tokenize. This is useful for NER or token classification.
    pad_to_multiple_of (`int`, *optional*):
        If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
        This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
        `>= 7.5` (Volta).
    padding_side (`str`, *optional*):
        The side on which the model should have padding applied. Should be selected between ['right', 'left'].
        Default value is picked from the class attribute of the same name.
    return_tensors (`str` or [`~utils.TensorType`], *optional*):
        If set, will return tensors instead of list of python integers. Acceptable values are:

        - `'tf'`: Return TensorFlow `tf.constant` objects.
        - `'pt'`: Return PyTorch `torch.Tensor` objects.
        - `'np'`: Return Numpy `np.ndarray` objects.

    return_token_type_ids (`bool`, *optional*):
        Whether to return token type IDs. If left to the default, will return the token type IDs according to
        the specific tokenizer's default, defined by the `return_outputs` attribute.

        [What are token type IDs?](../glossary#token-type-ids)
    return_attention_mask (`bool`, *optional*):
        Whether to return the attention mask. If left to the default, will return the attention mask according
        to the specific tokenizer's default, defined by the `return_outputs` attribute.

        [What are attention masks?](../glossary#attention-mask)
    return_overflowing_tokens (`bool`, *optional*, defaults to `False`):
        Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
        of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
        of returning overflowing tokens.
    return_special_tokens_mask (`bool`, *optional*, defaults to `False`):
        Whether or not to return special tokens mask information.
    return_offsets_mapping (`bool`, *optional*, defaults to `False`):
        Whether or not to return `(char_start, char_end)` for each token.

        This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using
        Python's tokenizer, this method will raise `NotImplementedError`.
    return_length  (`bool`, *optional*, defaults to `False`):
        Whether or not to return the lengths of the encoded inputs.
    verbose (`bool`, *optional*, defaults to `True`):
        Whether or not to print more information and warnings.
    **kwargs: passed to the `self.tokenize()` method

Return:
    [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

    - **input_ids** -- List of token ids to be fed to a model.

      [What are input IDs?](../glossary#input-ids)

    - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
      if *"token_type_ids"* is in `self.model_input_names`).

      [What are token type IDs?](../glossary#token-type-ids)

    - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
      `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

      [What are attention masks?](../glossary#attention-mask)

    - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
      `return_overflowing_tokens=True`).
    - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
      `return_overflowing_tokens=True`).
    - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
      regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
    - **length** -- The length of the inputs (when `return_length=True`)"""

TOKENIZER_FUNCS = {
    "cl100k": (tiktoken.get_encoding("cl100k_base"), cl100k),
    "llama": (LlamaTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K"), llama),
    "qwen": (AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"), qwen),
}
# https://huggingface.co/Qwen/Qwen3-0.6B
# https://www.datacamp.com/tutorial/fine-tuning-qwen3

def extract_fields(instance, tokenizer_name, tokenizer, tokenizer_func, eos_token):
    instance_id = instance["instance_id"]
    if instance["text"] is None or instance["patch"] is None:
        print(f"No text for {instance_id}")
        return {"input_ids": [], "labels": [], "text": "", "patch": ""}
    text_inputs = instance["text"].strip() + "\n"
    if text_inputs is None or instance["patch"] is None:
        print(f"No inputs for {instance_id}")
        return None
    patch = instance["patch"].strip()
    if len(eos_token) > 0:
        patch += f"\n{eos_token}"
    input_ids = tokenizer_func(text_inputs, tokenizer)
    if tokenizer_name in {"llama"}:
        label_ids = tokenizer_func(
            "\n" + patch, tokenizer
        )  # add newline to tokenize patch
        idx = label_ids.index(13)
        assert idx <= 2, (
            "Expected newline token id (13) to be one of the first three tokens"
        )
        label_ids = label_ids[idx + 1 :]  # remove newline tokens
    else:
        label_ids = tokenizer_func(patch, tokenizer)
    inputs = input_ids + label_ids[:-1]
    cond_len = len(input_ids) - 1
    labels = [-100] * cond_len + label_ids
    assert len(inputs) == len(labels)
    return {
        **instance,
        "input_ids": inputs,
        "labels": labels,
        "text": text_inputs,
        "patch": patch,
    }


def extract_test_fields(instance, tokenizer_name, tokenizer, tokenizer_func, eos_token):
    instance_id = instance["instance_id"]
    if instance["text"] is None or instance["patch"] is None:
        print(f"No text for {instance_id}")
        return None
    text_inputs = instance["text"].strip() + "\n"
    if text_inputs is None or instance["patch"] is None:
        print(f"No inputs for {instance_id}")
        return None
    patch = instance["patch"].strip()
    if len(eos_token) > 0:
        patch += f"\n{eos_token}"
    input_ids = tokenizer_func(text_inputs, tokenizer)
    label_ids = tokenizer_func(patch, tokenizer)
    inputs = input_ids
    labels = label_ids
    return {
        **instance,
        "input_ids": inputs,
        "labels": labels,
        "text": text_inputs,
        "patch": patch,
    }


def add_columns_from_dict(dataset, dict_columns):
    """dict_columns is a list of dicts with keys that are columns in dataset"""
    for column in dict_columns[0].keys():
        values = [d[column] for d in dict_columns]
        if column in dataset.column_names:
            dataset = dataset.remove_columns(column)
        dataset = dataset.add_column(column, values)
    return dataset


def main(
    dataset_name_or_path,
    output_dir,
    tokenizer_name,
    num_proc,
    push_to_hub_user,
):
    if push_to_hub_user is not None:
        hub_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
        if hub_token is None:
            raise ValueError("Must provide HUGGING_FACE_HUB_TOKEN to push to the Hub")
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    if tokenizer_name is not None:
        tokenizer, tokenizer_func = TOKENIZER_FUNCS[tokenizer_name]
        eos_token = getattr(tokenizer, "eos_token", "")
        if num_proc > 0 and tokenizer_name == "cl100k":
            logger.warning(
                "cl100k tokenizer does not support multiprocessing. Ignoring num_proc"
            )
            num_proc = 0

    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)
    dataset = dataset.filter(
        lambda x: len(x["text"]) <= 5_000_000
    )  # filter out superlong instances
    for split in dataset.keys():
        if split == "test":
            continue
        if num_proc > 0:
            dataset[split] = dataset[split].map(
                lambda instance: extract_fields(
                    instance,
                    tokenizer_name,
                    tokenizer,
                    tokenizer_func,
                    eos_token,
                ),
                num_proc=num_proc,
                batched=False,
                desc=f"Tokenizing {split}",
            )
        elif len(dataset[split]) > 0:
            new_values = list(
                map(
                    lambda x: extract_fields(
                        x, tokenizer_name, tokenizer, tokenizer_func, eos_token
                    ),
                    tqdm(
                        dataset[split],
                        total=len(dataset[split]),
                        desc=f"Tokenizing {split}",
                    ),
                )
            )
            dataset[split] = add_columns_from_dict(dataset[split], new_values)
    for split in ["test"]:
        if split not in dataset:
            logger.warning(f"Split {split} not in dataset. Skipping")
            continue
        if num_proc > 0:
            dataset[split] = dataset[split].map(
                lambda instance: extract_test_fields(
                    instance,
                    tokenizer_name,
                    tokenizer,
                    tokenizer_func,
                    eos_token,
                ),
                num_proc=num_proc,
                batched=False,
                desc=f"Tokenizing {split}",
            )
        elif len(dataset[split]) > 0:
            new_values = list(
                map(
                    lambda x: extract_test_fields(
                        x, tokenizer_name, tokenizer, tokenizer_func, eos_token
                    ),
                    tqdm(
                        dataset[split],
                        total=len(dataset[split]),
                        desc=f"Tokenizing {split}",
                    ),
                )
            )
            dataset[split] = add_columns_from_dict(dataset[split], new_values)
    output_file = Path(dataset_name_or_path).name + f"__tok-{tokenizer_name}"
    if push_to_hub_user is not None:
        output_file = f"{push_to_hub_user}/{output_file}"
        dataset.push_to_hub(output_file, use_auth_token=hub_token)
    else:
        output_file = Path(output_dir) / output_file
        dataset.save_to_disk(output_file)
    logger.warning(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--tokenizer_name", type=str, required=True, choices=TOKENIZER_FUNCS.keys()
    )
    parser.add_argument("--num_proc", type=int, default=0)
    parser.add_argument(
        "--push_to_hub_user",
        type=str,
        default=None,
        help="Push the dataset to the Hub user under this name.",
    )
    main(**vars(parser.parse_args()))
