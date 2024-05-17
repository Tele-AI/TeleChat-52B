# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes for TELECHAT."""
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm
import re
from transformers.convert_slow_tokenizer import import_protobuf
from transformers import AddedToken, PreTrainedTokenizer
from transformers.utils import logging
from transformers.tokenization_utils_base import TextInput

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "tokenizer_file": {},
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "telechat-tokenizer": 8192,
}
SPIECE_UNDERLINE = "▁"


class TELECHATTokenizer(PreTrainedTokenizer):
    """
    Construct a TELECHAT tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        sp_model_kwargs (`Dict[str, Any]`, `Optional`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.

    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=False,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=False,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(self, from_slow=False):
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = True
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        split_special_tokens = kwargs.pop("split_special_tokens", self.split_special_tokens)
        remove_dummy_prefix = kwargs.pop("remove_dummy_prefix", False)

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase. Might be super slow as well?
            escaped_special_toks = [re.escape(s_tok) for s_tok in (self.all_special_tokens)]
            escaped_special_toks += [
                re.escape(s_tok.content)
                for s_tok in (self._added_tokens_decoder.values())
                if not s_tok.special and s_tok.normalized
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        if split_special_tokens:
            no_split_token = []
            tokens = [text]
        else:
            no_split_token = self._added_tokens_encoder.keys()  # don't split on any of the added tokens
            # "This is something<special_token_1>  else"
            tokens = self.tokens_trie.split(text)

        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = self._added_tokens_decoder.get(self._added_tokens_encoder[token], None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                    if tok_extended.single_word and left and left[-1] != " ":
                        tokens[i - 1] += token
                        tokens[i] = ""
                    elif tok_extended.single_word and right and right[0] != " ":
                        tokens[i + 1] = token + tokens[i + 1]
                        tokens[i] = ""
                else:
                    raise ValueError(
                        f"{tok_extended} cannot be tokenized because it was not properly added"
                        f" to the tokenizer. This means that it is not an `AddedToken` but a {type(tok_extended)}"
                    )
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token, remove_dummy_prefix=remove_dummy_prefix))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We add a option to remove dummpy prefix during tokenization instead of changing the default behaviour of the sentencepiece tokenizer.
        This is useful when there're two tokenized sentences to be merged into one as the last one will have an extra dummy prefix which results in a
        inconsistant pattern.
        """
        tokens = self.sp_model.encode(text, out_type=str)
        if text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens
        if len(tokens) > 0 and kwargs.get("remove_dummy_prefix") is True:
            tokens[0] = tokens[0].replace(SPIECE_UNDERLINE, "", 1)
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        # prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                # if not prev_is_special and i != 0 and self.legacy:
                #     out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                # prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                # prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output
