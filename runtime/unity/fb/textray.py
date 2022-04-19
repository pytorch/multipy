#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# This file is originally copied from pytext/contrib/pytext_lib/models/textray.py.
# Save a stable copy here so it's easier to tryout unity for textray
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from accelerators.pytorch.lib.utils.batching import (
    listify,
    make_prediction_texts,
    destructure_tensor_list,
    max_tokens,
)
from accelerators.pytorch.lib.utils.move_fast import (
    copy_to_pinned_cpu,
    wait_for_copies,
)
from multiray.enums.feature_schema_enum import (
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_REPRESENTATION_1D,
    FEATURE_SCHEMA_REPRESENTATION_2D,
)
from multiray.enums.quantization_schema_enum import (
    QUANTIZATION_SCHEMA,
    QUANTIZATION_SCHEMA_FLOAT16_BYTE_PACKED,
    QUANTIZATION_SCHEMA_INT2_FIXED_VALUES,
)
from multiray.enums.signal_name_enum import (
    SIGNAL_NAME,
    SIGNAL_NAME_SIGNAL_1D,
    SIGNAL_NAME_SIGNAL_2D,
)
from pytext.contrib.pytext_lib.transforms.fb_roberta import TextRayTransform
from pytext.contrib.pytext_lib.utils.fb_patchers import patch_all
from pytorch.text.fb.models.roberta import RobertaEncoder


def normalize_embeddings(embeddings: torch.Tensor):
    # assume [batch, embed_dim] dimensions
    # eps to make sure everything works in fp16
    return torch.nn.functional.normalize(embeddings, eps=1e-6)


def sample_inputs():
    return [
        (["apple"],),
        (["i love you"],),
        (["my brilliant friend"],),
        (["apple", "i love you", "my brilliant friend"],),
    ]


class TextRayEncoder(nn.Module):
    """
    Frozen TextRay Encoder.
    Adapted from pytext.fb.pur.models.textray.RoBERTaTextRay
    https://fburl.com/diffusion/ir42lk7q

    Original docstring:
    RoBERTa model whose torchscript export ouputs a sentence_embedding,
    a per-token embedding and sequence length for each input.
    """

    def __init__(
        self,
        vocab_size: int = 250008,
        embedding_dim: int = 1024,
        num_attention_heads: int = 16,
        num_encoder_layers: int = 24,
        encoder_path: Optional[
            str
        ] = "manifold://pytext_training/tree/users/mikaell/textray_ckpts/prod/2021_may_272114236/encoder.pt",
        scaling: Optional[float] = None,
        strict: bool = True,
        normalize_before: bool = False,
        normalize_before_in_transformer: bool = False,
        max_input_text_length: int = 10000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        # TODO in the future build encoder through hydra config after upgrade to 1.1
        self.encoder = RobertaEncoder(
            encoder_path=encoder_path,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            num_encoder_layers=num_encoder_layers,
            output_dropout=0.0,
            dropout=0.0,
            scaling=scaling,
            freeze=True,
            strict=strict,
            normalize_before=normalize_before,
            normalize_before_in_transformer=normalize_before_in_transformer,
        )
        self.fix_signal_name: int = SIGNAL_NAME[SIGNAL_NAME_SIGNAL_1D]
        self.var_signal_name: int = SIGNAL_NAME[SIGNAL_NAME_SIGNAL_2D]
        self.fix_feature_schema: int = FEATURE_SCHEMA[FEATURE_SCHEMA_REPRESENTATION_1D]
        self.var_feature_schema: int = FEATURE_SCHEMA[FEATURE_SCHEMA_REPRESENTATION_2D]
        self.fix_quantization_schema: int = QUANTIZATION_SCHEMA[
            QUANTIZATION_SCHEMA_FLOAT16_BYTE_PACKED
        ]
        self.var_quantization_schema: int = QUANTIZATION_SCHEMA[
            QUANTIZATION_SCHEMA_INT2_FIXED_VALUES
        ]
        self.max_input_text_length = max_input_text_length
        self.transform = TextRayTransform()

    @torch.jit.export
    def tokenize(self, input_str: str) -> List[Tuple[str, int, int]]:
        if self.max_input_text_length is not None:
            input_str = input_str[: self.max_input_text_length]
        pieces = self.transform._token_transform.sp_model.EncodeAsPieces(input_str)
        tokens: List[Tuple[str, int, int]] = []
        end = 0
        for piece in pieces:
            # separator character
            original_piece = piece.lstrip("\u2581")
            start = input_str.find(original_piece, end)
            end = start + len(original_piece)
            token: Tuple[str, int, int] = (piece, start, end)
            tokens.append(token)
        return tokens

    # Adapted from pytext.data.bert_tensorizer.BertTensorizerBaseScriptImpl.tokenize
    # https://fburl.com/diffusion/qfcv3sm0
    @torch.jit.export
    def _tokenize(self, row_text: List[str]):
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = []
        if row_text is not None:
            for text in row_text:
                tokens = self.tokenize(text)
                per_sentence_tokens.append(tokens)
        return per_sentence_tokens

    def train(self, mode: bool = True):
        # Set model to eval to disable dropout if it is frozen
        return super().train(mode=False)

    def forward(
        self, tokens: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Default forward pass with expected type signature for training / eval,
        overridden when module is scripted:
            > self.eval().to(device=device, dtype=dtype)
            > self.forward = self.inference_forward
            > torch.jit.script(self)
        """
        return self.forward_util(tokens, pad_mask)

    def forward_util(
        self, tokens: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        # The idea is to use this class only for evaluating/exporting TextRay.
        # For that use-case we want the encoder frozen, and all dropouts set to 0.
        # This is currently implemented in the task here D28554973
        assert (
            not self.encoder.training
        ), "to ensure 0 dropouts, encoder must be in eval"
        reps = self.encoder.transformer(tokens)  # lists of T x B x C
        fix_rep = reps[-1][0, :, :]
        fix_rep = normalize_embeddings(fix_rep)
        # skip reps[0] because that is the embedding representation
        var_rep = torch.stack(reps[1:]).mean(dim=0).transpose(0, 1)  # B x T x C
        # for similarity evaluation use case we don't currently pass pad_mask
        seq_lens = (
            pad_mask.sum(dim=1) if pad_mask is not None else torch.empty_like(var_rep)
        )  # B
        # TODO add quantization here
        return fix_rep, var_rep, seq_lens

    @torch.jit.export
    def inference_util(self, batch: Dict[str, Any]):
        """
        TextRayTransform returns a dict of tensors, which is indexed
        and fed into forward_util, which generates the variable
        and fixed emeddings. Copy to pinned memory and return.
        """
        input_tensors: Dict[str, torch.Tensor] = self.transform(batch)
        tokens: torch.Tensor = input_tensors["token_ids"]
        pad_mask: torch.Tensor = input_tensors["pad_mask"]
        fix_reps, var_reps, seq_lens = self.forward_util(tokens, pad_mask)
        fix_reps_cpu = copy_to_pinned_cpu(fix_reps)
        var_reps_cpu = copy_to_pinned_cpu(var_reps)
        seq_lens_cpu = copy_to_pinned_cpu(seq_lens)
        wait_for_copies([fix_reps, var_reps, seq_lens])
        return fix_reps_cpu, [
            var_reps_cpu[i, : seq_lens_cpu[i]] for i in range(len(seq_lens_cpu))
        ]

    def inference_forward(
        self, texts: List[str]
    ) -> Dict[int, Tuple[List[torch.Tensor], int, int]]:
        """
        Forward pass for type signature expected by Predictor for inference
        Returns both fixed and variable representations
        """
        inputs: Dict[str, Any] = {}
        inputs["text"] = texts
        fix_res, var_res = self.inference_util(inputs)

        return {
            self.fix_signal_name: (
                listify(fix_res),
                self.fix_feature_schema,
                self.fix_quantization_schema,
            ),
            self.var_signal_name: (
                var_res,
                self.var_feature_schema,
                self.var_quantization_schema,
            ),
        }

    # Adapted from pytext.torchscript.module.PyTextEmbeddingModule (not overridden by MultirayEmbeddingModule)
    # https://fburl.com/diffusion/l1tfucqe
    @torch.jit.export
    def make_batch(
        self,
        mega_batch: List[
            Tuple[
                List[str],  # texts
                int,
            ]
        ],
        goals: Dict[str, str],
    ) -> List[List[Tuple[List[str], int,]]]:  # texts

        batchsize = len(mega_batch)

        if batchsize == 0:
            raise RuntimeError("Input batch must have at least 1 batch element")

        # The next lines sort all cross-request batch elements by the token length.
        # Note that cross-request batch element can in turn be a client batch.
        mega_batch_key_list = [
            (max_tokens(self._tokenize(x[0])), n) for (n, x) in enumerate(mega_batch)
        ]

        sorted_mega_batch_key_list = sorted(mega_batch_key_list)
        sorted_mega_batch = [mega_batch[n] for (_, n) in sorted_mega_batch_key_list]

        # TBD: allow model server to specify batch size in goals dictionary
        # the right batchsize depends on the target architecture and should
        # be passed via the goals config doctionary
        max_bs = int(goals.get("batchsize", "4"))
        len_mb = len(mega_batch)
        num_batches = (len_mb + max_bs - 1) // max_bs

        batch_list: List[
            List[
                Tuple[
                    List[str],  # texts
                    int,  # position
                ]
            ]
        ] = []

        start = 0

        for _i in range(num_batches):
            end = min(start + max_bs, len_mb)
            batch_list.append(sorted_mega_batch[start:end])
            start = end

        return batch_list

    @torch.jit.export
    def make_prediction(
        self,
        batch: List[
            Tuple[
                List[str],
            ]
        ],  # texts
    ) -> List[Dict[int, Tuple[List[torch.Tensor], int, int]]]:

        res_list: List[Dict[int, Tuple[List[torch.Tensor], int, int]]] = []

        flat_result: Dict[int, Tuple[List[torch.Tensor], int, int]] = self.forward(
            texts=make_prediction_texts(batch),
        )

        fix_res_list = destructure_tensor_list(
            [len(be[0]) for be in batch], flat_result[self.fix_signal_name][0]
        )
        var_res_list = destructure_tensor_list(
            [len(be[0]) for be in batch], flat_result[self.var_signal_name][0]
        )

        # group results by single id
        for f, v in zip(fix_res_list, var_res_list):
            res_list.append(
                {
                    self.fix_signal_name: (
                        f,
                        self.fix_feature_schema,
                        self.fix_quantization_schema,
                    ),
                    self.var_signal_name: (
                        v,
                        self.var_feature_schema,
                        self.var_quantization_schema,
                    ),
                }
            )

        return res_list

    @torch.jit.export
    def set_device(self, device: str):
        self.transform.set_device(device)
        print(f"Set device to {device}.")

    @torch.jit.export
    def set_padding_control(self, dimension: str, control: Optional[List[int]]):
        """
        This functions will be called to set a padding style.
        None - No padding
        List: first element 0, round seq length to the smallest list element larger than inputs
        """
        self.transform.set_padding_control(dimension, control)
        print(f"Set {dimension} padding control to {control}.")

    @torch.jit.export
    def prepare_for_inference(self, device, dtype):
        self.eval().to(device=device, dtype=dtype)
        self.forward = self.inference_forward

    @torch.jit.ignore
    def torchscriptify(self, device, dtype):
        """TODO: T103134644"""
        self.prepare_for_inference(device, dtype)
        return torch.jit.script(self)


if __name__ == "__main__":
    print("loading TextRay")
    patch_all()
    model = TextRayEncoder()
    model.prepare_for_inference(device="cuda", dtype=torch.float16)
    model.set_padding_control("sequence_length", list(range(0, 257, 8)))
    model.set_device("cuda")
    pred = model.make_prediction(sample_inputs())
    print(f"pred is {pred}")
