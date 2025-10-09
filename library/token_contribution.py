"""Utilities for computing token-level contributions with gradient-weighted attention.

This module implements a lightweight attribution pipeline that operates directly on the
trained transformer models without materialising the full attention tensors on disk.
It supports the following analyses that were discussed in the design proposal:

1. **Single token importance** via Grad×Input on top of the input embeddings.
2. **Token pair importance** through gradient-weighted attention rollout with a
   configurable Top-K neighbourhood restriction.
3. **Subset importance** by evaluating the logit delta that results from masking a
   token subset (a batched approximation of the Shapley value idea).

The implementation keeps intermediate tensors on the GPU, extracts only the sparse
statistics that we care about, and therefore avoids the quadratic memory blow up that
the previous approach suffered from.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_shap_top_tokens(path: str, top_n: int = 10) -> List[int]:
    """Load the top-N SHAP tokens from the given file.

    The SHAP files in StrainAMR share a simple tab-separated format with at least two
    columns where the second column stores the token ID. We only keep the first
    ``top_n`` entries to limit the search space of follow-up analyses.
    """

    tokens: List[int] = []
    if not path or not os.path.exists(path):
        return tokens

    with open(path, "r", encoding="utf-8") as handle:
        # Skip header
        next(handle, None)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                tokens.append(int(parts[1]))
            except ValueError:
                continue
            if len(tokens) >= top_n:
                break
    return tokens


def load_shap_annotations(path: str) -> Tuple[Dict[int, Dict[str, str]], List[str]]:
    """Parse auxiliary annotation columns from a SHAP ranking file."""

    annotations: Dict[int, Dict[str, str]] = {}
    columns: List[str] = []
    if not path or not os.path.exists(path):
        return annotations, columns

    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        if not header or len(header) < 2:
            return annotations, columns

        def _find_token_index(names: Iterable[str]) -> int:
            for candidate in ("Token_ID", "TokenID", "Token", "token_id", "token"):
                if candidate in names:
                    return names.index(candidate)
            return -1

        token_idx = _find_token_index(header)
        if token_idx < 0:
            return annotations, columns

        excluded = {header[0], header[token_idx], "Shap", "Shap_0", "Shap_1", "Value", "Rank"}
        columns = [name for name in header if name not in excluded]
        col_indices = {name: header.index(name) for name in columns}

        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= token_idx:
                continue
            token_str = parts[token_idx].strip()
            try:
                token_id = int(token_str)
            except ValueError:
                continue
            annotations[token_id] = {
                col: parts[col_indices[col]].strip() if col_indices[col] < len(parts) else ""
                for col in columns
            }

    return annotations, columns


@dataclass
class SampleContribution:
    sample_id: str
    token_scores: Mapping[str, MutableMapping[int, float]]
    pair_scores: Mapping[str, MutableMapping[Tuple[int, int], float]]
    subset_scores: Mapping[str, MutableMapping[Tuple[int, ...], float]]


@dataclass
class AttributionResult:
    token_scores: Mapping[str, MutableMapping[int, float]]
    pair_scores: Mapping[str, MutableMapping[Tuple[int, int], float]]
    subset_scores: Mapping[str, MutableMapping[Tuple[int, ...], float]]
    per_sample: Optional[List[SampleContribution]] = None


class TokenContributionAnalyzer:
    """Compute token/pair/subset contributions for a trained model.

    The analyzer attaches a small collection of hooks to the model in order to capture
    the input embeddings as well as the per-layer attention weights. Hooks are only
    active while ``record_mode`` is ``True`` which prevents interference with
    auxiliary forward passes (e.g. the batched masking procedure used for subset
    contributions).
    """

    def __init__(
        self,
        model: nn.Module,
        encoder_names: Sequence[str],
        device: Optional[torch.device] = None,
        baseline_token_id: int = 0,
    ) -> None:
        if isinstance(model, nn.DataParallel):
            model = model.module  # unwrap for hook registration
        self.model = model
        self.encoder_names = list(encoder_names)
        self.device = device or next(model.parameters()).device
        self.baseline_token_id = baseline_token_id

        self._embedding_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._attention_handles: List[torch.utils.hooks.RemovableHandle] = []

        self._current_embeddings: List[Optional[torch.Tensor]] = []
        self._current_attentions: List[List[Optional[torch.Tensor]]] = []

        self._record_mode = False

        self._register_hooks()

    # ------------------------------------------------------------------ utils --
    def _register_hooks(self) -> None:
        self._embedding_handles.clear()
        self._attention_handles.clear()
        self._current_embeddings = [None for _ in self.encoder_names]
        self._current_attentions = []

        for encoder_idx, encoder_name in enumerate(self.encoder_names):
            encoder = getattr(self.model, encoder_name, None)
            if encoder is None:
                self._current_attentions.append([])
                continue

            def _embedding_hook_factory(index: int):
                def hook(_: nn.Module, __: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                    if not self._record_mode:
                        return
                    output.retain_grad()
                    self._current_embeddings[index] = output

                return hook

            emb_handle = encoder.word_embedding.register_forward_hook(
                _embedding_hook_factory(encoder_idx)
            )
            self._embedding_handles.append(emb_handle)

            attn_list: List[Optional[torch.Tensor]] = [None] * len(getattr(encoder, "layers", []))
            self._current_attentions.append(attn_list)

            for layer_idx, layer in enumerate(getattr(encoder, "layers", [])):
                attention_module = getattr(layer, "attention", None)
                if attention_module is None:
                    continue

                def _attention_hook_factory(e_idx: int, l_idx: int):
                    def hook(
                        _: nn.Module,
                        __: Tuple[torch.Tensor, ...],
                        output: Tuple[torch.Tensor, torch.Tensor],
                    ) -> None:
                        if not self._record_mode:
                            return
                        if not isinstance(output, tuple) or len(output) < 2:
                            return
                        attn = output[1]
                        attn.retain_grad()
                        self._current_attentions[e_idx][l_idx] = attn

                    return hook

                handle = attention_module.register_forward_hook(
                    _attention_hook_factory(encoder_idx, layer_idx)
                )
                self._attention_handles.append(handle)

    # ----------------------------------------------------------------- cleanup --
    def close(self) -> None:
        for handle in self._embedding_handles:
            handle.remove()
        for handle in self._attention_handles:
            handle.remove()
        self._embedding_handles.clear()
        self._attention_handles.clear()

    # --------------------------------------------------------------- helpers --
    @staticmethod
    def _attention_rollout(attn_stack: torch.Tensor) -> torch.Tensor:
        """Perform the attention rollout described in Abnar & Zuidema (2020).

        ``attn_stack`` has shape ``(num_layers, batch, seq_len, seq_len)`` and is
        assumed to already be gradient weighted and averaged across attention heads.
        """

        num_layers, batch_size, seq_len, _ = attn_stack.shape
        eye = torch.eye(seq_len, device=attn_stack.device).unsqueeze(0).repeat(batch_size, 1, 1)
        rollout = eye
        for layer in range(num_layers):
            attn = attn_stack[layer]
            attn = torch.relu(attn)
            attn = attn + torch.eye(seq_len, device=attn_stack.device).unsqueeze(0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            rollout = torch.bmm(attn, rollout)
        return rollout

    def _model_forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        outputs = self.model(*inputs)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    # -------------------------------------------------------------- analysis --
    def analyze_dataset(
        self,
        tensors: Sequence[torch.Tensor],
        feature_labels: Sequence[str],
        shap_sets: Mapping[str, Sequence[int]],
        batch_size: int = 8,
        top_k_pairs: int = 20,
        subset_sizes: Sequence[int] = (2,),
        subset_sample_size: int = 32,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> AttributionResult:
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        token_scores: Dict[str, MutableMapping[int, float]] = {
            label: defaultdict(float) for label in feature_labels
        }
        pair_scores: Dict[str, MutableMapping[Tuple[int, int], float]] = {
            label: defaultdict(float) for label in feature_labels
        }
        subset_scores: Dict[str, MutableMapping[Tuple[int, ...], float]] = {
            label: defaultdict(float) for label in feature_labels
        }

        per_sample_token: Optional[List[Dict[str, MutableMapping[int, float]]]] = None
        per_sample_pair: Optional[List[Dict[str, MutableMapping[Tuple[int, int], float]]]] = None
        per_sample_subset: Optional[List[Dict[str, MutableMapping[Tuple[int, ...], float]]]] = None

        if sample_ids is not None:
            total_len = len(dataset)
            if total_len != len(sample_ids):
                raise ValueError("sample_ids length must match dataset length")
            per_sample_token = [
                {label: defaultdict(float) for label in feature_labels}
                for _ in sample_ids
            ]
            per_sample_pair = [
                {label: defaultdict(float) for label in feature_labels}
                for _ in sample_ids
            ]
            per_sample_subset = [
                {label: defaultdict(float) for label in feature_labels}
                for _ in sample_ids
            ]

        prev_training_state = self.model.training
        self.model.eval()

        try:
            sample_offset = 0
            for batch in loader:
                inputs = [tensor.to(self.device).long() for tensor in batch]

                self.model.zero_grad(set_to_none=True)
                self._record_mode = True
                logits = self._model_forward(inputs)
                logits.sum().backward()
                self._record_mode = False

                batch_size_eff = inputs[0].shape[0]
                token_contrib_per_feature: List[Optional[torch.Tensor]] = []
                attn_rollout_per_feature: List[Optional[torch.Tensor]] = []

                for f_idx, label in enumerate(feature_labels):
                    shap_tokens = set(shap_sets.get(label, []))
                    if not shap_tokens:
                        token_contrib_per_feature.append(None)
                        attn_rollout_per_feature.append(None)
                        continue

                    embedding = self._current_embeddings[f_idx]
                    if embedding is None:
                        token_contrib_per_feature.append(None)
                        attn_rollout_per_feature.append(None)
                        continue

                    grad = embedding.grad
                    token_contrib = (embedding * grad).sum(dim=-1)
                    token_contrib_per_feature.append(token_contrib.detach().cpu())

                    attentions = [att for att in self._current_attentions[f_idx] if att is not None]
                    if attentions:
                        grads = [att.grad for att in attentions]
                        att_stack = torch.stack(attentions)  # (layers, batch, heads, seq, seq)
                        grad_stack = torch.stack(grads)
                        grad_weighted = torch.relu(att_stack * grad_stack).mean(dim=2)
                        rollout = self._attention_rollout(grad_weighted).detach().cpu()
                        attn_rollout_per_feature.append(rollout)
                    else:
                        attn_rollout_per_feature.append(None)

                logits_cpu = logits.detach().cpu().squeeze(1)

                for sample_idx in range(batch_size_eff):
                    sample_index = sample_offset + sample_idx
                    base_inputs = [inp[sample_idx].detach().clone() for inp in inputs]
                    base_logit = float(logits_cpu[sample_idx])

                    for f_idx, label in enumerate(feature_labels):
                        shap_tokens = set(shap_sets.get(label, []))
                        if not shap_tokens:
                            continue

                        tokens_tensor = base_inputs[f_idx].cpu()
                        token_scores_tensor = token_contrib_per_feature[f_idx]
                        if token_scores_tensor is None:
                            continue

                        seq_scores = token_scores_tensor[sample_idx]
                        seq_rollout = attn_rollout_per_feature[f_idx]
                        if seq_rollout is not None:
                            seq_rollout = seq_rollout[sample_idx]

                        shap_positions: List[int] = [
                            int(pos)
                            for pos, token_id in enumerate(tokens_tensor.tolist())
                            if token_id != 0 and token_id in shap_tokens
                        ]

                        for pos in shap_positions:
                            token_id = int(tokens_tensor[pos].item())
                            token_scores[label][token_id] += float(seq_scores[pos].item())
                            if per_sample_token is not None:
                                per_sample_token[sample_index][label][token_id] += float(
                                    seq_scores[pos].item()
                                )

                        if seq_rollout is not None and shap_positions:
                            seq_rollout = seq_rollout.cpu()
                            for pos in shap_positions:
                                token_id = int(tokens_tensor[pos].item())
                                row = seq_rollout[pos]
                                values, indices = torch.topk(row, k=min(top_k_pairs, row.shape[0]))
                                for value, neighbour_idx in zip(values.tolist(), indices.tolist()):
                                    if neighbour_idx == pos:
                                        continue
                                    neighbour_token = int(tokens_tensor[neighbour_idx].item())
                                    if neighbour_token == 0:
                                        continue
                                    pair_scores[label][(token_id, neighbour_token)] += float(value)
                                    if per_sample_pair is not None:
                                        per_sample_pair[sample_index][label][
                                            (token_id, neighbour_token)
                                        ] += float(value)

                        if subset_sizes and shap_positions:
                            subset_delta = self._estimate_subset_delta(
                                base_inputs,
                                f_idx,
                                shap_positions,
                                base_logit,
                                subset_sizes,
                                subset_sample_size,
                            )
                            for subset_key, score in subset_delta.items():
                                subset_scores[label][subset_key] += score
                                if per_sample_subset is not None:
                                    per_sample_subset[sample_index][label][subset_key] += score

                sample_offset += batch_size_eff

        finally:
            self.model.train(prev_training_state)

        per_sample_results: Optional[List[SampleContribution]] = None
        if sample_ids is not None and per_sample_token is not None:
            per_sample_results = []
            for idx, sample_id in enumerate(sample_ids):
                per_sample_results.append(
                    SampleContribution(
                        sample_id=sample_id,
                        token_scores=per_sample_token[idx],
                        pair_scores=per_sample_pair[idx] if per_sample_pair else {},
                        subset_scores=per_sample_subset[idx] if per_sample_subset else {},
                    )
                )

        return AttributionResult(token_scores, pair_scores, subset_scores, per_sample_results)

    def _estimate_subset_delta(
        self,
        base_inputs: Sequence[torch.Tensor],
        feature_idx: int,
        shap_positions: Sequence[int],
        base_logit: float,
        subset_sizes: Sequence[int],
        subset_sample_size: int,
    ) -> Dict[Tuple[int, ...], float]:
        results: Dict[Tuple[int, ...], float] = defaultdict(float)
        if not subset_sizes:
            return results

        combos: List[Tuple[int, ...]] = []
        rng = random.Random(len(shap_positions))
        for size in subset_sizes:
            if size > len(shap_positions):
                continue
            positions = list(combinations(shap_positions, size))
            if subset_sample_size and len(positions) > subset_sample_size:
                positions = rng.sample(positions, subset_sample_size)
            combos.extend(positions)

        if not combos:
            return results

        chunk_size = max(1, min(32, subset_sample_size))
        with torch.no_grad():
            for start in range(0, len(combos), chunk_size):
                chunk = combos[start : start + chunk_size]
                masked_inputs = []
                for combo in chunk:
                    cloned_inputs = [tensor.clone() for tensor in base_inputs]
                    for pos in combo:
                        cloned_inputs[feature_idx][pos] = self.baseline_token_id
                    masked_inputs.append(cloned_inputs)

                stacked_inputs = [
                    torch.stack([masked[i] for masked in masked_inputs], dim=0).to(self.device)
                    for i in range(len(base_inputs))
                ]

                logits = self._model_forward(stacked_inputs).detach().cpu().squeeze(1)
                deltas = base_logit - logits

                for combo, delta in zip(chunk, deltas.tolist()):
                    subset_tokens = tuple(
                        sorted(int(base_inputs[feature_idx][pos].item()) for pos in combo)
                    )
                    results[subset_tokens] += float(delta)

        return results


def export_token_contributions(
    model: nn.Module,
    feature_arrays: Sequence[torch.Tensor],
    feature_labels: Sequence[str],
    shap_file_map: Mapping[str, str],
    output_dir: str,
    encoder_names: Sequence[str],
    device: Optional[torch.device] = None,
    batch_size: int = 8,
    top_k_pairs: int = 20,
    subset_sizes: Sequence[int] = (2,),
    subset_sample_size: int = 32,
    sample_ids: Optional[Sequence[str]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    shap_sets: Dict[str, Sequence[int]] = {}
    for label in feature_labels:
        shap_sets[label] = load_shap_top_tokens(shap_file_map.get(label, ""))

    if not any(shap_sets.values()):
        return

    tensors = [tensor.clone().long() for tensor in feature_arrays]
    analyzer = TokenContributionAnalyzer(model, encoder_names, device=device)
    try:
        result = analyzer.analyze_dataset(
            tensors,
            feature_labels,
            shap_sets,
            batch_size=batch_size,
            top_k_pairs=top_k_pairs,
            subset_sizes=subset_sizes,
            subset_sample_size=subset_sample_size,
            sample_ids=sample_ids,
        )
    finally:
        analyzer.close()

    shap_annotations: Dict[str, Dict[int, Dict[str, str]]] = {}
    shap_annotation_cols: Dict[str, List[str]] = {}
    for label, shap_path in shap_file_map.items():
        annotations, cols = load_shap_annotations(shap_path)
        shap_annotations[label] = annotations
        shap_annotation_cols[label] = cols

    for label in feature_labels:
        if not shap_sets.get(label):
            continue

        token_file = os.path.join(output_dir, f"{label}_token_grad_importance.tsv")
        pair_file = os.path.join(output_dir, f"{label}_token_pair_grad_importance.tsv")
        subset_file = os.path.join(output_dir, f"{label}_token_subset_delta.tsv")

        annotations = shap_annotations.get(label, {})
        annot_cols = shap_annotation_cols.get(label, [])

        with open(token_file, "w", encoding="utf-8") as handle:
            header = ["Token_ID", "GradInput"] + annot_cols
            handle.write("\t".join(header) + "\n")
            for token_id, score in sorted(
                result.token_scores[label].items(), key=lambda kv: abs(kv[1]), reverse=True
            ):
                row = [str(token_id), str(score)]
                token_meta = annotations.get(token_id, {})
                row.extend(token_meta.get(col, "") for col in annot_cols)
                handle.write("\t".join(row) + "\n")

        with open(pair_file, "w", encoding="utf-8") as handle:
            pair_header = ["Token_A", "Token_B", "GradWeightedAttention"]
            if annot_cols:
                pair_header.extend(f"Token_A_{col}" for col in annot_cols)
                pair_header.extend(f"Token_B_{col}" for col in annot_cols)
            handle.write("\t".join(pair_header) + "\n")
            for (token_a, token_b), score in sorted(
                result.pair_scores[label].items(), key=lambda kv: abs(kv[1]), reverse=True
            ):
                row = [str(token_a), str(token_b), str(score)]
                if annot_cols:
                    meta_a = annotations.get(token_a, {})
                    meta_b = annotations.get(token_b, {})
                    row.extend(meta_a.get(col, "") for col in annot_cols)
                    row.extend(meta_b.get(col, "") for col in annot_cols)
                handle.write("\t".join(row) + "\n")

        with open(subset_file, "w", encoding="utf-8") as handle:
            subset_header = ["Token_Subset", "DeltaLogit"]
            if annot_cols:
                subset_header.extend(f"Subset_{col}" for col in annot_cols)
            handle.write("\t".join(subset_header) + "\n")
            for subset_key, score in sorted(
                result.subset_scores[label].items(), key=lambda kv: abs(kv[1]), reverse=True
            ):
                subset_str = ",".join(str(token_id) for token_id in subset_key)
                row = [subset_str, str(score)]
                if annot_cols:
                    for col in annot_cols:
                        col_values = [
                            annotations.get(token_id, {}).get(col, "")
                            for token_id in subset_key
                        ]
                        row.append(";".join(col_values))
                handle.write("\t".join(row) + "\n")

        if result.per_sample:
            token_sample_file = os.path.join(
                output_dir, f"{label}_sample_token_grad_importance.tsv"
            )
            pair_sample_file = os.path.join(
                output_dir, f"{label}_sample_token_pair_grad_importance.tsv"
            )
            subset_sample_file = os.path.join(
                output_dir, f"{label}_sample_token_subset_delta.tsv"
            )

            with open(token_sample_file, "w", encoding="utf-8") as handle:
                header = ["Sample_ID", "Token_ID", "GradInput"] + annot_cols
                handle.write("\t".join(header) + "\n")
                for sample in result.per_sample:
                    scores = sample.token_scores.get(label, {})
                    for token_id, score in sorted(
                        scores.items(), key=lambda kv: abs(kv[1]), reverse=True
                    ):
                        token_meta = annotations.get(token_id, {})
                        row = [sample.sample_id, str(token_id), str(score)]
                        row.extend(token_meta.get(col, "") for col in annot_cols)
                        handle.write("\t".join(row) + "\n")

            with open(pair_sample_file, "w", encoding="utf-8") as handle:
                header = ["Sample_ID", "Token_A", "Token_B", "GradWeightedAttention"]
                if annot_cols:
                    header.extend(f"Token_A_{col}" for col in annot_cols)
                    header.extend(f"Token_B_{col}" for col in annot_cols)
                handle.write("\t".join(header) + "\n")
                for sample in result.per_sample:
                    scores = sample.pair_scores.get(label, {})
                    for (token_a, token_b), score in sorted(
                        scores.items(), key=lambda kv: abs(kv[1]), reverse=True
                    ):
                        row = [sample.sample_id, str(token_a), str(token_b), str(score)]
                        if annot_cols:
                            meta_a = annotations.get(token_a, {})
                            meta_b = annotations.get(token_b, {})
                            row.extend(meta_a.get(col, "") for col in annot_cols)
                            row.extend(meta_b.get(col, "") for col in annot_cols)
                        handle.write("\t".join(row) + "\n")

            with open(subset_sample_file, "w", encoding="utf-8") as handle:
                header = ["Sample_ID", "Token_Subset", "DeltaLogit"]
                if annot_cols:
                    header.extend(f"Subset_{col}" for col in annot_cols)
                handle.write("\t".join(header) + "\n")
                for sample in result.per_sample:
                    scores = sample.subset_scores.get(label, {})
                    for subset_key, score in sorted(
                        scores.items(), key=lambda kv: abs(kv[1]), reverse=True
                    ):
                        subset_str = ",".join(str(token_id) for token_id in subset_key)
                        row = [sample.sample_id, subset_str, str(score)]
                        if annot_cols:
                            for col in annot_cols:
                                col_values = [
                                    annotations.get(token_id, {}).get(col, "")
                                    for token_id in subset_key
                                ]
                                row.append(";".join(col_values))
                        handle.write("\t".join(row) + "\n")

