from __future__ import annotations

# ===========================================================================
# NLI templates + lexical pairs
# ===========================================================================
"""Controlled NLI template generator.

We synthesise tiny NLI examples by plugging single-word lexical items
into a fixed sentence frame. This gives us full control over the position
of the *content word* in the tokenized input, which is critical for
position-wise interventions like activation patching and DAS.

Each :class:`NLITemplate` defines:

- a ``premise_format`` containing the placeholder ``{word}``,
- a ``hypothesis_format`` containing the placeholder ``{word}``,
- an ``answer_prefix`` that nudges the model to emit a label token
  (we use a tiny verbalizer at decode time, e.g. " yes"/" no"/" maybe"),
- a ``monotonicity`` flag and a ``monotonicity_marker`` token used by
  DAS to localise the monotonicity-determining position.

We ship two banks of templates:

- ``UPWARD_TEMPLATES``: positive contexts where FORWARD → entailment.
- ``DOWNWARD_TEMPLATES``: negated contexts where FORWARD → neutral.

The default :data:`DEFAULT_TEMPLATES` includes all of them. For
single-template experiments (the fixed-position DAS convention) pick one
explicitly. Lexical pairs are auto-generated from the hypernym DAG in
:mod:`src.data.causal_model` so we don't have to maintain a hand-curated
list (which was small, noisy, and class-imbalanced in earlier revisions).
"""


from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .causal_models import (
    LABEL2ID,
    LexicalCausalModel,
    _HYPERNYMS,
    _TOP_LEVEL_CATEGORY,
    lexical_relation,
)


# ---------------------------------------------------------------------------
# Vocabulary derivation
# ---------------------------------------------------------------------------


def _leaf_words() -> Tuple[str, ...]:
    """Return the leaf words of the ontology (words with at least one
    hypernym chain, in insertion order)."""
    return tuple(w for w, chain in _HYPERNYMS.items() if chain)


def _all_words() -> Tuple[str, ...]:
    """All words appearing in the ontology, including internal hypernyms."""
    seen: Dict[str, None] = {}
    for w, chain in _HYPERNYMS.items():
        seen.setdefault(w, None)
        for h in chain:
            seen.setdefault(h, None)
    return tuple(seen.keys())


def auto_generate_pairs(
    *,
    include_equiv: bool = True,
    include_forward: bool = True,
    include_reverse: bool = True,
    include_disjoint: bool = True,
    max_disjoint_per_word: int = 4,
    seed: int = 0,
) -> Tuple[Tuple[str, str, str], ...]:
    """Auto-generate lexical pairs from the hypernym DAG.

    The four relations are:

    - **EQUIV**: ``(w, w, "EQUIV")`` for each leaf word.
    - **FORWARD**: ``(w, h, "FORWARD")`` for each (word, hypernym) in
      the transitive closure of ``_HYPERNYMS``.
    - **REVERSE**: the swap of every FORWARD pair.
    - **DISJOINT**: cross-category pairs sampled deterministically so each
      leaf word appears in at most ``max_disjoint_per_word`` DISJOINT
      pairs. Symmetric in (a, b).

    The result is sanity-checked against :func:`lexical_relation`, which
    is the source of truth.

    Parameters
    ----------
    include_*:
        Per-relation toggles (mainly useful for ablation tests).
    max_disjoint_per_word:
        Caps how often each leaf appears as the *premise* word in a
        DISJOINT pair. Keeps the dataset roughly class-balanced.
    seed:
        Deterministic ordering / sampling seed.
    """
    import random

    rng = random.Random(seed)
    leaves = list(_leaf_words())
    out: List[Tuple[str, str, str]] = []
    seen: set = set()

    def _emit(a: str, b: str, rel: str) -> None:
        key = (a, b, rel)
        if key in seen:
            return
        # Sanity check against the symbolic source-of-truth.
        actual = lexical_relation(a, b)
        if actual != rel:
            return
        seen.add(key)
        out.append(key)

    if include_equiv:
        for w in leaves:
            _emit(w, w, "EQUIV")

    if include_forward or include_reverse:
        for w, chain in _HYPERNYMS.items():
            for h in chain:
                if w == h:
                    continue
                if include_forward:
                    _emit(w, h, "FORWARD")
                if include_reverse:
                    _emit(h, w, "REVERSE")

    if include_disjoint:
        # Group leaves by top-level category and sample cross-category pairs.
        by_cat: Dict[str, List[str]] = {}
        for w in leaves:
            by_cat.setdefault(_TOP_LEVEL_CATEGORY.get(w, "_"), []).append(w)
        # Symmetric DISJOINT: build a budget per word.
        budget: Dict[str, int] = {w: max_disjoint_per_word for w in leaves}
        cats = list(by_cat.keys())
        rng.shuffle(cats)
        for ca, cb in combinations(cats, 2):
            wa = list(by_cat[ca])
            wb = list(by_cat[cb])
            rng.shuffle(wa)
            rng.shuffle(wb)
            for a in wa:
                if budget[a] <= 0:
                    continue
                for b in wb:
                    if budget[b] <= 0:
                        continue
                    if lexical_relation(a, b) != "DISJOINT":
                        continue
                    _emit(a, b, "DISJOINT")
                    budget[a] -= 1
                    budget[b] -= 1
                    if budget[a] <= 0:
                        break

    return tuple(out)


# Materialise once at import time so downstream code can use a stable
# default. Callers can rebuild with their own toggles via auto_generate_pairs.
LEXICAL_PAIRS: Tuple[Tuple[str, str, str], ...] = auto_generate_pairs()


# ---------------------------------------------------------------------------
# Template dataclass
# ---------------------------------------------------------------------------


@dataclass
class NLITemplate:
    """A reusable NLI surface form.

    The format strings must each contain a single ``{word}`` placeholder.
    The ``answer_prefix`` is concatenated after the hypothesis so we can
    read a label from the next-token logits over the verbalizer tokens.
    """

    name: str
    premise_format: str = "A {word} is on the table."
    hypothesis_format: str = "A {word} is on the table."
    answer_prefix: str = " Answer:"
    # The verbalizer maps NLI labels to surface strings that should be
    # tokenizable to a single token after a leading space (e.g. " yes",
    # " no", " maybe" for GPT-2).
    verbalizer: Dict[str, str] = field(
        default_factory=lambda: {
            "entailment": " yes",
            "neutral": " maybe",
            "contradiction": " no",
        }
    )
    # Whether the carrier sentence is upward- or downward-monotone in the
    # premise/hypothesis word slot. Used to look up the right
    # relation->label table.
    monotonicity: str = "upward"
    # Substring whose token position carries the monotonicity feature.
    # For "A X is on the table." the determiner "A" is the canonical
    # upward marker; for "No X is on the table." it's "No". Used by DAS
    # when target_variable == "monotonicity" to localise the intervention.
    monotonicity_marker: str = "A"

    def format_prompt(self, premise_word: str, hypothesis_word: str) -> str:
        """Materialize a full prompt for a (premise_word, hypothesis_word) pair."""
        premise = self.premise_format.format(word=premise_word)
        hypothesis = self.hypothesis_format.format(word=hypothesis_word)
        return f"{premise} {hypothesis}{self.answer_prefix}"


# A bank of upward-monotone templates. Each places the content word in a
# different syntactic environment so the rotation has to find a relation
# feature, not a surface-form feature.
UPWARD_TEMPLATES: Tuple[NLITemplate, ...] = (
    NLITemplate(
        name="on_the_table",
        premise_format="A {word} is on the table.",
        hypothesis_format="A {word} is on the table.",
        monotonicity="upward",
        monotonicity_marker="A",
    ),
    NLITemplate(
        name="i_saw_a",
        premise_format="I saw a {word} yesterday.",
        hypothesis_format="I saw a {word} yesterday.",
        monotonicity="upward",
        monotonicity_marker="a",
    ),
    NLITemplate(
        name="there_is_a",
        premise_format="There is a {word} in the garden.",
        hypothesis_format="There is a {word} in the garden.",
        monotonicity="upward",
        monotonicity_marker="a",
    ),
    NLITemplate(
        name="some_x_exists",
        premise_format="Some {word} exists.",
        hypothesis_format="Some {word} exists.",
        monotonicity="upward",
        monotonicity_marker="Some",
    ),
)

# Downward-monotone templates: negation flips FORWARD/REVERSE.
DOWNWARD_TEMPLATES: Tuple[NLITemplate, ...] = (
    NLITemplate(
        name="no_x_on_table",
        premise_format="No {word} is on the table.",
        hypothesis_format="No {word} is on the table.",
        monotonicity="downward",
        monotonicity_marker="No",
    ),
    NLITemplate(
        name="not_a_x",
        premise_format="It is not a {word}.",
        hypothesis_format="It is not a {word}.",
        monotonicity="downward",
        monotonicity_marker="not",
    ),
)


# DEFAULT_TEMPLATES is the union; experiments that want a single template
# should pick one explicitly (e.g. ``UPWARD_TEMPLATES[:1]``).
DEFAULT_TEMPLATES: Tuple[NLITemplate, ...] = UPWARD_TEMPLATES + DOWNWARD_TEMPLATES


# ---------------------------------------------------------------------------
# Example dataclass + generator
# ---------------------------------------------------------------------------


@dataclass
class NLIExample:
    """One controlled NLI example with full provenance."""

    template_name: str
    premise_word: str
    hypothesis_word: str
    lexical_relation: str
    label: str
    label_id: int
    prompt: str
    monotonicity: str = "upward"

    def as_dict(self) -> Dict[str, object]:
        return {
            "template_name": self.template_name,
            "premise_word": self.premise_word,
            "hypothesis_word": self.hypothesis_word,
            "lexical_relation": self.lexical_relation,
            "label": self.label,
            "label_id": self.label_id,
            "prompt": self.prompt,
            "monotonicity": self.monotonicity,
        }


def generate_examples(
    pairs: Sequence[Tuple[str, str, str]] = LEXICAL_PAIRS,
    templates: Sequence[NLITemplate] = DEFAULT_TEMPLATES,
    max_examples: Optional[int] = None,
) -> List[NLIExample]:
    """Materialize the cartesian product of ``pairs`` x ``templates``.

    Parameters
    ----------
    pairs:
        Iterable of ``(premise_word, hypothesis_word, relation)`` triples.
        Defaults to :data:`LEXICAL_PAIRS`.
    templates:
        Iterable of :class:`NLITemplate`. Defaults to :data:`DEFAULT_TEMPLATES`.
    max_examples:
        If given, truncate the output to at most this many examples.

    Returns
    -------
    list of :class:`NLIExample`.
    """
    out: List[NLIExample] = []
    for template in templates:
        # We respect the template's monotonicity when deriving labels.
        causal = LexicalCausalModel(monotonicity=template.monotonicity)
        for premise_word, hypothesis_word, expected_rel in pairs:
            trace = causal.run(
                premise_word=premise_word,
                hypothesis_word=hypothesis_word,
                context=template.name,
            )
            # Sanity-check that our hand-labeled relation agrees with the
            # symbolic model. If it doesn't, prefer the symbolic model and
            # warn loudly via assertion - this catches typos in
            # LEXICAL_PAIRS during development.
            assert trace["lexical_relation"] == expected_rel, (
                f"Relation mismatch for ({premise_word!r}, {hypothesis_word!r}): "
                f"hand-labeled {expected_rel!r} vs model {trace['lexical_relation']!r}"
            )
            prompt = template.format_prompt(premise_word, hypothesis_word)
            out.append(
                NLIExample(
                    template_name=template.name,
                    premise_word=premise_word,
                    hypothesis_word=hypothesis_word,
                    lexical_relation=trace["lexical_relation"],
                    label=trace["label"],
                    label_id=LABEL2ID[trace["label"]],
                    prompt=prompt,
                    monotonicity=template.monotonicity,
                )
            )
            if max_examples is not None and len(out) >= max_examples:
                return out
    return out


def label_distribution(examples: Sequence[NLIExample]) -> Dict[str, int]:
    """Count NLI labels in a list of examples. Useful for sanity checks."""
    out: Dict[str, int] = {}
    for ex in examples:
        out[ex.label] = out.get(ex.label, 0) + 1
    return out


def relation_distribution(examples: Sequence[NLIExample]) -> Dict[str, int]:
    """Count lexical relations in a list of examples."""
    out: Dict[str, int] = {}
    for ex in examples:
        out[ex.lexical_relation] = out.get(ex.lexical_relation, 0) + 1
    return out


# ===========================================================================
# Counterfactual dataset construction + controls + pair-level holdout
# ===========================================================================
"""Build (base, source) interchange-intervention pairs.

For each base example we pair it with one or more *source* examples and
compute the **gold counterfactual label** -- the label the model *should*
output after we transplant the value of some intermediate variable from
``source`` into ``base``.

Concretely, given a high-level intermediate variable ``V`` (e.g.
``lexical_relation`` or ``premise_word_identity``):

    counterfactual_label(base, source, V)
        = high_level.run(base_inputs,
                         interventions={V: high_level(source_inputs)[V]})

We materialize these tuples up front as a :class:`CounterfactualDataset`
which can be plugged directly into a DAS training loop or an IIA
evaluation loop. The tokenized fields (``base_input_ids``,
``source_input_ids``, ``intervention_pos``) are pre-computed so the
intervention loop just has to call the model.
"""


import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .causal_models import (
    LABEL2ID,
    LexicalCausalModel,
)
# DEFAULT_TEMPLATES, LEXICAL_PAIRS, NLITemplate, NLIExample,
# UPWARD_TEMPLATES, DOWNWARD_TEMPLATES, generate_examples
# are all defined above in the templates section of this same file.


# ---------------------------------------------------------------------------
# Single-example container
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualExample:
    """One (base, source) interchange-intervention example.

    Fields
    ------
    base:
        :class:`NLIExample` for the base input.
    source:
        :class:`NLIExample` for the source input.
    target_variable:
        Name of the high-level intermediate variable being patched.
    base_label_id:
        Gold label id for the *unintervened* base example.
    source_label_id:
        Gold label id for the *unintervened* source example.
    counterfactual_label_id:
        Gold label id after patching ``target_variable`` from source
        into base, according to the high-level causal model. **This is
        the prediction target for IIA.**
    intervention_pos:
        Token index in the base sequence whose representation should be
        replaced. Computed by :func:`build_counterfactual_dataset`.
    """

    base: NLIExample
    source: NLIExample
    target_variable: str
    base_label_id: int
    source_label_id: int
    counterfactual_label_id: int
    intervention_pos: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["base"] = self.base.as_dict()
        d["source"] = self.source.as_dict()
        return d


# ---------------------------------------------------------------------------
# Position localisation
# ---------------------------------------------------------------------------


def _word_token_position(
    tokenizer,
    prompt: str,
    target_word: str,
    occurrence: int = 0,
) -> Optional[int]:
    """Return the token index of ``target_word`` inside ``prompt``.

    We tokenize the prompt with ``return_offsets_mapping=True`` and look
    for the first token whose character span begins inside the substring
    range of the requested occurrence of ``target_word``. If the
    tokenizer cannot give offsets (rare for HF fast tokenizers), we fall
    back to a slower decode-based search.
    """
    if tokenizer.is_fast:
        enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        # Find the ``occurrence``-th occurrence of target_word in prompt.
        start = -1
        end = -1
        search_from = 0
        for _ in range(occurrence + 1):
            start = prompt.find(target_word, search_from)
            if start == -1:
                return None
            end = start + len(target_word)
            search_from = end
        for tok_idx, (a, b) in enumerate(offsets):
            if a == b:  # special token / empty span
                continue
            if a <= start < b or (start <= a < end):
                return tok_idx
        return None

    # Slow path: decode each prefix and find the first prefix that
    # contains the target word.
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    for i in range(len(ids)):
        decoded = tokenizer.decode(ids[: i + 1])
        if target_word in decoded:
            return i
    return None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class CounterfactualDataset(Dataset):
    """Torch Dataset over pre-tokenized counterfactual examples.

    Each item is a dict with keys:
        - ``base_input_ids``  : LongTensor [T]
        - ``source_input_ids``: LongTensor [T]
        - ``base_attention_mask``, ``source_attention_mask``
        - ``base_label_id``           : LongTensor scalar
        - ``source_label_id``         : LongTensor scalar
        - ``counterfactual_label_id`` : LongTensor scalar (training target)
        - ``intervention_pos``        : LongTensor scalar
        - ``target_variable``         : str (collated as list)
    """

    def __init__(
        self,
        examples: Sequence[CounterfactualExample],
        tokenizer,
        max_length: Optional[int] = None,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        # Tokenize once and pad to the longest prompt across base+source
        # so positions align. This is the cleanest setup for fixed-position
        # interventions.
        all_prompts = [ex.base.prompt for ex in self.examples] + [
            ex.source.prompt for ex in self.examples
        ]
        enc = tokenizer(
            all_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        n = len(self.examples)
        self._base_input_ids = enc["input_ids"][:n]
        self._source_input_ids = enc["input_ids"][n:]
        self._base_attn = enc["attention_mask"][:n]
        self._source_attn = enc["attention_mask"][n:]

    def __len__(self) -> int:
        return len(self.examples)

    def filter_by_position(self, position: int) -> "CounterfactualDataset":
        """Return a copy containing only examples whose intervention_pos equals *position*."""
        import copy
        indices = [i for i, ex in enumerate(self.examples) if int(ex.intervention_pos) == position]
        if not indices:
            raise ValueError(f"No examples found at intervention_pos={position}.")
        idx = torch.as_tensor(indices, dtype=torch.long)
        new_ds = copy.copy(self)
        new_ds.examples = [self.examples[i] for i in indices]
        new_ds._base_input_ids = self._base_input_ids[idx].clone()
        new_ds._source_input_ids = self._source_input_ids[idx].clone()
        new_ds._base_attn = self._base_attn[idx].clone()
        new_ds._source_attn = self._source_attn[idx].clone()
        return new_ds

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        pos = ex.intervention_pos if ex.intervention_pos is not None else 0
        return {
            "base_input_ids": self._base_input_ids[idx],
            "source_input_ids": self._source_input_ids[idx],
            "base_attention_mask": self._base_attn[idx],
            "source_attention_mask": self._source_attn[idx],
            "base_label_id": torch.tensor(ex.base_label_id, dtype=torch.long),
            "source_label_id": torch.tensor(ex.source_label_id, dtype=torch.long),
            "counterfactual_label_id": torch.tensor(
                ex.counterfactual_label_id, dtype=torch.long
            ),
            "intervention_pos": torch.tensor(pos, dtype=torch.long),
            "target_variable": ex.target_variable,
        }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


# Which surface word each high-level intermediate variable corresponds to.
# This is the bridge between the symbolic causal model and the token
# positions in the prompt that we will intervene on.
#
# Values:
#   "premise_word"        -- intervene on the premise's content word
#   "hypothesis_word"     -- intervene on the hypothesis's content word
#   "monotonicity_marker" -- intervene on the template's monotonicity-marking
#                            token (e.g. "A" / "No" / "Some" / "not")
_VARIABLE_TO_SOURCE_WORD: Dict[str, str] = {
    "premise_word_identity": "premise_word",
    "hypothesis_word_identity": "hypothesis_word",
    # ``lexical_relation`` is not localised to a single surface word at
    # the high level; we treat the *hypothesis* word position as the
    # default intervention site (it's where the model first has both
    # operands available) but callers can override.
    "lexical_relation": "hypothesis_word",
    # Monotonicity is localised to the carrier sentence's
    # monotonicity-marking token (the determiner/quantifier/negator).
    "monotonicity": "monotonicity_marker",
}


def build_counterfactual_dataset(
    tokenizer,
    *,
    target_variable: str = "lexical_relation",
    pairs: Sequence[Tuple[str, str, str]] = LEXICAL_PAIRS,
    templates: Sequence[NLITemplate] = DEFAULT_TEMPLATES,
    source_templates: Optional[Sequence[NLITemplate]] = None,
    n_examples: int = 64,
    seed: int = 0,
    max_length: Optional[int] = None,
    intervention_word: Optional[str] = None,
    require_label_change: bool = True,
    require_monotonicity_flip: Optional[bool] = None,
) -> CounterfactualDataset:
    """Generate a counterfactual dataset for interchange interventions.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer used to localise the intervention position.
    target_variable:
        Which intermediate variable in the high-level model to patch.
        Must be one of ``"premise_word_identity"``,
        ``"hypothesis_word_identity"``, ``"lexical_relation"``,
        ``"monotonicity"``.
    pairs, templates:
        Lexical pairs and base-side surface templates.
    source_templates:
        Optional separate template bank for the *source* side. Defaults
        to ``templates`` (same templates for base and source). Pass an
        opposite-monotonicity bank to construct monotonicity-flipping
        interventions.
    n_examples:
        Number of (base, source) pairs to generate.
    seed:
        RNG seed for reproducibility.
    max_length:
        Optional max sequence length passed to the tokenizer.
    intervention_word:
        Which surface word's token to patch. Defaults to the canonical
        position for ``target_variable`` (see ``_VARIABLE_TO_SOURCE_WORD``).
    require_label_change:
        If True (default), only keep pairs whose counterfactual label
        *differs* from the base label. This makes IIA evaluation strictly
        harder (random-guess baseline drops) and is the convention used
        in the original DAS paper. The default flipped to True in this
        revision; pass ``False`` to recover the old behaviour.
    require_monotonicity_flip:
        If True, require source.monotonicity != base.monotonicity. If
        None (default), True iff ``target_variable == "monotonicity"``.

    Returns
    -------
    :class:`CounterfactualDataset`
    """
    if target_variable not in _VARIABLE_TO_SOURCE_WORD:
        raise ValueError(
            f"target_variable {target_variable!r} not in "
            f"{sorted(_VARIABLE_TO_SOURCE_WORD)}"
        )
    intervention_word_key = intervention_word or _VARIABLE_TO_SOURCE_WORD[target_variable]

    if require_monotonicity_flip is None:
        require_monotonicity_flip = (target_variable == "monotonicity")

    rng = random.Random(seed)
    base_examples = generate_examples(pairs=pairs, templates=templates)
    if not base_examples:
        raise ValueError("No NLI examples were generated.")

    if source_templates is None:
        source_examples = base_examples
        source_templates_resolved: Sequence[NLITemplate] = templates
    else:
        source_examples = generate_examples(pairs=pairs, templates=source_templates)
        source_templates_resolved = source_templates

    # Lookup map for quick template-by-name access.
    template_by_name: Dict[str, NLITemplate] = {
        t.name: t for t in (list(templates) + list(source_templates_resolved))
    }

    out: List[CounterfactualExample] = []
    attempts = 0
    max_attempts = n_examples * 40  # avoid infinite loops if filters are tight

    while len(out) < n_examples and attempts < max_attempts:
        attempts += 1
        base = rng.choice(base_examples)
        base_template = template_by_name[base.template_name]

        # Source candidates depend on what we're intervening on.
        if require_monotonicity_flip:
            # For monotonicity DAS we want sources from a different
            # monotonicity class but a structurally compatible template
            # (matching marker position). The simplest invariant: source
            # template's monotonicity_marker token must land at the same
            # token index as the base's. We don't recompute positions
            # here; we just require opposite monotonicity, and rely on
            # filter_by_position downstream to keep aligned pairs.
            candidate_sources = [
                ex for ex in source_examples
                if ex.monotonicity != base.monotonicity
            ]
        else:
            # Default: same template so positions align.
            candidate_sources = [
                ex for ex in source_examples if ex.template_name == base.template_name
            ]
        if not candidate_sources:
            continue
        source = rng.choice(candidate_sources)
        source_template = template_by_name[source.template_name]

        # Compute the gold counterfactual label by querying the high-level
        # model with the intervention applied. The causal model picks the
        # right relation->label table from the BASE template's monotonicity
        # by default; if we're patching monotonicity itself, the cf trace
        # uses the SOURCE's monotonicity.
        causal = LexicalCausalModel(monotonicity=base.monotonicity)

        # Build the intervention dict: replace the target variable in the
        # base trace with the source's value of that variable.
        source_trace = causal.run(
            premise_word=source.premise_word,
            hypothesis_word=source.hypothesis_word,
            context=source.template_name,
            monotonicity=source.monotonicity,
        )
        interventions = {target_variable: source_trace[target_variable]}
        cf_trace = causal.run(
            premise_word=base.premise_word,
            hypothesis_word=base.hypothesis_word,
            context=base.template_name,
            interventions=interventions,
        )
        cf_label_id = cf_trace["label_id"]

        if require_label_change and cf_label_id == base.label_id:
            continue

        # Localise the intervention position inside the base prompt.
        if intervention_word_key == "monotonicity_marker":
            target_word = base_template.monotonicity_marker
            occurrence = 0
        else:
            target_word = {
                "premise_word": base.premise_word,
                "hypothesis_word": base.hypothesis_word,
            }[intervention_word_key]
            # If premise_word == hypothesis_word (EQUIV bases) we still want
            # the *second* occurrence when the variable is the hypothesis
            # word.
            occurrence = 0
            if (
                intervention_word_key == "hypothesis_word"
                and base.premise_word == base.hypothesis_word
            ):
                occurrence = 1
        pos = _word_token_position(
            tokenizer, base.prompt, target_word, occurrence=occurrence
        )
        if pos is None:
            # Skip examples we can't localise.
            continue

        out.append(
            CounterfactualExample(
                base=base,
                source=source,
                target_variable=target_variable,
                base_label_id=base.label_id,
                source_label_id=source.label_id,
                counterfactual_label_id=cf_label_id,
                intervention_pos=pos,
            )
        )

    if len(out) < n_examples:
        # Soft warning rather than a hard error: callers may have asked
        # for more examples than the (small) lexical bank can supply
        # under tight filters.
        import warnings

        warnings.warn(
            f"Only produced {len(out)} / {n_examples} counterfactual examples "
            f"after {attempts} attempts. Consider relaxing filters or "
            f"expanding LEXICAL_PAIRS.",
            RuntimeWarning,
        )

    return CounterfactualDataset(out, tokenizer=tokenizer, max_length=max_length)


def _template_monotonicity(
    template_name: str,
    templates: Sequence[NLITemplate],
) -> str:
    """Look up a template's monotonicity by name; default to ``"upward"``."""
    for t in templates:
        if t.name == template_name:
            return t.monotonicity
    return "upward"


# ---------------------------------------------------------------------------
# Controls: random-source and wrong-variable
# ---------------------------------------------------------------------------


def build_random_source_dataset(
    dataset: CounterfactualDataset,
    *,
    seed: int = 0,
) -> CounterfactualDataset:
    """Build the *random-source* control from an existing dataset.

    For each base example we keep the same base prompt and intervention
    position, but reassign a *random* source from the original pool. The
    counterfactual label is recomputed under the new (random) source. If
    DAS is genuinely encoding the target variable in the rotated
    subspace, IIA on this control should drop sharply -- the rotation
    should propagate whatever value the random source happens to have,
    not the desired (base-aligned) target value.

    Implementation note: this re-derives the cf label using the same
    causal model used in :func:`build_counterfactual_dataset`. Positions
    are not recomputed (the intervention site stays where it was in
    ``dataset``).
    """
    import copy

    rng = random.Random(seed)
    examples = list(dataset.examples)
    if len(examples) < 2:
        raise ValueError("Need at least 2 examples to build a random-source control.")

    new_examples: List[CounterfactualExample] = []
    sources_pool = [ex.source for ex in examples]
    for ex in examples:
        # Sample a source distinct from the current one.
        while True:
            new_source = rng.choice(sources_pool)
            if new_source is not ex.source:
                break
        causal = LexicalCausalModel(monotonicity=ex.base.monotonicity)
        src_trace = causal.run(
            premise_word=new_source.premise_word,
            hypothesis_word=new_source.hypothesis_word,
            context=new_source.template_name,
            monotonicity=new_source.monotonicity,
        )
        cf_trace = causal.run(
            premise_word=ex.base.premise_word,
            hypothesis_word=ex.base.hypothesis_word,
            context=ex.base.template_name,
            interventions={ex.target_variable: src_trace[ex.target_variable]},
        )
        new_examples.append(
            CounterfactualExample(
                base=ex.base,
                source=new_source,
                target_variable=ex.target_variable,
                base_label_id=ex.base.label_id,
                source_label_id=new_source.label_id,
                counterfactual_label_id=cf_trace["label_id"],
                intervention_pos=ex.intervention_pos,
            )
        )

    new_ds = copy.copy(dataset)
    new_ds.examples = new_examples
    # Re-tokenize sources (bases are unchanged, but it's simpler to rebuild).
    new_ds = CounterfactualDataset(
        new_examples, tokenizer=dataset.tokenizer
    )
    return new_ds


def build_wrong_variable_dataset(
    dataset: CounterfactualDataset,
    *,
    wrong_variable: str,
) -> CounterfactualDataset:
    """Build the *wrong-variable* control from an existing dataset.

    Keeps the (base, source, position) triples but recomputes the
    counterfactual label as if the intervention were targeting
    ``wrong_variable`` instead of the dataset's recorded
    ``target_variable``. If DAS at the original site really aligns with
    the *original* target variable, evaluating the trained rotation on
    this dataset should give chance-level IIA -- the rotation is
    propagating the right kind of information, but it's being scored
    against the wrong gold label.
    """
    if wrong_variable not in _VARIABLE_TO_SOURCE_WORD:
        raise ValueError(
            f"wrong_variable {wrong_variable!r} not in "
            f"{sorted(_VARIABLE_TO_SOURCE_WORD)}"
        )

    new_examples: List[CounterfactualExample] = []
    for ex in dataset.examples:
        causal = LexicalCausalModel(monotonicity=ex.base.monotonicity)
        src_trace = causal.run(
            premise_word=ex.source.premise_word,
            hypothesis_word=ex.source.hypothesis_word,
            context=ex.source.template_name,
            monotonicity=ex.source.monotonicity,
        )
        cf_trace = causal.run(
            premise_word=ex.base.premise_word,
            hypothesis_word=ex.base.hypothesis_word,
            context=ex.base.template_name,
            interventions={wrong_variable: src_trace[wrong_variable]},
        )
        new_examples.append(
            CounterfactualExample(
                base=ex.base,
                source=ex.source,
                target_variable=wrong_variable,
                base_label_id=ex.base.label_id,
                source_label_id=ex.source.label_id,
                counterfactual_label_id=cf_trace["label_id"],
                intervention_pos=ex.intervention_pos,
            )
        )

    return CounterfactualDataset(new_examples, tokenizer=dataset.tokenizer)


# ---------------------------------------------------------------------------
# Pair-level train/eval split
# ---------------------------------------------------------------------------


def pair_level_split(
    pairs: Sequence[Tuple[str, str, str]],
    *,
    train_frac: float = 0.8,
    seed: int = 0,
) -> Tuple[Tuple[Tuple[str, str, str], ...], Tuple[Tuple[str, str, str], ...]]:
    """Split lexical pairs into train/eval so no triple appears in both.

    This is stricter than splitting after counterfactual generation:
    here we guarantee the eval set never reuses a (premise_word,
    hypothesis_word, relation) combination from the train set, which is
    the relevant invariant for "did the rotation generalize to held-out
    lexical items?".

    Returns
    -------
    (train_pairs, eval_pairs)
        Two tuples whose union is ``pairs`` and whose intersection is
        empty. Stratified by relation so both splits have all 4 classes.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")

    rng = random.Random(seed)
    by_rel: Dict[str, List[Tuple[str, str, str]]] = {}
    for p in pairs:
        by_rel.setdefault(p[2], []).append(p)

    train: List[Tuple[str, str, str]] = []
    evalset: List[Tuple[str, str, str]] = []
    for rel, lst in by_rel.items():
        lst_copy = list(lst)
        rng.shuffle(lst_copy)
        n_train = max(1, int(round(len(lst_copy) * train_frac)))
        train.extend(lst_copy[:n_train])
        evalset.extend(lst_copy[n_train:])
    return tuple(train), tuple(evalset)
