"""nli_das — supporting library for the CS 221M final-project walkthrough.

Public API (everything else is internal):

Causal models
-------------
- :class:`LexicalCausalModel` -- simple 4-relation lexical NLI causal graph
- :func:`load_mqnli_causal_model` -- the full MQNLI 33-variable graph

Data
----
- :func:`build_counterfactual_dataset`
- :func:`build_random_source_dataset`
- :func:`build_wrong_variable_dataset`
- :func:`pair_level_split`
- :data:`LEXICAL_PAIRS`, :data:`UPWARD_TEMPLATES`, :data:`DOWNWARD_TEMPLATES`

Interventions
-------------
- :func:`make_das_config`
- :func:`train_das_alignment`
- :func:`evaluate_das_iia`
- :func:`run_patching_sweep`

Plotting
--------
- :func:`save_patching_heatmap_from_df`
"""

from .causal_models import (
    LexicalCausalModel,
    load_mqnli_causal_model,
    LABELS, LABEL2ID, ID2LABEL,
    LEXICAL_RELATIONS, MONOTONICITIES,
)
from .data import (
    LEXICAL_PAIRS,
    DEFAULT_TEMPLATES, UPWARD_TEMPLATES, DOWNWARD_TEMPLATES,
    NLIExample, NLITemplate,
    generate_examples, label_distribution, relation_distribution,
    CounterfactualExample, CounterfactualDataset,
    build_counterfactual_dataset,
    build_random_source_dataset,
    build_wrong_variable_dataset,
    pair_level_split,
)
from .das import (
    LabelVerbalizer,
    compute_iia, compute_iia_per_class,
    make_das_config, das_config_meta,
    train_das_alignment, DASTrainOutput,
    evaluate_das_iia,
)
from .patching import (
    run_patching_sweep, run_single_patch,
    PatchingResult, run_activation_patching_sweep,
)
from .plotting import save_patching_heatmap_from_df
