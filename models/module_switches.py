"""
Shared module-switch parsing for CDR-Mamba ablation/combination experiments.
"""


def canonical_ablation_name(name: str) -> str:
    n = (name or "none").lower()
    alias = {
        # 4-module protocol aliases
        "a1_nomamba": "m1_no_mamba",
        "a1_no_mamba": "m1_no_mamba",
        "a2_no_cdr": "m3_no_cdr_decision",
        "a3_no_reconcile": "m3_no_cdr_decision",
        "a4_no_slot": "m2_no_sasp",
        "a5_no_noise": "m4_no_noise",
        # CDR internal mechanism ablations
        "m7_consensus_only": "m7_consensus_only",
        "m8_divergence_only": "m8_divergence_only",
        "consensus_only": "m7_consensus_only",
        "divergence_only": "m8_divergence_only",
        "wo_consensus": "m8_divergence_only",
        "wo_divergence": "m7_consensus_only",
        "w/o_consensus": "m8_divergence_only",
        "w/o_divergence": "m7_consensus_only",
        "a8_consensus_only": "m7_consensus_only",
        "a9_divergence_only": "m8_divergence_only",
        # loss-only ablations
        "a6_no_orth": "m5_no_orth",
        "a7_no_align": "m6_no_align",
        "wo_orth": "m5_no_orth",
        "wo_align": "m6_no_align",
        "w/o_orth": "m5_no_orth",
        "w/o_align": "m6_no_align",
        # merged-module alias (M1 + M2)
        "m12_no_mamba_slot": "m12_no_mamba_sasp",
        "m12_no_mamba_sasp": "m12_no_mamba_sasp",
        # legacy aliases
        "m1_wo_sasp": "m2_no_sasp",
        "m2_wo_cdsm": "m3_no_cdr_decision",
        "m3_wo_rgr": "m3_no_cdr_decision",
    }
    return alias.get(n, n)


VALID_CDR_MODULE_COMBOS = [
    "from_ablation",
    "full",
    # keep-only one module
    "only_m1",
    "only_m2",
    "only_m3",
    "only_m4",
    # pairwise
    "m1_m2",
    "m1_m3",
    "m1_m4",
    "m2_m3",
    "m2_m4",
    "m3_m4",
    # three-module combinations
    "m1_m2_m3",
    "m1_m2_m4",
    "m1_m3_m4",
    "m2_m3_m4",
    # leave-one-out aliases
    "leaveout_m1",
    "leaveout_m2",
    "leaveout_m3",
    "leaveout_m4",
]


def _combo_to_requested(combo: str):
    combo_map = {
        "full": {"m1", "m2", "m3", "m4"},
        "only_m1": {"m1"},
        "only_m2": {"m2"},
        "only_m3": {"m3"},
        "only_m4": {"m4"},
        "m1_m2": {"m1", "m2"},
        "m1_m3": {"m1", "m3"},
        "m1_m4": {"m1", "m4"},
        "m2_m3": {"m2", "m3"},
        "m2_m4": {"m2", "m4"},
        "m3_m4": {"m3", "m4"},
        "m1_m2_m3": {"m1", "m2", "m3"},
        "m1_m2_m4": {"m1", "m2", "m4"},
        "m1_m3_m4": {"m1", "m3", "m4"},
        "m2_m3_m4": {"m2", "m3", "m4"},
        "leaveout_m1": {"m2", "m3", "m4"},
        "leaveout_m2": {"m1", "m3", "m4"},
        "leaveout_m3": {"m1", "m2", "m4"},
        "leaveout_m4": {"m1", "m2", "m3"},
    }
    if combo not in combo_map:
        raise ValueError(f"Unsupported cdr_module_combo: {combo}")
    return combo_map[combo]


def resolve_module_plan(ablation: str, module_combo: str):
    """
    Resolve active module switches for M1-M6.

    Returns a dict:
    {
      "ablation": canonical_ablation,
      "module_combo": combo_name,
      "requested": {"m1": bool, "m2": bool, "m3": bool, "m4": bool, "m5": bool, "m6": bool},
      "active": {"m1": bool, "m2": bool, "m3": bool, "m4": bool, "m5": bool, "m6": bool},
      "notes": [str, ...]
    }
    """
    ab = canonical_ablation_name(ablation)
    combo = (module_combo or "from_ablation").lower()
    if combo not in VALID_CDR_MODULE_COMBOS:
        raise ValueError(
            f"Invalid --cdr_module_combo={combo}. "
            f"Expected one of: {VALID_CDR_MODULE_COMBOS}"
        )

    notes = []
    if combo == "from_ablation":
        disable_m12 = ab == "m12_no_mamba_sasp"
        requested = {
            "m1": not ((ab == "m1_no_mamba") or disable_m12),
            "m2": not ((ab == "m2_no_sasp") or disable_m12),
            "m3": not (ab == "m3_no_cdr_decision"),
            "m4": not (ab == "m4_no_noise"),
            "m5": not (ab == "m5_no_orth"),
            "m6": not (ab == "m6_no_align"),
        }
    else:
        req_set = _combo_to_requested(combo)
        requested = {
            "m1": "m1" in req_set,
            "m2": "m2" in req_set,
            "m3": "m3" in req_set,
            "m4": "m4" in req_set,
            # loss-level modules are kept on unless ablation explicitly disables them.
            "m5": not (ab == "m5_no_orth"),
            "m6": not (ab == "m6_no_align"),
        }

    active = dict(requested)

    # Strict dependency: M3 requires M2-generated slot states.
    if active["m3"] and (not active["m2"]):
        active["m3"] = False
        notes.append("m3_disabled_because_m2_is_off")

    return {
        "ablation": ab,
        "module_combo": combo,
        "requested": requested,
        "active": active,
        "notes": notes,
    }
