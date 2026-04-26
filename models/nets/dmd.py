"""
Compatibility wrapper for CDR-Mamba model.

The full method implementation (including A1-A5 ablation switches)
is placed in `cdr_mamba_method.py`.
"""

from .cdr_mamba_method import CDRMamba, DMD

__all__ = ["CDRMamba", "DMD"]

