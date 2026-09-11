"""
Módulo de Avaliação, Explicabilidade (XAI) e Equidade (Fairness)
Tech Challenge - Fase 3 | FIAP PosTech
"""

from src.evaluation.fairness_audit import compute_slice_metrics, run_fairness_audit

__all__ = ["compute_slice_metrics", "run_fairness_audit"]
