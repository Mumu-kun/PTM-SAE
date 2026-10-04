"""Shared fixtures."""

import numpy as np
import pandas as pd
import pytest

from ptm_sae.data.splitting import SplitSettings

PTM_TYPES = (("phospho", 0.03), ("ubiq", 0.015), ("acetyl", 0.01), ("rare", 0.002))


@pytest.fixture
def small_split_settings() -> SplitSettings:
    """Settings sized for the synthetic corpus below: tiny MILP, floors that fit its type counts."""
    return SplitSettings(
        binary_clusters=30,
        node_limit=100,
        time_limit_s=60,
        min_type_sites=50,
        held_min_sites=100,
        val_min_sites=50,
        token_tolerance=0.03,
        polish_sweeps=30,
        audit_passes=2,
    )


@pytest.fixture
def make_corpus():
    """Factory for a lumpy synthetic corpus: many singleton clusters plus a few big ones, four PTM
    types with different densities. Returns (proteins, sites) in the splitter's input schema."""

    def build(n_clusters: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.RandomState(seed)
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        sizes = rng.choice([1, 1, 1, 2, 3, 5, 12, 30], size=n_clusters)
        proteins, sites = [], []
        for cluster_id, size in enumerate(sizes):
            for _ in range(size):
                uniprot_id = f"P{len(proteins):05d}"
                length = int(rng.randint(80, 400))
                sequence = "".join(rng.choice(alphabet, length))
                proteins.append(
                    {
                        "uniprot_id": uniprot_id,
                        "cluster_id": cluster_id,
                        "sequence": sequence,
                        "length": length,
                    }
                )
                for ptm_type, density in PTM_TYPES:
                    sites += [
                        {
                            "uniprot_id": uniprot_id,
                            "cluster_id": cluster_id,
                            "type_pooled": ptm_type,
                            "crosstalk": bool(rng.rand() < 0.1),
                        }
                        for _ in range(rng.binomial(length, density))
                    ]
        return pd.DataFrame(proteins), pd.DataFrame(sites)

    return build
