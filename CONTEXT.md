# Domain Model: PTM Modular SAE Engine

The core domain model and canonical terminology for interpreting Protein Language Models (PLMs) using Sparse Autoencoders (SAEs) focused on Post-Translational Modifications (PTMs).

## Language

### Protein Language Model (PLM)
A pretrained transformer operating on raw amino acid sequences (e.g., ESM-2 family). Hidden states at intermediate layers represent evolutionary, structural, and chemical constraints.
_Avoid_: Language model, protein transformer, backbone network

### Sparse Autoencoder (SAE)
An unsupervised neural network trained to reconstruct frozen PLM activations via an overcomplete latent dictionary under an explicit sparsity constraint ($L_0 \ll \text{dictionary\_dim}$).
_Avoid_: Autoencoder, dictionary model, decomposition head

### Post-Translational Modification (PTM)
A covalent enzymatic modification of an amino acid side chain (e.g., phosphorylation on Ser/Thr/Tyr, ubiquitination on Lys).
_Avoid_: Mutation, variant, protein edit

### Residue Collapse
A failure mode where an SAE latent fires generically for an amino acid identity (e.g., 'any lysine') rather than a functional biological context or modification state (e.g., 'ubiquitinated lysine in a degron motif').
_Avoid_: Dead feature, token bias, residue identity bias

### Activation Shard
A contiguous 2D binary file in SafeTensors format containing extracted real-residue activation vectors alongside an index manifest.
_Avoid_: Chunk, dump, batch file, tensor slice

### Residue Coordinate
The exact 1-indexed position of an amino acid in a canonical biological sequence matching UniProt/Swiss-Prot numbering. Maps 1-to-1 to 0-indexed activation row index `i` ($i = \text{position} - 1$).
_Avoid_: Token index, sequence offset, position index

### Auxiliary Sequence Context
A 1D representation of the entire protein sequence (computed via mean-pooling across all biological residues or extracting the `<cls>` token) stored separately from token shards to provide global gating and subcellular/family conditioning for the modular SAE.
_Avoid_: Pooled token, whole-protein vector, global embedding

### Chemical Stratum
The chemical amino acid class defining the evaluation denominator for an SAE feature (e.g., `serine_threonine`, `lysine`, `asparagine`, `cysteine`, `arginine`, `tyrosine`). Statistical enrichment is calculated strictly within stratum to guard against Residue Collapse.
_Avoid_: Amino acid group, target residue class

### Corpus Partition
The non-overlapping split (`discovery` vs `held_out`) assigned by clustering protein sequences at 50% sequence identity to prevent homology leakage. SAE dictionaries are trained only on `discovery`.
_Avoid_: Train/test split, data fold

### Negative Tier
The confidence classification of an unmodified residue: `verified` (experimentally proven non-modified), `hard` (unmodified residue in the same stratum on a modified protein), or `background` (unannotated candidate in the proteome).
_Avoid_: Negative sample, control token

### Ambiguity Mask
A per-type boolean exclusion flag marking residues where mass spectrometry peptide fragmentation could not resolve the exact modification coordinate. Ambiguous residues are excluded from both positive and negative evaluation tallies.
_Avoid_: Dropout mask, ignore flag

### Homology Cluster
A group of protein sequences clustered at 50% sequence identity (via MMseqs2 or CD-HIT). The cluster is the indivisible atomic unit of dataset partitioning: all proteins in a cluster are assigned to either `discovery` or `held_out` to prevent homology leakage.
_Avoid_: Family group, sequence bucket

### Corpus Definition
The selection criteria defining which proteins enter the analysis pipeline: `annotated_restricted` (proteins with at least one experimental PTM) or `proteome_wide` (all reviewed Swiss-Prot human proteins <= 1022 residues).
_Avoid_: Dataset filter, inclusion rule
