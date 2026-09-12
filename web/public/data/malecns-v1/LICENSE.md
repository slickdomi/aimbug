# License: connectome data

The files in this directory are adapted from the **Male CNS connectome v1.0**. That dataset is a collaboration between FlyEM (HHMI Janelia), the University of Cambridge (Department of Zoology), the MRC Laboratory of Molecular Biology, and Google Research: https://male-cns.janelia.org/

Both the original dataset and these adapted files are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Please cite:

> Berg, S. et al. *Sexual dimorphism in the complete Drosophila male central nervous system connectome.*
> Cell (2026), https://doi.org/10.1016/j.cell.2026.08.015. Preprint: *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, bioRxiv (2025), https://doi.org/10.1101/2025.10.09.680999

## Changes from the original

These files were produced from the flat-connectome release (`minconf-0.5` annotations, neurotransmitters, and connection weights) by `pipeline/build_connectome.py`. The changes:

- **Neurons kept:** only neurons with an assigned superclass, excluding glia.
- **Edges kept:** all edges between retained neurons.
- **Order and encoding:** neurons are reordered, and edges are re-encoded as compressed sparse rows (delta-coded varints, gzip).
- **Signs:** synapses are signed by a single neurotransmitter label per neuron.
- **Added fields:** per-neuron values computed by this project, namely eye-column viewing directions and connectome-derived receptive fields.

The adapted data is provided as is, without warranties. It is not endorsed by the original authors.
