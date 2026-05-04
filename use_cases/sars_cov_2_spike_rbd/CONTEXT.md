**WARNING**: don't take the following for granted, I'm not a biologist, I'm just trying to put words on what I understood so far.

## On SARS-CoV-2

* **SARS-CoV-2** is a specific coronavirus; it contains an RNA genome, proteins, and an envelope
* One of those proteins is the SARS-CoV-2 **Spike**, whose function is to bind to the ACE2 receptor of host cells (basically a protein attached to the membrane of the cell) and fuse the cell's membrane with the virus' envelope. Once the cells are infected, SARS-CoV-2 can replicate using the cell's machinery.
* There is a specific "slice" of this Spike protein that we call the **Receptor-Binding Domain (RBD)**—it corresponds to residues ~319–541—which is highly variable and responsible for binding to ACE2

## MSA & DMS

* **Multiple Sequence Alignment (MSA)** are alignments of biological sequences (for instance homologous proteins)
    * basically the result is a table whose rows are sequences of nucleotides or amino-acids, interleaved with gaps, and columns are such that "many columns are almost-constant vectors"
    * alignment is an NP-complete problem
    * a common tool / technique used is mafft (multiple alignment using fast Fourier transform)
    * MSA results are often stored in the FASTA format (a txt format delimiting sequences with '>')
* **Deep Mutational Scanning (DMS)** is a process in which you take a protein and apply all possible single amino-acid substitution (length x ~19 possibilities), and then measure a phenotype on the mutant

&rarr; We will train an RBM on an MSA of the RBD domain, then we will measure the correlation between the DMS binding results and the energy of our RBM. The reasoning is the following: if the mutant doesnt't bind to ACE2, it won't survive because it won't enter the host cell; if it binds, it might survive; thus the hypothesis is that the measured binding property would be negatively correlated with our model's energy.

### Downloading the fasta file

* in UniProt I searched for "Betacoronavirus AND gene:s" to get a list of proteins and download a fasta file
* then I needed to align these proteins

&rarr; you can install mafft with conda
```shell
conda install conda-forge::mafft
```
&rarr; then you can run something like
```shell
mafft uniprotkb_Betacoronavirus_AND_gene_s_2026_05_03.fasta > uniprotkb_Betacoronavirus_AND_gene_s_2026_05_03_aligned.fasta
```

## Others

* **Residue** is what we call an amino-acid as part of a protein chain

## Regarding RBM

We now have categorical visible units (instead of binary units). Hidden units remain binary.

&rarr; This changes the probing and sampling behavior for the visible units.
