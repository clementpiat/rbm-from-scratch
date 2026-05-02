## On SARS-CoV-2

* **SARS-CoV-2** is a coronavirus which contains a RNA genome, proteins, and an enveloppe
* One of those proteins is the SARS-CoV-2 **Spike**, which function is to bind to the ACE2 receptor of host cells (basically a protein attached to the membrane) in order to enter their cytoplasm. Once the cells are infected, SARS-CoV-2 can produce new proteins and replicate
* There is a specific slice of this Spike protein that we call the **Receptor-Binding Domain (RBD)**—it essentially contains residues from ~331 to ~531—which is highly variable and responsible for binding to ACE2

## MSA & DMS

* **Multi Sequence Alignment (MSA)** are alignments of biological sequences (for instance homologous proteins)
    * basically the result is a table whose rows are sequences of nucleotides or amino-acids, interleaved with blank spaces / gaps, and columns are such that "many columns are almost-constant vectors"
    * alignment is an NP-complete problem
    * a common tool / technique used is mafft (multiple alignment using fast Fourier transform)
    * MSA results are often stored in the FASTA format (a txt format delimiting sequences with '>')
* **Deep Mutational Scanning (DMS)** is a process in which you take a protein and apply all possible single amino acid mutation (length x 21 possibilities), and then observe the function of the mutant

&rarr; We will train an RBM on an MSA of the RBD domain, then we will measure the correlation between the DMS binding results and the energy of our RBM. The reasoning is the following: if the mutant doesnt't bind to ACE2, it won't survive because it won't enter the host cell; if it binds, it might survive; thus we expect the measure binding property to be correlated to our model's fitness / energy.
## Others

* **Residue** is what we call an amino-acid as part of a protein chain
