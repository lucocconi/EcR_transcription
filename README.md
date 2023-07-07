# EcR_transcription

Source codes for the paper "The ecdysone receptor promotes or suppresses proliferation according to ligand level". The codes describe the measurement and analysis of experimental data, as well as simulations of the theory described in the supplementary materials. 

The scripts "EcR_complex_statistics.nb" and "fit_to_reporter_data.py" can be used to re-generate the modelling figures appearing in the paper, and are associated with the models in SM Section I and II respectively. The data files "10ere.csv", "brk_subtr.csv" and "gbe.csv" contain experimental data from the 10ERE, Sil-10ERE and Enh-10ERE constructs used for fitting the thermodynamic model and for parameter estimation. 

The script "rnd_gene_generator.py" was used to generate the synthetic gene expression data shown in Fig.S4J, based on the data in files "all_RNA_seq_genes.csv", "genes_changing_norm.csv" and "genes_monotonic.csv" (the latter two of which are subsets of the first dataset). 
