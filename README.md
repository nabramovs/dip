# DIP: Disease Inheritance Patterns
Disease Inheritance Patterns (DIP) is a gene level metric which can aid dominant and recessive Mendelian disease genes discovery. 

### Project Description

Metrics that predict whether loss or disruption of one or both gene copies can cause disease are useful to aid the discovery of novel autosomal dominant (AD) and recessive (AR) genes. Recent gene constraint studies showed that gene intolerance to variation continuously correlated with disease inheritance patterns, but the most tolerant genes were deficient in both dominant and recessive genes and, therefore, could be segregated into a third “non-disease” causing gene group. However, existing supervised machine learning solutions to this problem were built on the assumption that genes had to be classified into two groups (e.g. AD/AR or haploinsufficient/haplosufficient), and predictions were difficult to interpret in ambiguous cases with no clear mode of inheritance. Here, we present a novel gene level metric that continuously ranks 15,794 autosomal genes by Disease Inheritance Patterns (DIP), which was developed by combining multiple supervised machine learning models with gene variation intolerance metrics. DIP performance was comparable with existing metrics in distinguishing AD from AR genes. However, it more effectively prioritised disease genes in general, resulting in a more optimal ranking of genes across the spectrum of disease inheritance patterns. The first and last five percentiles were significantly enriched with AD and cell non-essential genes (approximately 4.6 and 4.2 times, respectively), whereas AR genes were 3 times more frequently seen in the middle ranks (41-51%). Although, perfect categorical classification of genes by mode of inheritance might not be possible, continuous metrics can provide a better estimation of a gene’s predisposition to certain modes of inheritance, especially in ambiguous cases.

### Required Datasets
DIP analysis requires local version of gnomAD v2.0.1 database, specifically the following collections: _exome_variants_, _genome_variants_, _exons_, _genes_, _transcripts_. Instructions how to install gnomAD database can be found here:
https://github.com/macarthur-lab/gnomad_browser

Additionally, it requires following datasets:
- The HGNC approved gene symbols can be obtained from
https://www.genenames.org/download/statistics-and-files/
- The Gene Discovery Informatics Toolkit (GDIT) supplementary dataset can be obtained from
https://www.nature.com/articles/s41525-019-0081-z
- The Ensembl gene-transcript-protein ids to HGNC name and id mapping file for build GRCh37/hg19 can be obtained from
https://grch37.ensembl.org/biomart/martview/
- The STRING protein-protein interactions data can be obtained from
https://string-db.org/
- GeVIR, LOEUF, and VIRLoF scores can be obtained from supplementary data (Supplementary Table 2) at
https://www.nature.com/articles/s41588-019-0560-2
- UNEECON gene scores can be obtained from
https://psu.app.box.com/s/wur3td0dawju9qtvu7w8orkxu5ur0oo6/file/517942997406
- DOMINO v1 final, train and validation (from package) gene scores can be obtained from
https://wwwfbm.unil.ch/domino/download.html
- GnomAD Structural Variants dataset v2.1.1 can be obtained from
https://gnomad.broadinstitute.org/downloads
- GPP training gene lists and scores can be obtained from supplementary data (Material 7 and 8) at
https://doi.org/10.1007/s00439-019-02021-9
- Gene4Denovo dataset can be obtained from
http://www.genemed.tech/gene4denovo/uploads/Candidate_gene.txt
- The cell essential genes, can be obtained from MacArthur repository
https://github.com/macarthur-lab/gene_lists/blob/master/lists/NEGv1_subset_universe.tsv
- The cell non-essential genes, can be obtained from MacArthur repository
https://github.com/macarthur-lab/gene_lists/blob/master/lists/homozygous_lof_tolerant_twohit.tsv
- The olfactory genes, can be obtained from MacArthur repository
https://github.com/macarthur-lab/gene_lists/blob/master/lists/olfactory_receptors.tsv
- The list of severe haploinsufficient genes can be obtained from Supplementary data (Table S5a) at
https://www.biorxiv.org/content/10.1101/148353v1

**Note 1**: All datasets have to be placed into _./source_data_ directory, have names and formatted as defined at the top of _import_data.py_ script (can be changed there).

### Usage

Download all required datasets, place them in _./source_data_ folder.
To install required python modules run:
```
pip install -r requirements.txt
```
To import datasets into the local database run: 
```
python import_data.py
```
To create gene lists in the database for models training and evaluation run:
```
python gene_groups.py
```
To create gnomAD "Acceptor/Length" feature run:
```
python exon_splicing.py
```
To create other features run:
```
python features.py
```
To create STRING ML models (PPI-N1, PPI-N2, PPI-V-ADR, PPI-V-DND, used as features) run:
```
python string_ppi.py
```
To select best features and tune model parameters run:
```
python feature_selection.py
```
To obtain predictions from the best performing ADR (D_RT) and DND (DR_T) ML models, and merged them into DIP gene ranking run:
```
python models.py
```
To report models performance and produce all the tables run:
```
python evaluation.py
```
To produce all the figures run:
```
python figures.py
```
**Note**: Individual operations can be enabled/disabled via comments in the _main_ method in all scripts, check them before performing the analysis.
