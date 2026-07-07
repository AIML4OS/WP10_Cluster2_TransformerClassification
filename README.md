# Cluster2 - TransformerClassification

<a href="https://datalab.sspcloud.fr/launcher/ide/rstudio?name=Dev_Transformer_Classification&version=2.4.5&s3=region-79669f20&init.personalInit=«https%3A%2F%2Fraw.githubusercontent.com%2FAIML4OS%2FWP10_Cluster2_TransformerClassification%2Frefs%2Fheads%2Fmain%2Fsspcloud%2Finit-trainees.sh»&init.personalInitArgs=«exercises%2Fexercise1.qmd»&git.repository=«https%3A%2F%2Fgithub.com%2FAIML4OS%2FWP10_Cluster2_TransformerClassification»&git.branch=«main»&autoLaunch=true" target="_blank" rel="noopener" data-original-href="https://datalab.sspcloud.fr/launcher/ide/rstudio?name=Dev_Transformer_Classification&version=2.4.5&s3=region-79669f20&init.personalInit=«https%3A%2F%2Fraw.githubusercontent.com%2FAIML4OS%2FWP10_Cluster2_TransformerClassification%2Frefs%2Fheads%2Fmain%2Fsspcloud%2Finit-trainees.sh»&init.personalInitArgs=«exercises%2Fexercise1.qmd»&git.repository=«https%3A%2F%2Fgithub.com%2FAIML4OS%2FWP10_Cluster2_TransformerClassification»&git.branch=«main»&autoLaunch=true"><img src="https://custom-icon-badges.demolab.com/badge/SSP%20Cloud-Launch_with_RStudio-blue?logo=vsc&amp;logoColor=white" alt="Onyxia"></a>

## Goal

This repository aims to show Austria's use of text classification model, for the classification of ISCO codes, based on a transformer model trained from scratch.
This work was carried out within Cluster 2 of Work Package 10 "Text-to-Code" (WP10) of the AIMLL4OS project.

You can find more information about WP10 on  the [CROS website](https://cros.ec.europa.eu/book-page/aiml4os-wp10-text-code-experiences-and-potential-use-aiml-classifying-and-coding), its [GitHub Repository](https://github.com/AIML4OS/WP10) and its dedicated [GitHub Pages](https://aiml4os.github.io/WP10/). 

## Report Overview
The report provides an introduction to transformer-based models for text classification in official statistics used at Statistics Austria. It explains the underlying transformer architecture, discusses practical considerations for training and evaluation, and outlines best practices for applying these models to automated coding tasks such as assigning NACE, ISCO, or ISCED classifications.

## Code Structure
The `exercises/` folder contains practical examples for training and evaluating transformer-based text classification models using publicly available data from Statistics Austria's [classification data bank](https://www.statistik.at/KDBWeb/kdb_VersionAuswahl.do). The code scripts in the `exercises/R` contain implementations for data pre- and postprocessing, model building and training.

## Runnable Example
The file `exercises/run_transformer_model.qmd` guides users through the complete workflow, including data preprocessing, tokenization, model training, inference, and evaluation, providing reproducible examples that complement the concepts introduced in the accompanying report.


## Useful Documentation and Links
The runnable toy example can be run on an SSPCloud instance via this [link](https://datalab.sspcloud.fr/launcher/ide/rstudio?name=Dev_Transformer_Classification&version=2.4.5&s3=region-79669f20&init.personalInit=«https%3A%2F%2Fraw.githubusercontent.com%2FAIML4OS%2FWP10_Cluster2_TransformerClassification%2Frefs%2Fheads%2Fmain%2Fsspcloud%2Finit-trainees.sh»&init.personalInitArgs=«exercises%2Fexercise1.qmd»&git.repository=«https%3A%2F%2Fgithub.com%2FAIML4OS%2FWP10_Cluster2_TransformerClassification»&git.branch=«main»&autoLaunch=true)

The report to our work can be found [here](https://aiml4os.github.io/WP10_Cluster2_TransformerClassification/).

This repository follows the AIML4OS [template](https://aiml4os.github.io/training-material-starting-pack/) provided by the [Work Package 6](https://cros.ec.europa.eu/book-page/aiml4os-wp6-knowledge-repository-and-training-materials).
