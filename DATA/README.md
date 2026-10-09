# Dataset attribution

The CSVs were supplied with the original ConvXAI-GEN prototype. They retain the prototype's column names and preprocessing conventions.

- **Iris**: R. A. Fisher, Iris, UCI Machine Learning Repository, DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76). [UCI dataset page](https://archive.ics.uci.edu/dataset/53/iris). UCI lists the dataset as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This copy uses `Species` as the label and centimetre feature names. `load_data.py` removes the extra `Id` column before modelling.
- **Wine Quality**: P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis, UCI Machine Learning Repository. [UCI dataset page](https://archive.ics.uci.edu/dataset/186/wine+quality). UCI lists the dataset as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This prototype combines wine types and uses the added `color` column as its classification target. It does **not** model wine-quality ratings as the target; other numeric columns, including quality if present, remain predictors.

The source files are preserved as supplied rather than presented as byte-identical UCI downloads. Manifest hashes identify the precise local inputs used to generate the included explanations. No endorsement by the dataset authors is implied. Dataset rights remain separate from the repository's MIT code license.
