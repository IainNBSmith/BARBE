# BARBE

This repository contains the code for "Explaining Decisions of Black-box Models using BARBE" paper [1]. **B**lack-box **A**ssocaition **R**ule-**B**ased **E**xplanation (**BARBE**) is a model-independent framework that can explain the decisions of any black-box classifier for tabular datasets with high-precision.

We rely on the rules provided by SigDirect [1'] (the associative classifier we leveraged in BARBE) to explain each decision of a black-box model. 

The repository stores local versions of existing packages [LIME](https://github.com/marcotcr/lime), [VAE-LIME](https://github.com/domenVres/Robust-LIME-SHAP-and-IME), [S-LIME](https://github.com/ZhengzeZhou/slime), [LORE](https://github.com/riccotti/LORE), and [Anchor](https://github.com/marcotcr/anchor) for the purposes of conducting experiments. The directory barbe contains all code for BARBE including experiments used in faithful evaluations [2] and counterfactual generation [3]. Alongside a still in-development front-end (BARBiE).

# The organization of the barbe directory
* dataset - directory containing each dataset (cleaned and unclean) used in different experiments
* experiments - directory containing experiments for [2] and [3] along with helper code for averaging results. Contains subdirectories: Results (raw result tables) and SummarizedResults (tables of averaged results)
* poster - directory containing all posters presented related to BARBE
* pretrained - directory containing pre-trained models from each article for reproducibility, loaded using the [dill package](https://pypi.org/project/dill/)
* styles - directory containing style objects used by implementation of front-end application built using the [shiny package](https://shiny.posit.co/py/)
* tests - directory containing code for various tests for BARBE and front-end wrapper functions used for other explainers
* utils - general directory of utilies applied in BARBE and experiments
  * {anchor, fieap, lime, lore}_interface.py - wrappers for explainer models that have various data input formats and model formats, used to homogenize their input and output
  * bbmodel_interface.py - wrapper for black-box model to check compatability of scikit or torch based models ensuring that the data input/output formats from BARBE are consistent with the black-box
  * dummy_interfaces.py - dummy explainer that outputs default baseline explanations to compare to
  * experiment_datasets.py - utility for loading datasets for experiments
  * sigdirect_interface.py - wrapper for sigdirect to transform data into sigdirect's expected formats of input/output and handle broad hyperparameters used in BARBE
  * simulation_dataset.py - utility for producing simulated data for diverse experiments on BARBE and other explainers
  * visualizer_javascript.js - javascript code used by visualizer to handle unique reactive elements
  * visualizer_utility.py - utilities to get required details from data, black-boxes, and the explainer not handled directly by the visualizer
* counterfactual.py - defines the object BarbeCounterfactual that handles generating counterfactuals from data as requested by BARBE
* discretizer.py - defines the object CategoricalEncoder used in many instances to handle encoding categorical values into one-hot or ordinal values for BARBE and wrapped explainers
* explainer.py - defines the object BARBE that takes data and learns local explanations orchistrating each other utility
* perturber.py - defines the objects BarbePerturber and ClassBalancedPerturbed which produce perturbations of an input piece of data output in the pandas DataFrame format
* visualizer.py - contains code for running the visual front-end for BARBE using shiny

# Work in this Repository

[[1 - Link to BARBE Paper]](https://link.springer.com/chapter/10.1007/978-3-031-39821-6_6) 

[[2 - Link to Faithful Evaluations]](https://caiac.pubpub.org/pub/gs2ywmlt/release/1?readingCollection=e093cfd6) Designed to provide true performance comparisons by weighing fidelity of a surrogate against the black-box based on the distance (k-Nearest Neighbors and Euclidean distance) of samples from the explained sample. Also included improvements to BARBE to produce perturbations more faithful to the original data to improve training.

[3 - Link to Counterfactual Generation (currently in review)] Using BARBE as a source of counterfactual rules we produce and enhance generated counterfactuals by adding negative features (producing mixed postive and negative rules) and sorting the order of rules to apply based on general importance or specific importance linked to the sample. We compare to LORE as another surrogate based explainer and [DiCE](https://github.com/interpretml/DiCE) as an explainer specifically designed to make counterfactuals.

[[1' - Link to SigDirect paper]](https://content.iospress.com/articles/intelligent-data-analysis/ida163141)

