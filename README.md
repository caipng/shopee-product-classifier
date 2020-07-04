# Shopee Product Classifier

A CNN built with Keras for Shopee's Product Detection competition.

This repo is the final stage where training and validation data is combined to train the final model, hence the lack of a validation dataset.

1.  `main.py` downloads and processes the dataset required

2.  `train_1.py` trains the first model: `ResNet50` base model frozen

    - `n_epochs` and `n_cycles` can be adjusted as fit

3.  `train_2.py` trains the second model: `conv1-3` are frozen, `conv4-5` along with last layer are fine tuned

    - as in `model1`, `n_epochs` and `n_cycles` can be adjusted as fit

4.  Select the appropriate models in the `model/` directory to be used for the ensemble, and move them to `model/final`

5.  `train_ensemble.ipynb` trains the `XGBClassifier` to be used for final prediction

6.  `submission.py` generates `submission.csv` (\*\*`submission.py` is bugged; interrupt the script once batch number reaches 0 ; `submission_fix.ipynb` generates the final file ready for upload to Kaggle)

### Results

Public Leaderboard - 229 of 770 with 0.73977 accuracy (as at 1800 4 Jul)

_first try at image classifications/CNNs so its ok i guess_ 🤷

_this is a basic model with transfer learning, data augmentation, cyclic learning rates and snapshot ensembling; tried things like activity regularizers and BlurPooling but didnt seem to improve validation accuracy; there are definitely a plethora of ideas that were left untested due to time and computing power constraints_
