# Shopee Product Classifier

A CNN built with Keras for Shopee's Product Detection competition. This repo is the final stage where training and validation data is combined to train the final model, hence the lack of a validation dataset.

1. `python main.py` to download and process the dataset required
2. `python train_1.py` to train the first model (`ResNet50` base model frozen); `n_epochs` and `n_cycles` can be adjusted as fit
3. `python train_2.py` to train the second model (`conv1-3` are frozen, `conv4-5` along with last layer are fine tuned); as in `model1`, `n_epochs` and `n_cycles` can be adjusted as fit
4. Select the appropriate models in the `model/` directory to be used for the ensemble, and move them to `model/final`
5. `train_ensemble.ipynb` can then be used to train the `XGBClassifier` to be used for final prediction
6. Lastly, `python submission.py` is ran to generate `submission.csv`
   - `submission.py` is not fully functional, it is needed to stop the script once the batch number becomes 0 again, and `submission_fix.ipynb` is used to generate the final file ready for upload to Kaggle
