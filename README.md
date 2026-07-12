# CNN

Image classification on the Intel Image Classification dataset (buildings, forest, glacier, mountain, sea, street) - built and compared a few CNN versions to see what actually improves performance.

## Dataset

[Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) - natural scene images across six classes: buildings, forest, glacier, mountain, sea, street. Uses the `seg_train` and `seg_test` folders (the `seg_pred` folder isn't needed for this). `seg_train` gets split 80/20 into training and validation sets, and the final models are evaluated on `seg_test`.

## What's in here

Downloaded the dataset directly using the Kaggle API and loaded images with `ImageDataGenerator`, resizing to 150x150 and rescaling pixel values to 0-1. Built and trained three versions of a CNN, each an improvement on the last:

1. **Basic CNN** - one conv layer + pooling + dense layers, just to get a baseline
2. **Modified/Advanced CNN** - deeper network with more Conv2D and MaxPooling layers, batch normalization, and different filter counts
3. **Augmented training** - same advanced model but trained on augmented data (random rotation, zoom, flips, brightness changes) applied only to the training set, to help it generalize better

Also added Dropout and EarlyStopping at the end to reduce overfitting, so training stops automatically once validation performance stops improving instead of always running the full number of epochs.

Each version gets trained and evaluated on `seg_test`, and accuracy/loss curves are plotted after each one to compare how they performed against each other.

## Files

- `CNN.ipynb` - the notebook
- Uses the Intel Image Classification dataset from Kaggle (downloaded directly in the notebook using the Kaggle API)

## Running it

```
pip install tensorflow kaggle
```

You'll need a `kaggle.json` API key (from your Kaggle account settings) to download the dataset - the notebook has a Colab upload step for this. Written for Google Colab, adjust the paths if running locally.

## Notes

The jump from the basic CNN to the advanced one (batch norm + dropout + more layers) is where most of the accuracy improvement comes from. Data augmentation helps more with generalization (less overfitting) than raw accuracy, so it's most useful if training vs validation accuracy were diverging in the earlier versions.
