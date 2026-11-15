# Text Language Identification

## Project overview

This project uses deep learning to identify the language of a text sample. The
model reads a piece of text and returns a probability for each supported
language. The language with the highest probability is used as the predicted
label.

Language identification is useful in applications such as multilingual search,
content routing, document organization, translation workflows, and automatic
language selection in user interfaces. The project focuses on a supervised
classification approach: each training example contains the original text and
its known language label.

The complete experiment is implemented in
[`Text_Language_Identification.ipynb`](./Text_Language_Identification.ipynb).
The notebook includes data inspection, preprocessing, model training,
evaluation, visualizations, and example predictions.

## Dataset

The notebook loads the data from `dataset.csv`. The expected schema is:

| Column | Description |
| --- | --- |
| `Text` | The text that should be classified |
| `language` | The language label associated with the text |

The recorded dataset contains 22,000 rows and 22 language classes. The classes
are Arabic, Chinese, Dutch, English, Estonian, French, Hindi, Indonesian,
Japanese, Korean, Latin, Persian, Portugese, Pushto, Romanian, Russian,
Spanish, Swedish, Tamil, Thai, Turkish, and Urdu.

Before training, the notebook:

1. Removes rows with a missing text or language value.
2. Converts language labels to strings and removes surrounding whitespace.
3. Removes duplicate text/label pairs.
4. Examines the class distribution and text lengths.
5. Uses stratified splitting so that each dataset partition maintains a
   representative distribution of languages.

The data is divided into 70% training data, 15% validation data, and 15% test
data. The validation set is used to monitor training, while the test set is
kept for the final evaluation.

## Text preprocessing

The text is converted into integer sequences using TensorFlow/Keras
tokenization. Each distinct token receives an integer index, and unknown words
are represented using an out-of-vocabulary token. Sequences are padded or
truncated to a maximum length of 600 tokens so that they can be supplied to
the neural network in fixed-size batches.

The label column is converted to integer class IDs with a label encoder. The
encoder is also saved so that predictions made after training can be converted
back to the original language names.

## Model architecture

The classifier is a sequential TensorFlow/Keras model with the following
stages:

1. **Embedding layer** - maps token IDs into 128-dimensional dense vectors.
2. **LSTM layer** - processes the sequence and learns context-dependent
   patterns in the text.
3. **Dropout layer** - applies a dropout rate of 0.3 to reduce overfitting.
4. **Dense layer** - contains 64 ReLU units for feature transformation.
5. **Output layer** - contains 22 softmax units, one for each language class.

The model is trained with a batch size of 64 for nine configured epochs. The
best checkpoint is selected using validation loss, rather than automatically
using the final epoch. This helps retain the version that generalizes best to
unseen examples.

## Results

The recorded run achieved the following results on the held-out test set:

| Metric | Result |
| --- | ---: |
| Test accuracy | 96.04% |
| Test loss | 0.1580 |
| Macro precision | 0.96 |
| Macro recall | 0.96 |
| Macro F1-score | 0.96 |
| Test examples | 3,279 |

The class-level results show strong performance for languages such as Arabic,
Korean, Thai, Turkish, and Urdu. Some visually or lexically similar classes,
including English, Dutch, Spanish, Portuguese, and Latin, are more difficult
to separate. The notebook includes a classification report and a 22 x 22
confusion matrix to make these errors easier to analyze.

The notebook also displays misclassified examples with their actual language,
predicted language, and prediction confidence. These examples are useful for
understanding whether an error is caused by short input, mixed-language text,
shared vocabulary, transliteration, or an unusual writing style.

## Generated files

During execution, the notebook writes reusable model artifacts to the
`models/` directory:

- `best_language_lstm.keras` - checkpoint with the best validation loss.
- `language_lstm.keras` - saved model after training.
- `tokenizer.json` - vocabulary and tokenization configuration.
- `label_encoder.pkl` - mapping between class IDs and language names.

The `documentation_output/` directory contains the generated project report
and figures, including the class distribution, text-length distribution,
training accuracy, model architecture, and confusion matrix.

## How to run

1. Place `dataset.csv` in the project directory.
2. Open `Text_Language_Identification.ipynb` in Jupyter Notebook, JupyterLab,
   or Google Colab.
3. Run the cells from top to bottom so that preprocessing objects are created
   before the model is trained.
4. Review the validation curves and evaluation metrics after training.
5. Use the saved tokenizer, label encoder, and model together when making
   predictions on new text.

The notebook uses Python libraries including pandas, NumPy, scikit-learn,
Matplotlib, Seaborn, and TensorFlow. Training a neural network can take
several minutes or longer depending on the available CPU/GPU.

## Example prediction workflow

For a new text sample, the same preprocessing used during training must be
applied:

1. Load the tokenizer and model.
2. Convert the input text into a token sequence.
3. Pad the sequence to 600 tokens.
4. Run the sequence through the model.
5. Select the class with the largest softmax probability.
6. Decode the class ID with the saved label encoder.

Using the original tokenizer and label encoder is important. Creating new
preprocessing objects at prediction time can change token IDs or class IDs and
produce incorrect results.

## Limitations and future improvements

The model is evaluated on data drawn from the same general distribution as the
training data. Performance may be lower on very short text, noisy text,
code-switched text, transliterated text, dialects, or languages that are not
included in the 22 classes. A prediction should therefore be treated as a
classification estimate rather than a guaranteed linguistic analysis.

Possible future improvements include adding more training examples, balancing
the classes, testing character-level features, comparing the LSTM with a
Transformer-based model, calibrating confidence scores, and adding an
explicit “unknown language” or low-confidence result for unsupported input.
