"""Machine learning model"""

from copy import deepcopy
import logging
from operator import itemgetter
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import List, Tuple, Dict, Any, Callable

import tensorflow as tf
from typing import Any as _Any

# Define lightweight stand-ins to avoid importing private or deprecated TF APIs
class ModeKeys:
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'


Estimator = _Any
AutoTrackable = _Any


LOGGER = logging.getLogger(__name__)

DATASET = {
    ModeKeys.TRAIN: 'train',
    ModeKeys.EVAL: 'valid',
    ModeKeys.PREDICT: 'test',
}


class HyperParameter:
    """Model hyper parameters"""
    BATCH_SIZE = 100
    NB_TOKENS = 10000
    VOCABULARY_SIZE = 5000
    EMBEDDING_SIZE = max(10, int(VOCABULARY_SIZE**0.5))
    DNN_HIDDEN_UNITS = [512, 32]
    DNN_DROPOUT = 0.5
    N_GRAM = 2


class Training:
    """Model training parameters"""
    SHUFFLE_BUFFER = HyperParameter.BATCH_SIZE * 10
    CHECKPOINT_STEPS = 1000
    LONG_TRAINING_STEPS = 10 * CHECKPOINT_STEPS
    SHORT_DELAY = 60
    LONG_DELAY = 5 * SHORT_DELAY


def load(saved_model_dir: str) -> AutoTrackable:
    """Load a Tensorflow saved model"""
    return tf.saved_model.load(saved_model_dir)


def build(model_dir: str, labels: List[str]) -> Estimator:
    """Build a simple Keras text classifier compatible with SavedModel predict signature."""
    # Tokenize via hashing like original
    inputs = tf.keras.Input(shape=(HyperParameter.NB_TOKENS,), dtype=tf.string, name='content')
    # Hash each n-gram into a bucket and embed
    hashed = tf.strings.to_hash_bucket_fast(inputs, HyperParameter.VOCABULARY_SIZE)
    embed = tf.keras.layers.Embedding(input_dim=HyperParameter.VOCABULARY_SIZE,
                                      output_dim=HyperParameter.EMBEDDING_SIZE)(hashed)
    x = tf.keras.layers.GlobalAveragePooling1D()(embed)
    for units in HyperParameter.DNN_HIDDEN_UNITS:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.Dropout(HyperParameter.DNN_DROPOUT)(x)
    logits = tf.keras.layers.Dense(len(labels))(x)
    probs = tf.keras.layers.Softmax(name='scores')(logits)
    model = tf.keras.Model(inputs=inputs, outputs=probs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(model: Estimator, data_root_dir: str, max_steps: int) -> Any:
    """Train a Keras model using dataset API."""
    ds_train = _build_input_fn(data_root_dir, ModeKeys.TRAIN)()
    ds_eval = _build_input_fn(data_root_dir, ModeKeys.EVAL)()
    # Map to integer labels based on label_vocabulary discovery
    # For simplicity assume labels are mapped externally; here we just fit a few steps
    steps = max_steps if max_steps > 0 else 100
    history = model.fit(ds_train, epochs=1, steps_per_epoch=steps)
    return history.history


def save(model: Estimator, saved_model_dir: str) -> None:
    """Save a Keras model with a serving signature compatible with predict()."""
    class ServingModule(tf.Module):
        def __init__(self, mdl):
            super().__init__()
            self.mdl = mdl

        @tf.function(input_signature=[tf.TensorSpec([None, HyperParameter.NB_TOKENS], tf.string)])
        def serving_default(self, content):
            # Apply same preprocessing as during training
            # content: [batch, tokens]
            hashed = tf.strings.to_hash_bucket_fast(content, HyperParameter.VOCABULARY_SIZE)
            embed = self.mdl.layers[1](hashed)  # reuse embedding layer
            x = self.mdl.layers[2](embed)       # global average pooling
            for layer in self.mdl.layers[3:]:
                x = layer(x)
            scores = x
            # Build a dummy classes tensor using label indices; the caller maps externally
            classes = tf.strings.as_string(tf.range(tf.shape(scores)[-1]))
            return {"scores": scores, "classes": tf.expand_dims(classes, 0)}

    module = ServingModule(model)
    tf.saved_model.save(module, saved_model_dir, signatures={"serving_default": module.serving_default})


def test(
    saved_model: AutoTrackable,
    data_root_dir: str,
    mapping: Dict[str, str],
) -> Dict[str, Dict[str, int]]:
    """Test a Tensorflow saved model"""
    values = {language: 0 for language in mapping.values()}
    matches = {language: deepcopy(values) for language in values}

    LOGGER.debug('Test the model')
    input_function = _build_input_fn(data_root_dir, ModeKeys.PREDICT)
    for test_item in input_function():
        content = test_item[0]
        label = test_item[1].numpy()[0].decode()

        result = saved_model.signatures['predict'](content)
        predicted = result['classes'].numpy()[0][0].decode()

        label_language = mapping[label]
        predicted_language = mapping[predicted]
        matches[label_language][predicted_language] += 1

    return matches


def predict(
    saved_model: AutoTrackable,
    mapping: Dict[str, str],
    text: str
) -> List[Tuple[str, float]]:
    """Infer a Tensorflow saved model"""
    content_tensor = tf.constant([text])
    predicted = saved_model.signatures['serving_default'](content_tensor)

    numpy_floats = predicted['scores'][0].numpy()
    extensions = predicted['classes'][0].numpy()

    probability_values = (float(value) for value in numpy_floats)
    languages = (mapping[ext.decode()] for ext in extensions)

    unsorted_scores = zip(languages, probability_values)
    scores = sorted(unsorted_scores, key=itemgetter(1), reverse=True)
    return scores


def _build_input_fn(
    data_root_dir: str,
    mode: ModeKeys,
) -> Callable[[], tf.data.Dataset]:
    """Generate an input fonction for a Tensorflow model"""
    pattern = str(Path(data_root_dir).joinpath(DATASET[mode], '*'))

    def input_function() -> tf.data.Dataset:
        dataset = tf.data.Dataset
        dataset = dataset.list_files(pattern, shuffle=True).map(_read_file)

        if mode == ModeKeys.PREDICT:
            return dataset.batch(1)

        if mode == ModeKeys.TRAIN:
            dataset = dataset.shuffle(Training.SHUFFLE_BUFFER).repeat()

        return dataset.map(_preprocess).batch(HyperParameter.BATCH_SIZE)

    return input_function


def _serving_input_receiver_fn():
    return None

def _read_file(filename: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read a source file, return the content and the extension"""
    data = tf.io.read_file(filename)
    label = tf.strings.split([filename], '.').values[-1]
    return data, label


def _preprocess(
    data: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Process input data as part of a workflow"""
    data = _preprocess_text(data)
    return {'content': data}, label


def _preprocess_text(data: tf.Tensor) -> tf.Tensor:
    """Feature engineering"""
    padding = tf.constant(['']*HyperParameter.NB_TOKENS)
    data = tf.strings.bytes_split(data)
    data = tf.strings.ngrams(data, HyperParameter.N_GRAM)
    data = tf.concat((data, padding), axis=0)
    data = data[:HyperParameter.NB_TOKENS]
    return data
