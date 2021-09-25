'''
The TensorFlow Lite Model Maker library simplifies
the process of adapting and converting a TensorFlow model
to particular input data when deploying this model for on-device ML applications.
'''

import numpy as np
import os

from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.text_classifier import AverageWordVecSpec
from tflite_model_maker.text_classifier import DataLoader

import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')

'''TensorFlow Lite Model Maker currently supports MobileBERT, 
averaging word embeddings and BERT-Base models.
'average_word_vec', 'mobilebert_classifier'	, 'bert_classifier'
'''
# choose a text classification model architecture.
spec = model_spec.get('average_word_vec')

train_data = DataLoader.from_csv(
      filename='train.csv',
      text_column='sentence',
      label_column='label',
      model_spec=spec,
      is_training=True)

test_data = DataLoader.from_csv(
      filename='dev.csv',
      text_column='sentence',
      label_column='label',
      model_spec=spec,
      is_training=False)

# train the TensorFlow model with the training data.
model = text_classifier.create(train_data, model_spec=spec, epochs=10)

# Examine the detailed model structure.
model.summary()

# Evaluate the model with the test data.
loss, acc = model.evaluate(test_data)

# Step 5. Export as a TensorFlow Lite model.
model.export(export_dir='average_word_vec/')

# Model Maker supports multiple post-training quantization options using QuantizationConfig as well.
# # Let's take float16 quantization as an instance

config = QuantizationConfig.for_float16()
model.export(export_dir='.', tflite_filename='model_fp16.tflite', quantization_config=config)