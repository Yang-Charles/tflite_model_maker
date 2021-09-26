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
def model_select(model):
      spec = model_spec.get()
      return spec

def data_loader_train(data_dir, model_spec, is_training):
      train_data = DataLoader.from_csv(
            filename=data_dir+'/train.csv',
            text_column='sentence',
            label_column='label',
            model_spec=model_spec,
            is_training=is_training)
      return train_data

def data_loader_test(data_dir, model_spec, is_training):
      test_data = DataLoader.from_csv(
            filename=data_dir+'/dev.csv',
            text_column='sentence',
            label_column='label',
            model_spec=model_spec,
            is_training=is_training)
      return test_data

def text_classifier(train_data, test_data, spec, epochs, export_dir):
      # train the TensorFlow model with the training data.
      model = text_classifier.create(train_data, model_spec=spec, epochs=epochs)
      # Examine the detailed model structure.
      model.summary()
      # Evaluate the model with the test data.
      loss, acc = model.evaluate(test_data)
      # Step 5. Export as a TensorFlow Lite model.
      model.export(export_dir=export_dir)

      # Model Maker supports multiple post-training quantization options using QuantizationConfig as well.
      # # Let's take float16 quantization as an instance
      config = QuantizationConfig.for_float16()
      model.export(export_dir='result/average_word_vec/', tflite_filename='model_fp16.tflite', quantization_config=config)

if __name__ == '__main__':
      data_dir = os.path.join(os.path.dirname(__file__), 'data/SST-2')
      model_spec = model_select('bert_classifier')
      train_data = data_loader_train(data_dir, model_spec, True)
      test_data = data_loader_test(data_dir, model_spec, False)

      export_dir = 'result/average_word_vec/'
      text_classifier(train_data, test_data, model_spec, 10, export_dir)
      print("model train complete !")
