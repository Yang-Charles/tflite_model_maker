import tensorflow as tf
import os
import pandas as pd
# data_dir = tf.keras.utils.get_file(
#       fname='SST-2.zip',
#       origin='https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
#       extract=True)

data_dir = os.path.join(os.path.dirname(__file__), 'data/SST-2')


'''
we will load the dataset into a Pandas dataframe and 
change the current label names (0 and 1) to a more human-readable 
ones (negative and positive) and use them for model training.
'''
def replace_label(original_file, new_file):
  # Load the original file to pandas. We need to specify the separator as
  # '\t' as the training data is stored in TSV format
  df = pd.read_csv(original_file, sep='\t')

  # Define how we want to change the label name
  label_map = {0: 'negative', 1: 'positive'}

  # Excute the label change
  df.replace({'label': label_map}, inplace=True)

  # Write the updated dataset to a new file
  df.to_csv(new_file)

# Replace the label name for both the training and test dataset. Then write the
# updated CSV dataset to the current folder.
replace_label(os.path.join(data_dir, 'train.tsv'), data_dir + '/train.csv')
replace_label(os.path.join(data_dir, 'dev.tsv'), data_dir + '/dev.csv')


# Choose a text classification model architecture.
spec = model_spec.get('average_word_vec')
# Step 2. Load the training and test data, then preprocess them according to a specific model_spec.
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