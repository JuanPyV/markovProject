from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix
import itertools
import os
import glob
import os.path as path
"""holaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"""



class HMMTrainer(object):
   def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
      self.model_name = model_name
      self.n_components = n_components
      self.cov_type = cov_type
      self.n_iter = n_iter
      self.models = []
      if self.model_name == 'GaussianHMM':
         self.model = hmm.GaussianHMM(n_components=self.n_components,        covariance_type=self.cov_type,n_iter=self.n_iter)
      else:
         raise TypeError('Invalid model type') 

   def train(self, X):
      np.seterr(all='ignore')
      self.models.append(self.model.fit(X))
      # Run the model on input data
   def get_score(self, input_data):
      return self.model.score(input_data)



hmm_models = []
input_folder = 'genres/'
# Parse the input directory
for dirname in os.listdir(input_folder):
    # Get the name of the subfolder
   subfolder = os.path.join(input_folder, dirname)
   if not os.path.isdir(subfolder):
          continue
      # Extract the label
   label = subfolder[subfolder.rfind('/') + 1:]
    # Initialize variables
   X = np.array([])
   y_words = []
   # Iterate through the audio files (leaving 1 file for testing in each class)
   for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
      # Read the input file
      filepath = os.path.join(subfolder, filename)
      sampling_freq, audio = wavfile.read(filepath)
      # Extract MFCC features
      mfcc_features = mfcc(audio, sampling_freq)
      # Append to the variable X
      if len(X) == 0:
         X = mfcc_features
      else:
         X = np.append(X, mfcc_features, axis=0)

            # Append the label
      y_words.append(label)    
   print('X.shape =', X.shape)
    # Train and save HMM model
   hmm_trainer = HMMTrainer(n_components=10)
   hmm_trainer.train(X)
   hmm_models.append((hmm_trainer, label))
   hmm_trainer = None

input_folder = 'test/'
real_labels = []
pred_labels = []
for dirname in os.listdir(input_folder):
   subfolder = os.path.join(input_folder, dirname)
   if not os.path.isdir(subfolder):
      continue
   # Extract the label
   label_real = subfolder[subfolder.rfind('/') + 1:]

   for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
      real_labels.append(label_real)
      filepath = os.path.join(subfolder, filename)
      sampling_freq, audio = wavfile.read(filepath)
      mfcc_features = mfcc(audio, sampling_freq)
      max_score = -9999999999999999999
      output_label = None
      for item in hmm_models:
         hmm_model, label = item
         score = hmm_model.get_score(mfcc_features)
         if score > max_score:
            max_score = score
            output_label = label
pred_labels.append(output_label)