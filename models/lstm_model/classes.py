from keras.models import load_model as l
import os

def load_model(path = 'lstm_60_30_d100_d10_100epochs.keras'):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    model = l(os.path.join(module_dir,path))
    return model