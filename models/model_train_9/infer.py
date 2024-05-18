import numpy as np
from classes import get_model
from time import time

model = get_model()
model.load_weights('1DCNN_Transformer_L-dim192_train9_1405-100epochs.weights.h5')
start = time()
input = np.random.randn(16,30,258)
output = model.predict(input)
end = time()
print('Inference_time: ',end-start)