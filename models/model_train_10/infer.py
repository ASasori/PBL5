import numpy as np
from classes import load_model
from time import time

model = load_model()
start = time()
input = np.random.randn(16,30,258)
output = model.predict(input)
end = time()
print('Inference_time: ',end-start)