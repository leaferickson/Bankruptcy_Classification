import pickle
import pandas as pd
import numpy as np

# read in the model
means = np.array(pickle.load(open("means.pkl", "rb")))
stdevs = np.array(pickle.load(open("stdevs.pkl", "rb")))
model = pickle.load(open("model.pkl", "rb"))
data = np.array(pd.read_csv("test.csv"))


# create a function to take in user-entered amounts and apply the model
def bankruptcy_prediction(model = model):
    
    scaled_data = (data - means) / stdevs
    predictions = model.predict(scaled_data)

    dat2 = pd.DataFrame(data)
    dat2["prediction"] = predictions

    return dat2