import pandas as pd

location = '../schizophrenia_dataset.csv'
n_training = 8000
total = 10000

def splitLabel(data):
    data_features = data.drop('Diagnosis', axis="columns").to_numpy()
    data_label = data['Diagnosis'].to_numpy()

    return (data_label, data_features)

def getData(): 
    data = pd.read_csv(location)

    training = splitLabel(data.loc[1:n_training])
    test = splitLabel(data.loc[n_training:total])

    return (training, test)



