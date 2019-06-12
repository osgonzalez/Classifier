import LibsPipDownloader

#Load DataSet
from sklearn.datasets import load_boston
boston_dataset = load_boston()

print("\n\n\n\n\n\n")

print(boston_dataset.keys())
print(boston_dataset['feature_names'])
print(boston_dataset['data'])
print(boston_dataset['target'])




print("Done!")
