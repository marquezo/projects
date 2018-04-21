from TSPDataset import TSPDataset
import pickle

test_dataset = TSPDataset(20, 1000)

file = open('test_data_set', 'wb')
pickle.dump(test_dataset, file)
file.close()
