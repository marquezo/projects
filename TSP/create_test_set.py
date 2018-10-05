from TSPDataset import TSPDataset
import pickle

test_dataset = TSPDataset(10, 1000)

file = open('test_data_set_10', 'wb')
pickle.dump(test_dataset, file)
file.close()
