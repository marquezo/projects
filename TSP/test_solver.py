
import pytspsa

from torch.utils.data import DataLoader
import numpy as np
import pickle

tsp20_dataset = pickle.load(open("test_data_set_20", "rb"))

tsp20_loader = DataLoader(tsp20_dataset, batch_size=1, shuffle=False, num_workers=1)

tours = []
paths = []

for batch_id, sample_batch in enumerate(tsp20_loader):
        sample_batch = sample_batch.squeeze().numpy()

        c= sample_batch.T
        solver = pytspsa.Tsp_sa()
        solver.set_num_nodes(20)
        solver.add_by_coordinates(c)
        solver.set_t_v_factor(4.0)

        # solver.sa() or sa_auto_parameter() will solve the problem.
        solver.sa_auto_parameter(12)

        # getting result
        solution = solver.getBestSolution()

        tours.append(solution.getlength())
        paths.append(solution.getRoute())

        print ("batch id {}, result: {}".format(batch_id, solution.getlength()))

tf = open('results_20e', 'wb')
pickle.dump(tours, tf)
tf.close()

tf = open('results_20e_paths', 'wb')
pickle.dump(paths, tf)
tf.close()

print "Average length of tour: ", (np.mean(tours))

