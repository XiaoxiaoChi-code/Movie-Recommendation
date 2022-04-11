import numpy as np
import time
from util.dataProcessing import MovieLensHandler


class UANDORANDRecommender:
    def __init__(self, raw_data, data, num_of_function , num_of_tables , num_of_choice , poolsize , seed):
        self.raw_data = raw_data
        self.data = data
        (self.num_of_users, self.num_of_services) = data.shape
        self.num_of_functions = num_of_function
        self.num_of_choice = num_of_choice
        self.num_of_tables = num_of_tables
        self.lsh_family = []
        for i in range(self.num_of_tables):
            self.lsh_family.append(andorandLSH(self.num_of_functions))
            self.lsh_family[i].fit(self.num_of_services, poolsize,self.num_of_choice, seed + i)

    def classify(self):
        ref = []
        for i in range(self.num_of_functions):
            ref.append((1 << (self.num_of_choice * (i+1))) - (1 << (self.num_of_choice * i)))
        ref = np.array(ref)
        self.similarity_matrix = np.ones((self.num_of_users, self.num_of_users)).astype(int)

        for i in range(self.num_of_tables):
            hash_values = self.lsh_family[i].get_batch_hash_value(self.data.T)
            hash_values = hash_values.reshape((self.num_of_users,1)).astype(int)
            mat = hash_values ^ hash_values.T
            self.similarity_matrix[mat == 0] = 1

            for j in range(self.num_of_functions):

                mat_copy = np.array(mat)
                mat_copy = mat_copy & ref[j]
                mat[mat_copy == 0] = 1

            mat[mat != 1] = 0
            self.similarity_matrix = self.similarity_matrix & mat


    def evaluate(self, indices, threshold = 0):

        rmse = 0
        mae = 0
        num_of_similar = 0
        num_of_samples = 0
        for idx in indices:          
            #predicted_columns = np.multiply(self.data[idx] == 0, self.raw_data[idx]>0)
            predicted_columns = np.multiply(self.data[idx] == -1, self.raw_data[idx] > 0)
            data_for_predicted = self.data[self.similarity_matrix[idx] > threshold,:]
            num_of_similar += data_for_predicted.shape[0]
            data_for_predicted = data_for_predicted[:, predicted_columns>0]
            data_for_reference = self.raw_data[idx, predicted_columns>0]
            valid_counts = np.sum(data_for_predicted>0, axis=0)
            data_for_predicted = data_for_predicted[:, valid_counts>0]
            if (data_for_predicted.shape[1] == 0):
                continue
            data_for_reference = data_for_reference[valid_counts>0]
            valid_counts = valid_counts[valid_counts>0]
            data_for_predicted[data_for_predicted==-1] = 0
            rt = np.sum(data_for_predicted, axis=0)/valid_count
            error = np.sum(np.abs(rt - data_for_reference))
            mae += error
            rmse += error * error
            num_of_samples += rt.shape[0]
            if num_of_samples >0:
                return mae / num_of_samples, np.sqrt(rmse / num_of_samples)
            return -1, -1
        

if __name__ == '__main__':
    handler = MovieLensHandler()
    raw_data = handler.read_ratings()
    times = 5
    num_of_hash_tables = [10, 8, 6, 4]
    num_of_hash_functions = [4, 6, 8, 10]
    num_of_choice = [2, 3, 4, 5]

    for t in range(times):
        seed = t + 1
        num_of_test_samples = 500
        data = handler.erase_data(0.1, seed=seed)
        np.random.seed(seed)
        test_data = np.random.choice(data.shape[0], num_of_test_samples, replace=False)
        for (t_idx, table_number) in enumerate(num_of_hash_tables):
            for (f_idx, function_number) in enumerate(num_of_hash_functions):
                for (s_idx, subfunction_number) in enumerate(num_of_choice):
                    recommender = UANDORANDRecommender(raw_data, data, num_of_function=function_number, num_of_tables=table_number, num_of_choice=subfunction_number, poolsize=10, seed=seed)
                    recommender.classify()
                    print(recommender.evaluate(test_data))





        
