from neural_network_experiments.lib.train import Train
from neural_network_experiments.lib.fcNetwork import FCNetwork
from neural_network_experiments.lib.util import split_data_into_batches, train_test_split, test_loss
from neural_network_experiments.lib.baseExperiment import BaseExperiment
from neural_network_experiments.lib.syntheticData import SyntheticData
from torch import nn
import matplotlib.pyplot as plt

class Experiment1(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        # Get data
        data = SyntheticData(self.config["num_samples"], [self.config["input_dim"], self.config["teacher_hidden_layer_dim"], self.config["output_dim"]]).data
        train_data, test_data = train_test_split(data)

        # Get into batched form
        train_data = split_data_into_batches(train_data, len(train_data))
        test_data = split_data_into_batches(test_data, len(test_data))

        # Train each student model
        loss_fn = nn.MSELoss()
        student_models = {} # dictionary indexed by dim of hidden layer
        for hidden_layer_dim in range(self.config["student_hidden_layer_dim_min"],self.config[ "student_hidden_layer_dim_max"]):
            student_models[hidden_layer_dim] = FCNetwork([self.config["input_dim"], hidden_layer_dim, self.config["output_dim"]])
            train = Train(train_data, student_models[hidden_layer_dim], loss_fn)
            train.train_to_convergence(convergence_threshold=1e-4, max_iterations=1e+4)

        # Test loss for each model
        self.test_loss = []
        for hidden_layer_dim in range(self.config["student_hidden_layer_dim_min"], self.config[ "student_hidden_layer_dim_max"]):
            self.test_loss.append(test_loss(test_data, student_models[hidden_layer_dim], loss_fn))
            
    def write_results(self, results_folder_path):
        plt.plot(range(self.config["student_hidden_layer_dim_min"], self.config[ "student_hidden_layer_dim_max"]), self.test_loss)
        plt.xlabel("Hidden Layer Dimension")
        plt.ylabel("Test Loss")
        plt.title("1 Hidden Layer Linear Model Comparison")
        plt.savefig(results_folder_path + "/" + "test_loss")

if __name__ == '__main__':
    # Config
    config = {
        "num_samples": 10,
        "input_dim": 1,
        "output_dim": 1,
        "teacher_hidden_layer_dim": 1,
        "student_hidden_layer_dim_min": 1, 
        "student_hidden_layer_dim_max": 2
    }

    # Running Experiment
    experiment = Experiment1(config)
    experiment.run()
    experiment.results()