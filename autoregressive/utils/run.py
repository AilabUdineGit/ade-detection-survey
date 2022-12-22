import json
from tabulate import tabulate

class Run:
    def __init__(self, task):
        self.batch_size = int(task['train_config']['batch_size'])
        self.test_batch_size = self.batch_size
        self.learning_rate = float(task['train_config']['learning_rate'])
        self.epsilon = float(task['train_config']['epsilon'])
        self.dropout = float(task['train_config']['dropout'])
        self.epochs = int(task['train_config']['epochs'])
        self.random_seed = int(task['train_config']['random_seed'])
        self.source_len = int(task['train_config']['source_len'])
        self.target_len = int(task['train_config']['target_len'])

        self.split_folder = task['split_folder']
        self.corpus = task['corpus']
        self.architecture = task['architecture']
        self.model = task['model']
        self.id = task['id']
        self.train_mode = task['train_mode']

    def __str__(self):
        rows = [
                ["BS", self.batch_size],
                ["LR", self.learning_rate],
                ["epochs", self.epochs],
                ["source", self.source_len],
                ["target", self.target_len]]
        return tabulate(rows, headers=["", "value"]) + "\n"
                


class Runs:
    def __init__(self, runs_json):
        with open(runs_json, "r") as fp:
            runs = json.load(fp)

        self.runs = [Run(r) for r in runs]
        self.n_runs = len(self.runs)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.n_runs:
            run = self.runs[self.n]
            self.n += 1
            return run
        else:
            raise StopIteration
