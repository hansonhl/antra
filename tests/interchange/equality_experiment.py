from datasets import EqualityDataset, PremackDataset, PremackDatasetLeafFlattened
from itertools import product
import numpy as np
import os
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import warnings


class EqualityExperiment:

    def __init__(self,
            dataset_class=EqualityDataset,
            n_hidden=1,
            model=None,
            n_trials=10,
            train_sizes=list(range(104, 100001, 5000)),
            embed_dims=[2, 10, 25, 50, 100],
            hidden_dims=[2, 10, 25, 50, 100],
            alphas=[0.00001, 0.0001, 0.001],
            learning_rates=[0.0001, 0.001, 0.01],
            test_set_class_size=250):
        self.dataset_class = dataset_class
        self.n_hidden = n_hidden
        self.model = model
        self.n_trials = n_trials
        self.train_sizes = train_sizes
        self.class_size = int(max(self.train_sizes) / 2)
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.alphas = alphas
        self.learning_rates = learning_rates
        grid = (self.embed_dims, self.hidden_dims, self.alphas, self.learning_rates)
        self.grid = list(product(*grid))
        self.test_set_class_size = test_set_class_size

    def run(self):
        data = []

        print(f"Grid size: {len(self.grid)} * {self.n_trials}; "
              f"{len(self.grid)*self.n_trials} experiments")

        for embed_dim, hidden_dim, alpha, lr in self.grid:

            print(f"Running trials for embed_dim={embed_dim} hidden_dim={hidden_dim} "
                  f"alpha={alpha} lr={lr} ...", end=" ")

            start = time.time()

            scores = []

            for trial in range(1, self.n_trials+1):

                mod = self.get_model(hidden_dim, alpha, lr, embed_dim)

                X_train, X_test, y_train, y_test, test_dataset = \
                  self.get_new_train_and_test_sets(embed_dim)

                # Record the result with no training if the model allows it:
                try:
                    preds = mod.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    scores.append(acc)
                    d = {
                        'trial': trial,
                        'train_size': 0,
                        'embed_dim': embed_dim,
                        'hidden_dim': hidden_dim,
                        'alpha': alpha,
                        'learning_rate': lr,
                        'accuracy': acc,
                        'batch_pos': 0,
                        'batch_neg': 0}
                    if hasattr(self, "pretraining_metadata"):
                        d.update(self.pretraining_metadata)
                    data.append(d)
                except NotFittedError:
                    pass

                for train_size in self.train_sizes:

                    train_size_start = 0

                    if train_size < 40:
                        X_batch, y_batch = self.get_minimal_train_set(
                            train_size, embed_dim, test_dataset)
                        batch_pos = sum([1 for label in y_batch if label == 1])
                    else:
                        X_batch = X_train[train_size_start: train_size]
                        y_batch = y_train[train_size_start: train_size]
                        batch_pos = sum([1 for label in y_train[: train_size] if label == 1])

                    train_size_start = train_size

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod.fit(X_batch, y_batch)

                    # Predictions:
                    preds = mod.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    scores.append(acc)
                    d = {
                        'trial': trial,
                        'train_size': train_size,
                        'embed_dim': embed_dim,
                        'hidden_dim': hidden_dim,
                        'alpha': alpha,
                        'learning_rate': lr,
                        'accuracy': acc,
                        'batch_pos': batch_pos,
                        'batch_neg': len(X_batch) - batch_pos}
                    if hasattr(self, "pretraining_metadata"):
                        d.update(self.pretraining_metadata)
                    data.append(d)

            elapsed_time = round(time.time() - start, 0)

            print(f"mean: {round(np.mean(scores), 2)}; max: {max(scores)}; took {elapsed_time} secs")

        self.data_df = pd.DataFrame(data)
        return (self.data_df,mod,X_train)

    def to_csv(self, base_output_filename, output_dirname="results"):
        self.data_df.to_csv(
            os.path.join(output_dirname, base_output_filename),
            index=None)

    def get_model(self, hidden_dim, alpha, lr, embed_dim):
        if self.model is None:
            return MLPClassifier(
                max_iter=1,
                hidden_layer_sizes=tuple([hidden_dim] * self.n_hidden),
                activation='relu',
                alpha=alpha,
                solver='adam',
                learning_rate_init=lr,
                beta_1=0.9,
                beta_2=0.999,
                warm_start=True)
        else:
            return self.model(
                hidden_dim=hidden_dim,
                alpha=alpha,
                lr=lr,
                embed_dim=embed_dim)

    def get_new_train_and_test_sets(self, embed_dim):
        train_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=self.class_size,
            n_neg=self.class_size)
        X_train, y_train = train_dataset.create()

        test_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=self.test_set_class_size,
            n_neg=self.test_set_class_size)
        X_test, y_test = test_dataset.create()

        train_dataset.test_disjoint(test_dataset)

        return X_train, X_test, y_train, y_test, test_dataset

    def get_minimal_train_set(self, train_size, embed_dim, other_dataset):
        class_size = int(train_size / 2)
        train_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=class_size,
            n_neg=class_size)
        X_batch, y_batch = train_dataset.create()

        train_dataset.test_disjoint(other_dataset)

        return X_batch, y_batch
