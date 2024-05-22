from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

class Boosting:
    def __init__(
        self,
        base_model_params: dict = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        subsample: float = 0.3,
        early_stopping_rounds: int = None,
        plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)
        self.plot: bool = plot
        self.history = defaultdict(list)
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        bootstrap_indices = np.random.choice(np.arange(x.shape[0]), size=int(self.subsample * x.shape[0]))
        x_sample = x[bootstrap_indices]
        y_sample = y[bootstrap_indices]
        model = self.base_model_class(**self.base_model_params)
        model.fit(x_sample, self.loss_derivative(y_sample, predictions[bootstrap_indices]))
        gamma = self.find_optimal_gamma(y_sample, predictions[bootstrap_indices], model.predict(x_sample))
        self.gammas.append(gamma)
        self.models.append(model)
    def fit(self, x_train, y_train, x_valid, y_valid):
        train_predictions = np.zeros(y_train.shape[0], dtype=float)[:, None]
        valid_predictions = np.zeros(y_valid.shape[0], dtype=float)[:, None]
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions[:, 0])
            train_predictions += self.learning_rate * self.models[-1].predict(x_train)[:, None]
            valid_predictions += self.learning_rate * self.models[-1].predict(x_valid)[:, None]
            train_loss = self.loss_fn(y_train, train_predictions[:, 0])
            valid_loss = self.loss_fn(y_valid, valid_predictions[:, 0])
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            if self.early_stopping_rounds is not None:
                if len(self.history['valid_loss']) >= self.early_stopping_rounds and \
                        valid_loss > np.min(self.history['valid_loss'][-self.early_stopping_rounds:]):
                    break
        if self.plot:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.history['train_loss'])), self.history['train_loss'], label='Train')
            plt.plot(range(len(self.history['valid_loss'])), self.history['valid_loss'], label='Valid')
            plt.xlabel('Number of iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
    def predict_proba(self, x):
        predictions = np.zeros((x.shape[0], 2), dtype=float)
        for model, gamma in zip(self.models, self.gammas):
            predictions += gamma * model.predict(x)[:, None]
        return self.sigmoid(predictions)
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=-1, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]
    def score(self, x, y):
        return roc_auc_score(y, self.predict_proba(x)[:, 1])
    @property
    def feature_importances_(self):
        importances = np.zeros(self.models[0].feature_importances_.shape)
        for gamma, model in zip(self.gammas, self.models):
            importances += gamma * model.feature_importances_
        importances /= np.sum(importances)
        return importances