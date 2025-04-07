import copy


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None or score < self.best_score - self.delta:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)
