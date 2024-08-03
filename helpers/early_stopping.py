import torch


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, path: str) -> None:
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")

        torch.save(model.state_dict(), path)

        self.val_loss_min = val_loss
