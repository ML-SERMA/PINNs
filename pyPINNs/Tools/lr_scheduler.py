
class StepLRWithDecayingRestart:
    def __init__(
        self,
        optimizer,
        step_size=1000,
        gamma=0.9,
        restart_every=20000,
        restart_decay=0.8
    ):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.restart_every = restart_every
        self.restart_decay = restart_decay

        self.iteration = 0
        self.local_iteration = 0

        # read base LR from optimizer (single source of truth)
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self.iteration += 1
        self.local_iteration += 1

        # restart
        if self.iteration % self.restart_every == 0:
            self.local_iteration = 0
            self.base_lrs = [lr * self.restart_decay for lr in self.base_lrs]

            for pg, lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = lr
            return

        # step decay
        if self.local_iteration % self.step_size == 0:
            for pg in self.optimizer.param_groups:
                pg["lr"] *= self.gamma

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]



















