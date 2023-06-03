class MultiOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
    
    def state_dict(self):
        return tuple([op.state_dict() for op in self.optimizers])
    
    def load_state_dict(self, state_dicts):
        assert len(state_dicts) == len(self.optimizers)
        for op, state_dict in zip(self.optimizers, state_dicts):
            op.load_state_dict(state_dict)