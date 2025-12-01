
import matplotlib.pyplot as plt

class LearningRater:
    def __init__(self
                 , start: float=0.00001, max: float=0.00005, min: float=0.00001
                 , rampup_epochs: int=5, sustain_epochs: int=0
                 , exp_decay: float=0.8):
        self.start = start
        self.max = max
        self.min = min
        self.rampup_epochs = rampup_epochs
        self.sustain_epochs = sustain_epochs
        self.exp_decay = exp_decay
        
    def __call__(self, epoch: int) -> float:
        if epoch < self.rampup_epochs:
            lr = (self.max - self.start) / self.rampup_epochs * epoch + self.start
        else:
            lr = self.max
        return lr

def lrfn_origin(epoch):
    LR_START = 0.00001
    LR_MAX = 0.00005 #* strategy.num_replicas_in_sync
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8
    
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

def get_lr_callback(max_epochs: int = 25, plot_schedule: bool = False):    
    if plot_schedule:
        lrfn = LearningRater()
        rng = [i for i in range(25 if max_epochs < 25 else max_epochs)]
        y = [lrfn(x) for x in rng]
        plt.plot(rng, y)

if __name__ == '__main__':
    get_lr_callback(plot_schedule=True)
