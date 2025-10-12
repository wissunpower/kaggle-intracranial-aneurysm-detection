
import matplotlib.pyplot as plt


def lrfn(epoch):
    LR_START = 0.00001
    LR_MAX = 0.00005 #* strategy.num_replicas_in_sync
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8
    
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    else:
        lr = LR_MAX
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
        rng = [i for i in range(25 if max_epochs < 25 else max_epochs)]
        y = [lrfn(x) for x in rng]
        plt.plot(rng, y)

if __name__ == '__main__':
    get_lr_callback(plot_schedule=True)
