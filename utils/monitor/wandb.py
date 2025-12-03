
import wandb


class WandbInitializer:
    def __init__(self, project: str, save_dir: str, name: str, group: str):
        wandb.init(project=project, dir=save_dir, name=name, group=group)
