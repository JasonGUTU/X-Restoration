import time

from models.sr import SuperResolution
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid


MODEL_NAME_DICT = {
    'sr': SuperResolution,
}


def load_model(model_name):
    if model_name in MODEL_NAME_DICT.keys():
        model_cls = MODEL_NAME_DICT[model_name]
    else:
        raise Exception(f'Check your network name, {model_name} is not in the following available networks: \n{MODEL_NAME_DICT.keys()}')
    return model_cls


class SRLoggingCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        if pl_module.hparams.visualize_gt:
            print(f'\nLogging Ground Truth to Logger...')
            for idx, data in enumerate(trainer.val_dataloaders[0]):
                hr, lr = data
                pl_module.logger.experiment.add_image(f'image_{idx}_validation_results', make_grid(hr), 0)
        print(f'Start training {trainer.max_epochs} epochs, in total {trainer.max_epochs * len(trainer.train_dataloader)} steps:')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pl_module.logger.experiment.add_scalar('learning_rate', trainer.optimizers[0].param_groups[0]['lr'], pl_module.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        print(f'\tSaving model to {trainer.default_root_dir}/{pl_module.hparams.exp_name}-interval-{trainer.current_epoch}-{trainer.global_step}.ckpt')
        trainer.save_checkpoint(f'{trainer.default_root_dir}/{pl_module.hparams.exp_name}-interval-{trainer.current_epoch}-{trainer.global_step}.ckpt')

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        this_epoch_time = time.time() - self.epoch_time
        print(f'\tEpoch {trainer.current_epoch} trained with time {this_epoch_time:.2f}s ...')