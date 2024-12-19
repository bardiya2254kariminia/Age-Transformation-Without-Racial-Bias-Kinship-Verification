# from argparse import ArgumentParser
# import argparse
# import os

# from utils.pipeline_utils import default_train_results_dir

# # from configs.paths_config import model_paths


# def str_to_gender(s):
#     s = str(s).lower()
#     if s in ("m", "man", "0"):
#         return 0
#     elif s in ("f", "female", "1"):
#         return 1
#     else:
#         raise KeyError("No gender found")


# def str_to_bool(s):
#     s = s.lower()
#     if s in ("true", "t", "yes", "y", "1"):
#         return True
#     elif s in ("false", "f", "no", "n", "o"):
#         return False
#     else:
#         raise KeyError("Invalid boolean")


# class Keyboared:

#     def __init__(self):
#         self.parser = argparse.ArgumentParser(
#             description="AgeProgression on PyTorch.",
#             formatter_class=argparse.RawTextHelpFormatter,
#         )
#         self.initialize()

#     def initialize(self):
#         self.parser.add_argument("--mode", choices=["train", "test"], default="train")

#         # train params
#         self.parser.add_argument("--epochs", "-e", default=1, type=int)
#         self.parser.add_argument(
#             "--models-saving",
#             "--ms",
#             dest="models_saving",
#             choices=("always", "last", "tail", "never"),
#             default="tail",
#             type=str,
#             help="Model saving preference.{br}"
#             "\talways: Save trained model at the end of every epoch (default){br}"
#             "\tUse this option if you have a lot of free memory and you wish to experiment with the progress of your results.{br}"
#             "\tlast: Save trained model only at the end of the last epoch{br}"
#             "\tUse this option if you don't have a lot of free memory and removing large binary files is a costly operation.{br}"
#             '\ttail: "Safe-last". Save trained model at the end of every epoch and remove the saved model of the previous epoch{br}'
#             "\tUse this option if you don't have a lot of free memory and removing large binary files is a cheap operation.{br}"
#             "\tnever: Don't save trained model{br}"
#             "\tUse this option if you only wish to collect statistics and validation results.{br}"
#             "All options except 'never' will also save when interrupted by the user.".format(
#                 br=os.linesep
#             ),
#         )
#         self.parser.add_argument(
#             "--batch-size", "--bs", dest="batch_size", default=2, type=int
#         )
#         self.parser.add_argument(
#             "--weight-decay", "--wd", dest="weight_decay", default=1e-5, type=float
#         )
#         self.parser.add_argument(
#             "--learning-rate", "--lr", dest="learning_rate", default=2e-5, type=float
#         )
#         self.parser.add_argument("--b1", "-b", dest="b1", default=0.5, type=float)
#         self.parser.add_argument("--b2", "-B", dest="b2", default=0.999, type=float)
#         self.parser.add_argument(
#             "--shouldplot", "--sp", dest="sp", default=False, type=bool
#         )

#         # test params
#         self.parser.add_argument("--age", "-a", required=False, type=int)
#         self.parser.add_argument("--gender", "-g", required=False, type=str_to_gender)
#         self.parser.add_argument("--watermark", "-w", action="store_true")

#         # shared params
#         self.parser.add_argument(
#             "--cpu",
#             "-c",
#             action="store_true",
#             help="Run on CPU even if CUDA is available.",
#             default="cpu",
#         )
#         self.parser.add_argument(
#             "--load",
#             "-l",
#             required=False,
#             default=None,
#             help="Trained models path for pre-training or for testing",
#         )
#         self.parser.add_argument(
#             "--input",
#             "-i",
#             default=None,
#             help="Training dataset path (default is {}) or testing image path".format(
#                 default_train_results_dir()
#             ),
#         )
#         self.parser.add_argument("--output", "-o", default="")
#         self.parser.add_argument(
#             "-z", dest="z_channels", default=50, type=int, help="Length of Z vector"
#         )
#         self.parser.add_argument(
#             "--exp_dir", type=str, help="Path to experiment output directory"
#         )
#         self.parser.add_argument(
#             "--dataset_type",
#             default="ffhq_encode",
#             type=str,
#             help="Type of dataset/experiment to run",
#         )

#         # self.parser.add_argument('--encoder_type', default='Encoder4Editing', type=str, help='Which encoder to use')

#         # self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
#         # self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
#         self.parser.add_argument(
#             "--workers", default=4, type=int, help="Number of train dataloader workers"
#         )
#         self.parser.add_argument(
#             "--test_workers",
#             default=2,
#             type=int,
#             help="Number of test/inference dataloader workers",
#         )

#         # self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
#         # self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
#         # self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
#         # self.parser.add_argument('--start_from_latent_avg', action='store_true',
#         # help='Whether to add average latent vector to generate codes from encoder.')
#         self.parser.add_argument(
#             "--lpips_type", default="alex", type=str, help="LPIPS backbone"
#         )

#         self.parser.add_argument(
#             "--lpips_lambda",
#             default=0.8,
#             type=float,
#             help="LPIPS loss multiplier factor",
#         )
#         self.parser.add_argument(
#             "--id_lambda", default=0.1, type=float, help="ID loss multiplier factor"
#         )
#         self.parser.add_argument(
#             "--l2_lambda", default=1.0, type=float, help="L2 loss multiplier factor"
#         )

#         self.parser.add_argument(
#             "--stylegan_size",
#             default=1024,
#             type=int,
#             help="size of pretrained StyleGAN Generator",
#         )
#         # self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

#         # self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
#         # self.parser.add_argument('--image_interval', default=100, type=int,
#         #                         help='Interval for logging train images during training')
#         # self.parser.add_argument('--board_interval', default=50, type=int,
#         #                         help='Interval for logging metrics to tensorboard')
#         # self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
#         # self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

#         # Discriminator flags
#         # self.parser.add_argument('--w_discriminator_lambda', default=0, type=float, help='Dw loss multiplier')
#         # self.parser.add_argument('--w_discriminator_lr', default=2e-5, type=float, help='Dw learning rate')
#         # self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
#         # self.parser.add_argument("--d_reg_every", type=int, default=16,
#         #                         help="interval for applying r1 regularization")
#         # self.parser.add_argument('--use_w_pool', action='store_true',
#         #                         help='Whether to store a latnet codes pool for the discriminator\'s training')
#         # self.parser.add_argument("--w_pool_size", type=int, default=50,
#         #                         help="W\'s pool size, depends on --use_w_pool")

#         # e4e specific
#         self.parser.add_argument(
#             "--delta_norm", type=int, default=2, help="norm type of the deltas"
#         )
#         self.parser.add_argument(
#             "--delta_norm_lambda",
#             type=float,
#             default=2e-4,
#             help="lambda for delta norm loss",
#         )

#         # Progressive training
#         self.parser.add_argument(
#             "--progressive_steps",
#             nargs="+",
#             type=int,
#             default=None,
#             help="The training steps of training new deltas. steps[i] starts the delta_i training",
#         )
#         self.parser.add_argument(
#             "--progressive_start",
#             type=int,
#             default=None,
#             help="The training step to start training the deltas, overrides progressive_steps",
#         )
#         self.parser.add_argument(
#             "--progressive_step_every",
#             type=int,
#             default=2_000,
#             help="Amount of training steps for each progressive step",
#         )

#         # Save additional training info to enable future training continuation from produced checkpoints
#         self.parser.add_argument(
#             "--save_training_data",
#             action="store_true",
#             help="Save intermediate training data to resume training from the checkpoint",
#         )
#         self.parser.add_argument(
#             "--sub_exp_dir",
#             default=None,
#             type=str,
#             help="Name of sub experiment directory",
#         )
#         self.parser.add_argument(
#             "--keep_optimizer",
#             action="store_true",
#             help="Whether to continue from the checkpoint's optimizer",
#         )
#         self.parser.add_argument(
#             "--resume_training_from_ckpt",
#             default=None,
#             type=str,
#             help="Path to training checkpoint, works when --save_training_data was set to True",
#         )
#         self.parser.add_argument(
#             "--update_param_list",
#             nargs="+",
#             type=str,
#             default=None,
#             help="Name of training parameters to update the loaded training checkpoint",
#         )


from argparse import ArgumentParser
import argparse
import os

from utils.pipeline_utils import default_train_results_dir

# from configs.paths_config import model_paths


def str_to_gender(s):
    s = str(s).lower()
    if s in ("m", "man", "0"):
        return 0
    elif s in ("f", "female", "1"):
        return 1
    else:
        raise KeyError("No gender found")


def str_to_bool(s):
    s = s.lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    elif s in ("false", "f", "no", "n", "o"):
        return False
    else:
        raise KeyError("Invalid boolean")


class Keyboared:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="AgeProgression on PyTorch.",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--mode", choices=["train", "test"], default="train")

        # train params
        self.parser.add_argument("--epochs", "-e", default=1, type=int)
        self.parser.add_argument(
            "--models-saving",
            "--ms",
            dest="models_saving",
            choices=("always", "last", "tail", "never"),
            default="tail",
            type=str,
            help="Model saving preference.{br}"
            "\talways: Save trained model at the end of every epoch (default){br}"
            "\tUse this option if you have a lot of free memory and you wish to experiment with the progress of your results.{br}"
            "\tlast: Save trained model only at the end of the last epoch{br}"
            "\tUse this option if you don't have a lot of free memory and removing large binary files is a costly operation.{br}"
            '\ttail: "Safe-last". Save trained model at the end of every epoch and remove the saved model of the previous epoch{br}'
            "\tUse this option if you don't have a lot of free memory and removing large binary files is a cheap operation.{br}"
            "\tnever: Don't save trained model{br}"
            "\tUse this option if you only wish to collect statistics and validation results.{br}"
            "All options except 'never' will also save when interrupted by the user.".format(
                br=os.linesep
            ),
        )
        self.parser.add_argument(
            "--batch-size", "--bs", dest="batch_size", default=16, type=int
        )
        self.parser.add_argument(
            "--weight-decay", "--wd", dest="weight_decay", default=1e-5, type=float
        )
        self.parser.add_argument(
            "--learning-rate", "--lr", dest="learning_rate", default=2e-5, type=float
        )
        self.parser.add_argument("--b1", "-b", dest="b1", default=0.5, type=float)
        self.parser.add_argument("--b2", "-B", dest="b2", default=0.999, type=float)
        self.parser.add_argument(
            "--shouldplot", "--sp", dest="sp", default=False, type=bool
        )

        # test params
        self.parser.add_argument("--age", "-a", required=False, type=int)
        self.parser.add_argument("--gender", "-g", required=False, type=str_to_gender)
        self.parser.add_argument("--watermark", "-w", action="store_true")

        # shared params
        self.parser.add_argument(
            "--cpu",
            "-c",
            action="store_true",
            help="Run on CPU even if CUDA is available.",
            default="cpu",
        )
        self.parser.add_argument(
            "--load",
            "-l",
            required=False,
            default=None,
            help="Trained models path for pre-training or for testing",
        )
        self.parser.add_argument(
            "--input",
            "-i",
            default=r"cacd_utkface_hold_dir/content/drive/MyDrive/data/CACD_UTKFace",
            help="Training dataset path (default is {}) or testing image path".format(
                default_train_results_dir()
            ),
        )
        self.parser.add_argument("--output", "-o", default=r"code\artifacts")
        self.parser.add_argument(
            "-z", dest="z_channels", default=50, type=int, help="Length of Z vector"
        )
        self.parser.add_argument(
            "--exp_dir", type=str, help="Path to experiment output directory"
        )
        self.parser.add_argument(
            "--dataset_type",
            default="ffhq_encode",
            type=str,
            help="Type of dataset/experiment to run",
        )

        # self.parser.add_argument('--encoder_type', default='Encoder4Editing', type=str, help='Which encoder to use')

        # self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        # self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument(
            "--workers", default=4, type=int, help="Number of train dataloader workers"
        )
        self.parser.add_argument(
            "--test_workers",
            default=2,
            type=int,
            help="Number of test/inference dataloader workers",
        )

        # self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        # self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        # self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        # self.parser.add_argument('--start_from_latent_avg', action='store_true',
        # help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument(
            "--lpips_type", default="alex", type=str, help="LPIPS backbone"
        )

        self.parser.add_argument(
            "--lpips_lambda",
            default=0.8,
            type=float,
            help="LPIPS loss multiplier factor",
        )
        self.parser.add_argument(
            "--id_lambda", default=0.1, type=float, help="ID loss multiplier factor"
        )
        self.parser.add_argument(
            "--l2_lambda", default=1.0, type=float, help="L2 loss multiplier factor"
        )

        self.parser.add_argument(
            "--stylegan_size",
            default=1024,
            type=int,
            help="size of pretrained StyleGAN Generator",
        )
        # self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

        # self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        # self.parser.add_argument('--image_interval', default=100, type=int,
        #                         help='Interval for logging train images during training')
        # self.parser.add_argument('--board_interval', default=50, type=int,
        #                         help='Interval for logging metrics to tensorboard')
        # self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        # self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

        # Discriminator flags
        # self.parser.add_argument('--w_discriminator_lambda', default=0, type=float, help='Dw loss multiplier')
        # self.parser.add_argument('--w_discriminator_lr', default=2e-5, type=float, help='Dw learning rate')
        # self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        # self.parser.add_argument("--d_reg_every", type=int, default=16,
        #                         help="interval for applying r1 regularization")
        # self.parser.add_argument('--use_w_pool', action='store_true',
        #                         help='Whether to store a latnet codes pool for the discriminator\'s training')
        # self.parser.add_argument("--w_pool_size", type=int, default=50,
        #                         help="W\'s pool size, depends on --use_w_pool")

        # e4e specific
        self.parser.add_argument(
            "--delta_norm", type=int, default=2, help="norm type of the deltas"
        )
        self.parser.add_argument(
            "--delta_norm_lambda",
            type=float,
            default=2e-4,
            help="lambda for delta norm loss",
        )

        # Progressive training
        self.parser.add_argument(
            "--progressive_steps",
            nargs="+",
            type=int,
            default=None,
            help="The training steps of training new deltas. steps[i] starts the delta_i training",
        )
        self.parser.add_argument(
            "--progressive_start",
            type=int,
            default=None,
            help="The training step to start training the deltas, overrides progressive_steps",
        )
        self.parser.add_argument(
            "--progressive_step_every",
            type=int,
            default=2_000,
            help="Amount of training steps for each progressive step",
        )

        # Save additional training info to enable future training continuation from produced checkpoints
        self.parser.add_argument(
            "--save_training_data",
            action="store_true",
            help="Save intermediate training data to resume training from the checkpoint",
        )
        self.parser.add_argument(
            "--sub_exp_dir",
            default=None,
            type=str,
            help="Name of sub experiment directory",
        )
        self.parser.add_argument(
            "--keep_optimizer",
            action="store_true",
            help="Whether to continue from the checkpoint's optimizer",
        )
        self.parser.add_argument(
            "--resume_training_from_ckpt",
            default=None,
            type=str,
            help="Path to training checkpoint, works when --save_training_data was set to True",
        )
        self.parser.add_argument(
            "--update_param_list",
            nargs="+",
            type=str,
            default=None,
            help="Name of training parameters to update the loaded training checkpoint",
        )
