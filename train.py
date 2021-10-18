import os
import argparse

import torch

from stats import StatTracker
from datasets import build_dataset, get_dataset, get_encoder_size
from model import Model
from checkpoint import Checkpointer
from task_train import train_model

parser = argparse.ArgumentParser(description='IMC-SwAV - Training')

parser.add_argument('--dataset', type=str, default='C10')
parser.add_argument('--path', type=str, default=None, help='Root directory for the dataset')

# Transformation parameters/number of crops sizes. First index reports the high resolution, low resolutions follows.
parser.add_argument('--nmb_workers', type=int, default=8, help='Number of workers on Transformation process')
parser.add_argument("--nmb_crops", type=int, default=[2, 4], nargs="+",
                    help="list of number of crops (i.e.: [2, 4])")
parser.add_argument("--size_crops", type=int, default=[28, 18], nargs="+",
                    help="crops resolutions (i.e.: [28, 18])")
parser.add_argument("--max_scale_crops", type=float, default=[1., 0.4], nargs="+",
                    help="argument in RandomResizedCrop (i.e.: [1., 0.5])")
parser.add_argument("--min_scale_crops", type=float, default=[0.2, 0.08], nargs="+",
                    help="argument in RandomResizedCrop (i.e.: [0.2, 0.08])")
parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 256)')

# Model and training parameters
parser.add_argument('--tau', type=float, default=0.1, help='Temperature parameter on Softmax (Eq. 2)')
parser.add_argument('--eps', type=float, default=0.05, help='Epsilon scalar of Sinkhorn-Knopp (Eq. 3)')
parser.add_argument('--warmup', type=int, default=500, help='Epoch of warmup schedule')
parser.add_argument('--epochs', type=int, default=500, help='Training epoch')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument("--project_dim", type=int, default=128, help="Project embedding dimension")
parser.add_argument("--prototypes", type=int, default=1000, help="Number of prototypes")
parser.add_argument("--model_type", type=str, default='resnet18', help="Type of ResNet")

# parameters for output, logging, checkpointing, etc
parser.add_argument('--output_dir', type=str, default='./default_run',
                    help='Storing path for Tensorboard events and checkpoints')
parser.add_argument('--cpt_load_path', type=str, default=None, help='Load checkpoint path+name(if available)')
parser.add_argument('--cpt_name', type=str, default='imc_swav.cpt', help='Checkpoint name during training')
parser.add_argument('--run_name', type=str, default='default_run', help='Tensorboard summary name')
parser.add_argument("--dev", type=str, help='GPU device number (if applying)')
parser.add_argument("--l2_w", type=float, default=1e-5, help='l_2 weights')

args = parser.parse_args()

if args.dev is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev


def main():
    # create output dir (only if it doesn't exist)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # get the dataset
    dataset = get_dataset(args.dataset)
    encoder_size = get_encoder_size(dataset)

    # get a helper object for tensorboard logging
    log_dir = os.path.join(args.output_dir, args.run_name)
    stat_tracker = StatTracker(log_dir=log_dir)

    # get training and testing loaders
    train_loader, test_loader, num_classes = \
        build_dataset(dataset=dataset, batch_size=args.batch_size, nmb_workers=args.nmb_workers,
                      nmb_crops=args.nmb_crops, size_crops=args.size_crops,
                      min_scale_crops=args.min_scale_crops, max_scale_crops=args.max_scale_crops, path=args.path)

    torch_device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
    checkpointer = Checkpointer(args.output_dir, args.cpt_name)
    if args.cpt_load_path:
        model = checkpointer.restore_model_from_checkpoint(args.cpt_load_path)
    else:
        # create new model with random parameters
        model = Model(n_classes=num_classes, encoder_size=encoder_size, prototypes=args.prototypes,
                      project_dim=args.project_dim, tau=args.tau, eps=args.eps, model_type=args.model_type)
        checkpointer.track_new_model(model)

    model = model.to(torch_device)

    train_model(model, args.learning_rate, train_loader, test_loader, args.nmb_crops, stat_tracker,
                checkpointer, torch_device, args.warmup, args.epochs, args.l2_w)


if __name__ == "__main__":
    print(args)
    main()
