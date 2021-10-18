import argparse
import torch
from stats import AverageMeterSet
from datasets import build_dataset, get_dataset
from checkpoint import Checkpointer
from utils import test_model
import os

parser = argparse.ArgumentParser(description='IMC-SwAV - Testing')

parser.add_argument('--cpt_load_path', type=str, default=None, help='path from which to load checkpoint (if available)')
parser.add_argument('--dataset', type=str, default='c10')
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
parser.add_argument("--dev", type=str, help='GPU device number (if applying)')

args = parser.parse_args()

if args.dev is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev


def test(model, test_loader, device, stats):
    test_model(model, test_loader, device, stats)


def main():
    # get the dataset
    dataset = get_dataset(args.dataset)

    _, test_loader, num_classes = \
        build_dataset(dataset=dataset, batch_size=args.batch_size, nmb_workers=args.nmb_workers,
                      nmb_crops=args.nmb_crops, size_crops=args.size_crops,
                      min_scale_crops=args.min_scale_crops, max_scale_crops=args.max_scale_crops)
    checkpointer = Checkpointer()
    torch_device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
    model = checkpointer.restore_model_from_checkpoint(args.cpt_load_path)
    model = model.to(torch_device)

    test_stats = AverageMeterSet()
    test(model, test_loader, torch_device, test_stats)
    stat_str = test_stats.pretty_string()
    print(stat_str)


if __name__ == "__main__":
    print(args)
    main()
