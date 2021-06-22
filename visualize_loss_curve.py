import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help="input file")
parser.add_argument('--smooth', required=False, action='store_true')

args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


avg_losses = AverageMeter()
all_losses = []

with open(args.input, "r") as loss_file:
    losses = loss_file.readlines()
    for loss in losses:

        if args.smooth:
            avg_losses.update(float(loss))
            all_losses.append(avg_losses.avg)
        else:
            all_losses.append(float(loss))

    plt.plot(all_losses)
    plt.show()
