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


def read_losses(path, smooth=False):
    all_losses = []
    avg_losses = AverageMeter()

    with open(path, "r") as loss_file:
        losses = loss_file.readlines()
    for loss in losses:

        if smooth:
            avg_losses.update(float(loss))
            all_losses.append(avg_losses.avg)
        else:
            all_losses.append(float(loss))

    return all_losses


async_sgd = read_losses(args.input, args.smooth)
naive_sgd = read_losses("losses.txt", args.smooth)

plt.title("Loss vs. Steps")
plt.ylabel("Loss")
plt.xlabel('Steps')
plt.plot(async_sgd)
plt.plot(naive_sgd)
plt.legend(['Async. SGD', 'SGD'])
plt.show()
