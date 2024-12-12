#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import subprocess


def change_config(filename, key, value):
  with open(filename) as f:
    config = yaml.safe_load(f)
    config[key] = value

  with open(filename, "w") as f:
    yaml.dump(config, f)


def plot(fn, data, title, pre, fname, xdata, ydata, cmap):
  fig = plt.figure(fn, figsize=(8, 6))
  ax = fig.add_subplot(111)
  # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)

  im = plt.imshow(data, interpolation='nearest', cmap=cmap)

  for (j, i), label in np.ndenumerate(data):
    if pre == "Score_MSE":
      ax.text(i, j, "{:2.1f}".format(label), ha='center', va='center')
    else:
      ax.text(i, j, "{:.3f}".format(label), ha='center', va='center')

  plt.xlabel('voxel size')
  plt.ylabel('frames')
  plt.colorbar()
  plt.xticks(np.arange(len(xdata)), xdata)
  plt.yticks(np.arange(len(ydata)), ydata)
  plt.title(title)
  plt.savefig(pre + fname + ".svg")
  plt.savefig(pre + fname + ".pdf")


if __name__ == '__main__':
  cfg_file = "./experiments/grid_search_nframes_voxelsize.yaml"
  dataset = "/_data/datasets/SemanticKITTI/dataset/"
  sequence = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"}
  offset = 70
  adaption = "mergemesh"
  frames = [1, 2, 3, 4, 5, 10, 20]
  voxel_size = [0.25, .1, .075, .05, .04]

  # test settings
  # sequence = {"00", "08"}
  # adaption = "cp"
  # voxel_size = [.5]
  # frames = [1, 2]

  IoU = np.zeros((len(frames), len(voxel_size)))
  Acc = np.zeros((len(frames), len(voxel_size)))
  MSE = np.zeros((len(frames), len(voxel_size)))
  fname = "_" + adaption + "_f" + str(frames[0]) + "-" + str(frames[-1]) \
      + "_v" + str(voxel_size[0]) + "-" + str(voxel_size[-1])

  # run approach
  change_config(cfg_file, "adaption", adaption)
  for f, ff in enumerate(frames):
    change_config(cfg_file, "number_of_scans", ff)
    for v, vv in enumerate(voxel_size):
      for seq_idx, seq in enumerate(sequence):
        print("Run %s @ %s with %d frames, voxel size of %f" %
              (adaption, seq, ff, vv))
        change_config(cfg_file, "voxel_size", vv)
        out = subprocess.check_output([sys.executable,
                                       "lidar_deform.py",
                                       "-c", cfg_file,
                                       "-d", dataset,
                                       "-s", seq,
                                       "-o", str(offset),
                                       "--one_scan",
                                       "-b"])
        out_decoded = out.decode().strip()
        # print(out_decoded)
        if adaption == "mergemesh":
          t = 3
        else:
          t = 0
        iou_ = float(out.splitlines()[-4 - t].decode().strip()[4:])
        acc_ = float(out.splitlines()[-3 - t].decode().strip()[4:])
        mse_ = float(out.splitlines()[-2 - t].decode().strip()[4:])
        IoU[f, v] += iou_
        Acc[f, v] += acc_
        MSE[f, v] += mse_
        print("-> IoU %f, Acc %f, MSE %f" % (iou_, acc_, mse_))
        print("->", out.splitlines()[-1 - t].decode().strip())

  # Get mean over all sequences
  IoU /= len(sequence)
  Acc /= len(sequence)
  MSE /= len(sequence)

  # IoU plot
  plot(1, IoU, "Grid Search IoU Score", "Score_IoU", fname, voxel_size, frames,
       plt.cm.summer)

  # Acc plot
  plot(2, Acc, "Grid Search Acc Score", "Score_Acc", fname, voxel_size, frames,
       plt.cm.summer)

  # MSE plot
  plot(3, MSE, "Grid Search MSE Score", "Score_MSE", fname, voxel_size, frames,
       plt.cm.summer_r)

  plt.show()
