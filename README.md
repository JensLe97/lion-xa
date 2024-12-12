# LiOn-XA: Unsupervised Domain Adaptation via LiDAR-Only Cross-Modal Adversarial Training

Official code for our IROS 2024 paper.

## Paper

[LiOn-XA: Unsupervised Domain Adaptation via LiDAR-Only Cross-Modal Adversarial Training](https://arxiv.org/abs/2410.15833)  
Thomas Kreutz, Jens Lemke, Max Mühlhäuser, Alejandro Sanchez Guinea

We propose a domain adaptation approach for 3D LiDAR point cloud semantic segmentation in country-to-country, sensor-to-sensor, dataset-to-dataset scenarios.

If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/2410.15833):

```
@inproceedings{kreutz2024lionxa,
    author = {Kreutz, Thomas and Lemke, Jens and Mühlhäuser, Max and Guinea, Alejandro},
    year = {2024},
    month = {10},
    title = {LiOn-XA: Unsupervised Domain Adaptation via LiDAR-Only Cross-Modal Adversarial Training},
    booktitle={IROS},
}
```
## Preparation
### Prerequisites
Tested with
* Python 3.8.13
* PyTorch 1.8.2
* CUDA 11.1

### Installation
This setup is tested using a studio from lightning.ai having a GPU with 24 GB VRAM. You need to setup the environment with GPU enabled to install the 3D network SparseConNet correctly.

1. Clone this repository
```
$ git clone https://github.com/JensLe97/lion-xa.git
$ cd lion-xa
```
2. This step only applies if you use lightning.ai to work in an all-in-one platform for AI development. If you do so, you need to setup CUDA to version 11.1 as it defaults to CUDA version 12
```
$ wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
$ sudo sh cuda_11.1.1_455.32.00_linux.run
```
* During the installation, unselect CUDA Driver and chose to update symlink
* Set the environment variables
```
$ export CUDA_HOME=/usr/local/cuda-11.1
$ export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
$ export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6+PTX"
```
3. Create or update an existing conda environment (e.g. named cloudspace) and install the requiremnents with
```
$ conda env update --file environment.yml --prune
```
4. Install SparseConvNet with
```
$ pip install git+https://github.com/facebookresearch/SparseConvNet.git
```

### Datasets
#### nuScenes and nuScenes-lidarseg
* Please download the full dataset (v1.0) with Trainval and Test from the [nuScenes website](https://www.nuscenes.org/nuscenes#download). Extract the files to your nuScenes root directory (e.g. `/teamspace/studios/this_studio/data/datasets/nuScenes`). Also download nuScenes-lidarseg data (All) and extract it to the same root directory.
* Download the pkl files from this [Google Drive](https://drive.google.com/drive/folders/1zSZ9xE4UkKBMCMH0le7KdSxvbyjuuUp8), place them in the nuScenes root directory and rename them to `train_all.pkl`, `val_all.pkl` and `test_all.pkl`.

You need to perform preprocessing to generate the data for lion-xa first.

Please edit the script `lion-xa/data/nuscenes/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the nuScenes dataset
* `out_dir` should point to the desired output directory to store the pickle files. This should be different if you preprocess the nuScenes-lidarseg data. (e.g., `/teamspace/studios/this_studio/data/datasets/nuScenes/nuscenes_preprocess(_lidarseg)/preprocess`)
* If you want to preprocess data for nuScenes-lidarseg, set the the `lidarseg` parameter in the preprocess function calls to `True`

#### SemanticKITTI
* Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and
additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip)
from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract
everything into the same folder. Links to the datasets:

```
$ wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip
$ wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
$ wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
$ wget http://semantic-kitti.org/assets/data_odometry_labels.zip  
```

Please edit the script `lion-xa/data/semantic_kitti/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the SemanticKITTI dataset (e.g., `/teamspace/studios/this_studio/data/datasets/SemanticKITTI`)
* `out_dir` should point to the desired output directory to store the pickle files (e.g., `/teamspace/studios/this_studio/data/datasets/SemanticKITTI/semantic_kitti_preprocess_lidar`)

##### Create target-like dataset for SemanticKITTI

Convert SemanticKITTI data to target-like data with SemanticPOSS as the target dataset. 
We use the implementation from [LiDAR Transfer](https://github.com/PRBonn/lidar_transfer). Please follow the installation process in the [README](./LiDAR_Transfer/README.md). The absolute path to the script, source and target directory needs to be specified in `LiDAR_Transfer/experiments/run_lidar_deform_kitti2poss.sh`. 
```
$ cd LiDAR_Transfer/experiments
$ bash run_lidar_deform_kitti2poss.sh
```

#### SemanticPOSS
* Please download the files from the [SemanticPOSS website](http://www.poss.pku.edu.cn/semanticposs.html) and extract them to the SemanticPOSS root directory, e.g., `/teamspace/studios/this_studio/data/datasets/SemanticPOSS`

Please edit the script `lion-xa/data/semantic_poss/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the SemanticKITTI dataset (e.g., `/teamspace/studios/this_studio/data/datasets/SemanticPOSS`)
* `out_dir` should point to the desired output directory to store the pickle files (e.g., `/teamspace/studios/this_studio/data/datasets/SemanticPOSS/semantic_poss_preprocess_lidar`)

## Training
### LiOn-XA without Discriminators
Specify the path of the preprocessed datasets (DATASET_SOURCE and DATASET_TARGET) and the path to the output directory. For nuScenes, change `config/nuscenes/usa_singapore/lion_xa_sn_sgd.yaml`:
```
NuScenesSCN:
    root_dir: "/teamspace/studios/this_studio/data/datasets/nuScenes/nuscenes_preprocess/preprocess"
OUTPUT_DIR: "/teamspace/studios/this_studio/lion-xa/LION_XA/output/@"
```

You can run the training with
```
$ cd <root dir of this repo>
$ python LION_XA/train_lion_xa.py --cfg=LION_XA/config/nuscenes/usa2singapore/lion_xa_sn_sgd.yaml
```

You can start the trainings on the other UDA scenarios analogously:
```
$ python LION_XA/train_lion_xa.py --cfg=LION_XA/config/nuscenes/usa2singapore/lion_xa_lidarseg_sn_scn.yaml
$ python LION_XA/train_lion_xa.py --cfg=LION_XA/config/semantic_kitti2poss/lion_xa_sn.yaml
$ python LION_XA/train_lion_xa.py --cfg=LION_XA/config/kitti2nuscenes/lion_xa_sn.yaml
```

### LiOn-XA with Discriminators
To train the model with the adversarial training technique, run:
```
$ python LION_XA/train_lion_xa_dis.py --cfg=LION_XA/config/nuscenes/usa2singapore/lion_xa_sn_sgd.yaml
$ python LION_XA/train_lion_xa_dis.py --cfg=LION_XA/config/nuscenes/usa2singapore/lion_xa_lidarseg_sn_scn.yaml
$ python LION_XA/train_lion_xa_dis.py --cfg=LION_XA/config/semantic_kitti2poss/lion_xa_sn.yaml
$ python LION_XA/train_lion_xa_dis.py --cfg=LION_XA/config/kitti2nuscenes/lion_xa_sn.yaml
```

### LiOn-XA with Target-like Data (and Discriminators)
For the SemanticKITTI to SemanticPOSS scenario, you can use the target-like data to train the model. Specify the path of the target-like dataset in `config/semantic_kitti2poss/lion_xa_sn.yaml`:
```
  SemanticKITTISCN:
    root_dir: "/teamspace/studios/this_studio/data/datasets/SemanticKITTI"
    trgl_dir: "/teamspace/studios/this_studio/data/datasets/SemanticKITTI/semantic_kitti2poss"
```
Then start the training with
```
$ python LION_XA/train_lion_xa_dis.py --cfg=LION_XA/config/semantic_kitti2poss/lion_xa_sn.yaml
```
or additionally include the discriminators:
```
$ python LION_XA/train_lion_xa_dis_tgl.py --cfg=LION_XA/config/semantic_kitti2poss/lion_xa_sn.yaml
```
### View training results
Run tensorboard in the console or use the build-in tensorboard app from lightning.ai
```
$ tensorboard --logdir .
```
Open `http://localhost:6006/` in a browser

## Testing
You can provide which checkpoints you want to use for testing. We used the ones
that performed best on the validation set during training (the best val iteration for 2D and 3D is
shown at the end of each training). Note that `@` will be replaced
by the output directory for that config file. For example:
```
$ cd <root dir of this repo>
$ python lion-xa/LION_XA/test.py --cfg=lion-xa/LION_XA/config/nuscenes/usa2singapore/lion_xa_sn_sgd.yaml @/model_2d_095000.pth @/model_3d_095000.pth
```
You can also provide an absolute path without `@`.

## Model Zoo

You can download the models and logs with the scores from the paper at
[TU datalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4327.5).

## Acknowledgements
This implementation is based on [xMUDA](https://github.com/valeoai/xmuda). Furthermore we use the implementation from [LiDAR Transfer](https://github.com/PRBonn/lidar_transfer) to generate target-like data.

## License
LiOn-XA is released under the [Apache 2.0 license](./LICENSE).
