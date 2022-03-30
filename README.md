# Fusion Painting

##### Multimodal Fusion with Adaptive Attention for 3D Object Detection.

Unofficial code implementation of [FusionPainting][1]

## Model
TODO

## How it works

The Net is feeded with 2D/3D semantics and a pointcloud with points in xyz format
(intensity could be added, but it is not implemented right now). With that data, it learns to choose the best source of semantics
given each point from a pointcloud. The generated attention mask has the following format:
* Assuming 3 points and 1 voxel with a maximum capacity of 3 points.

```
[[0.1283],
 [0.7079],
 [0.4358]]
```

In the first and third point, the model is more confident about the class from 3D semantics.
In the second point though, it believes that the class from 2D semantics is more likely to be correct.

## Dependencies
TODO

## Usage
Train model

`bash train.sh <gpu> <log.txt>`

Internally calls train_model.py

* Arg 1: Gpu id that will be used to train the model.
* Arg 2: Logs filename.

Generate labels

`python val_model.py --config_path <dataset.yaml> --model <model.pt> --out_path <out/dataset/labels>`

* config_path: Path to the dataset's configs in yaml format.
* model: Weights from a trained model.
* out_path: Path where output labels are saved.

Build video from labels

`python visualize.py --labels_path <out/dataset/labels>--video <video.avi> --data <dataset/pandaset> --config_path <dataset.yaml>`

* labels_path: Path where labels are located.
* video: Output video path (and name.extension).
* data: Dataset folder.
* config_path: Path to the dataset's configs in yaml format.

## Testing setup
* 2D semantics

  Obtained from [Pandaset][2] and [Segmentron][3] and projected onto a pointcloud.

* 3D semantics

  Obtained from [Pandaset][2] and [Cylinder][4] 3D.

* Ground Truth

  [Pandaset][2] 3D semantic segmentation ground truth.

## Results
TODO


[1]: https://arxiv.org/pdf/2106.12449.pdf
[2]: https://arxiv.org/pdf/2112.12610.pdf
[3]: https://github.com/LikeLy-Journey/SegmenTron
[4]: https://arxiv.org/pdf/2008.01550.pdf
