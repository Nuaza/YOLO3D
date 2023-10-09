# 3D目标检测的YOLO实现 YOLO For 3D Object Detection

#### Note
原作者目前创建了一个新的仓库，使用pytorch lightning技术和更多的骨干网络来提升YOLO3D的性能，目前正在开发中。具体可参考 [ruhyadi/yolo3d-lightning](https://github.com/ruhyadi/yolo3d-lightning)

关于作者 [Mousavian et al](https://arxiv.org/abs/1612.00496) 的论文 *3D Bounding Box Estimation Using Deep Learning and Geometry* 的非官方实现的一个fork. YOLO3D 的检测头使用的是YOLOv5，并使用Resnet18/VGG11作为回归器

![inference](docs/demo.gif)

## 安装
可以使用像anaconda或者docker镜像这样的虚拟环境进行安装，对于anaconda请使用以下命令：

### Anaconda虚拟环境
创建虚拟环境
```
conda create -n yolo3d python=3.8 numpy
```
安装1.8版本以上的Pytorch和对应的torchvision。如果你的GPU不支持，请参考 [Nelson Liu blogs](https://github.com/nelson-liu/pytorch-manylinux-binaries). 
```
pip install torch==1.8.1 torcvision==0.9.1
```
最后，安装相关依赖文件
```
pip install -r requirements.txt
```

### Docker引擎
Docker引擎是一种安装项目所需要的所有文件的最简单的方法。从仓库拉取docker镜像：
```
docker pull ruhyadi/yolo3d:latest
```
使用以下命令，由docker镜像运行docker容器：
```
cd ${YOLO3D_DIR}
./runDocker.sh
```
你将会进入docker容器的交互界面。可以使用以下代码来直接运行推理代码或者flask程序

### 下载预训练模型
若要运行推理或训练代码，可以下载预训练的Resnet18或VGG11模型。原作者已经训练了10个epochs。
```
cd ${YOLO3D_DIR}/weights
python get_weights.py --weights resnet18
```

## 推理
若要使用预训练模型进行推理，可以使用以下命令。它可以被运行在conda虚拟环境或者docker容器中。
```
python inference.py \
    --weights yolov5s.pt \
    --source eval/image_2 \
    --reg_weights weights/resnet18.pkl \
    --model_select resnet18 \
    --output_path runs/ \
    --show_result --save_result
```
也可以在Colab Notebook中运行推理程序，请参考 [这个链接](https://colab.research.google.com/drive/1vhgGRRDqHEqsrqZXBjBJHDFWJk9Pw0qZ?usp=sharing).

## 训练
YOLO3D模型可以使用Pytorch或Pytorch lightning进行训练。若要进行训练，你需要一个 [comet.ml](https://www.comet.ml) （用于可视化你的训练损失/准确度）的APi key。请参考comet.ml的官方文档以获得API key
```
python train.py \
    --epochs 10 \
    --batch_size 32 \
    --num_workers 2 \
    --save_epoch 5 \
    --train_path ./dataset/KITTI/training \
    --model_path ./weights \
    --select_model resnet18 \
    --api_key xxx
```
可以使用以下命令调用pytorch lightning进行训练：
```
!python train_lightning.py \
    --train_path dataset/KITTI/training \
    --checkpoint_path weights/checkpoints \
    --model_select resnet18 \
    --epochs 10 \
    --batch_size 32 \
    --num_workers 2 \
    --gpu 1 \
    --val_split 0.1 \
    --model_path weights \
    --api_key xxx
```

## 参考
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [shakdem/3D-BoungingBox](https://github.com/skhadem/3D-BoundingBox)

```
@misc{mousavian20173d,
      title={3D Bounding Box Estimation Using Deep Learning and Geometry}, 
      author={Arsalan Mousavian and Dragomir Anguelov and John Flynn and Jana Kosecka},
      year={2017},
      eprint={1612.00496},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### 汉化by [Nuaza](https://github.com/Nuaza)