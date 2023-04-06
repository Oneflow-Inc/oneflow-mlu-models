## oneflow-cambricon-models

### resnet50-imagenet

在 MLU 370 上使用resnet50训练和推理imagenet，详细文档见[这里](resnet50-imagenet/README.md)

### 数据集准备

把 imagenet 数据集放在 oneflow-cambricon-models/ 下即可。

### 训练

训练默认使用 imagenet 数据集，和 ResNet18 网络。

单卡训练
```shell
python3 main.py
```

多卡 DDP 训练

```shell
python3 -m oneflow.distributed.launch --nproc_per_node 4 main.py --multiprocessing-distributed
```

### 推理

推理只需要在训练命令的基础上加上 `-e` 选项即可

单卡推理
```
python3 main.py -e
```
多卡 DDP 推理
```
python3 -m oneflow.distributed.launch --nproc_per_node 4 main.py --multiprocessing-distributed -e
```

### 其他选项
- 如果需要更改网络，可以使用 `-a` 选项。例如要训练 ResNet50，则可以运行如下命令。可以支持 flowvision.models 中的网络。
```
python3 main.py -a resnet50
```

- 如果 imagenet 数据集没有放在 oneflow-cambricon-models/ 下，可以手动指定位置，例如
```
python3 main.py /path/to/your/dataset
```
- 如果没有 imagenet 数据集，可以使用 `--dummy` 来生成随机数据作为训练数据。
```
python3 main.py -a resnet50 --dummy
```
- 其他常用选项有：`-b` 设置 batch size，`-p` 设置多少 iter 输出一次日志，`--lr` 设置学习率等。详细操作可以阅读[文档](resnet50-imagenet/README.md)

---

### libai-gpt2

在 MLU 370 上使用 GPT2 微调以及推理 StableDiffusion 的咒语。

### 推理步骤如下

```shell
git clone --recursive https://github.com/Oneflow-Inc/oneflow-cambricon-models.git
cd oneflow-cambricon-models/libai
pip install pybind11
pip install -e .
```

libai的gpt2推理实现是在projects/MagicPrompt文件夹中，这个Magicprompt是我们自己用gpt2预训练后做推理的项目，用于将一个简单的句子转换成stable diffusion的咒语。接着把从 `https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-model.zip` 这里下载的模型解压到任意路径，并在 libai/ 下全局搜索`/data/home/magicprompt`将其替换为解压后的模型路径，我们就可以跑起来gpt2生成咒语的推理脚本了。

单卡模式运行命令如下：

```shell
python3 -m oneflow.distributed.launch projects/MagicPrompt/pipeline.py 1
```

4卡模型并行+流水并行运行命令如下：

```shell
"""
修改libai/projects/MagicPrompt/pipeline.py 94-94行
data_parallel=1,
tensor_parallel=2,
pipeline_parallel=2,
pipeline_stage_id=[0] * 6 + [1] * 6,
pipeline_num_layers=12,
"""
python3 -m oneflow.distributed.launch --nproc_per_node 4 projects/MagicPrompt/pipeline.py
```
### 微调步骤如下

训练只需要下载数据集 (`wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/libai/magicprompt/magicprompt.zip`) 并且修改下`projects/MagicPrompt/configs/gpt2_training.py`第13到15行的路径就可以了。然后执行下面的命令进行单卡的训练：

```shell
bash tools/train.sh tools/train_net.py projects/MagicPrompt/configs/gpt2_training.py 1
```

如果要训练多卡的数据，则只需要调整 `libai/projects/MagicPrompt/configs/gpt2_training.py` 第67行的dist即可，比如4卡的数据并行可以将其调整为：

```shell
dist=dict(
            data_parallel_size=4,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            # pipeline_num_layers = 12,
            # custom_pipeline_stage_id = [0] * 6 + [1] * 6,
            # pipeline_num_layers=model.cfg.hidden_layers,
        ),
```

然后使用如下的命令进行训练：

```shell
bash tools/train.sh tools/train_net.py projects/MagicPrompt/configs/gpt2_training.py 4
```
