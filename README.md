# oneflow-mlu-models

## 下载代码并安装

```shell
git clone --recursive https://github.com/Oneflow-Inc/oneflow-mlu-models.git
cd oneflow-mlu-models/libai
pip install pybind11
pip install -e .
```



## 模型列表

* <a href="ResNet50">ResNet50</a>：在 MLU 370 上训练和评估ResNet50模型。

* <a href="Libai-GPT2">Libai-GPT2</a> ：在 MLU 370 上使用 GPT2 微调以及推理 StableDiffusion 的咒语。

  

## <a id="ResNet50">ResNet50</a>

切换路径到 oneflow-mlu-models/resnet50/。

#### 准备数据集

从 ImageNet 官网下载数据集，解压后存放在 /ssd/dataset/ImageNet/extract 目录下，如果您的 ImageNet 数据集在其他路径中存放，请在脚本后面指定数据集路径，详情见下方**其他选项**中描述。

#### 训练

训练默认使用 ImageNet 数据集，和 ResNet50 网络。

- 单卡训练

  ```shell
  python3 main.py --channels-last
  ```

- 4卡 DDP 训练

  ```shell
  python3 -m oneflow.distributed.launch --nproc_per_node 4 main.py --multiprocessing-distributed --channels-last
  ```

#### 评估

评估只需要在训练命令的基础上加上 `-e` 选项即可，此时会在 imagenet 的集上对模型进行评估。

- 单卡评估

  ```shell
  python3 main.py --channels-last -e
  ```

- 4卡 DDP 评估

  ```shell
  python3 -m oneflow.distributed.launch --nproc_per_node 4 main.py --multiprocessing-distributed --channels-last -e
  ```

#### 推理benchmark

benchmark 模式下，模型会使用固定的数据只进行前向推理，该模式的目的是测试显卡的FLOPS。
```shell
python3 main.py --channels-last --benchmark
```

#### 其他选项
- 如果需要更改网络，可以使用 `-a` 选项。例如要训练 ResNet18，则可以运行如下命令。可以支持 flowvision.models 中的网络。

  ```shell
  python3 main.py -a resnet18 --channels-last
  ```

- 如果 ImageNet 数据集没有放在 /ssd/dataset/ImageNet/extract 下，可以手动指定位置，例如

  ```shell
  python3 main.py /path/to/your/dataset/extract --channels-last
  ```
- 如果没有 ImageNet 数据集，可以使用 `--dummy` 来生成随机数据作为训练数据。

  ```shell
  python3 main.py -a resnet50 --channels-last --dummy
  ```
- 其他常用选项有：`-b` 设置 batch size，`-p` 设置多少 iter 输出一次日志，`--lr` 设置学习率等。详细操作可以阅读[文档](resnet50-imagenet/README.md)

---

## <a id="Libai-GPT2">Libai-GPT2</a>

切换路径到 oneflow-mlu-models/libai/。

#### 推理

libai的gpt2推理实现是在projects/MagicPrompt文件夹中，这个Magicprompt是我们自己用gpt2预训练后做推理的项目，用于将一个简单的句子转换成stable diffusion的咒语。

下载模型和数据（ `wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-model.zip` ），并解压到当前目录，之后我们就可以用gpt2来生成咒语了。

- 单卡推理

  ```shell
  python3 -m oneflow.distributed.launch projects/MagicPrompt/pipeline.py 1
  ```

- 4卡模型并行+流水并行推理

  修改文件`libai/projects/MagicPrompt/pipeline.py`，将tensor_parallel和pipeline_parallel都设置为2，

  ```python
  data_parallel=1,
  tensor_parallel=2,
  pipeline_parallel=2,
  ```

  之后运行命令如下进行推理，

  ```shell
  python3 -m oneflow.distributed.launch --nproc_per_node 4 projects/MagicPrompt/pipeline.py
  ```

#### 训练微调

下载数据集（`wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/libai/magicprompt/magicprompt.zip`），并解压到当前目录。

- 单卡训练

```shell
bash tools/train.sh tools/train_net.py projects/MagicPrompt/configs/gpt2_training.py 1
```

- 4卡数据并行训练

  修改文件`libai/projects/MagicPrompt/configs/gpt2_training.py`，将data_parallel_size设置为4。

  ```python
  data_parallel_size=4,
  tensor_parallel_size=1,
  pipeline_parallel_size=1,
  ```

  然后使用如下的命令进行训练：

  ```shell
  bash tools/train.sh tools/train_net.py projects/MagicPrompt/configs/gpt2_training.py 4
  ```
