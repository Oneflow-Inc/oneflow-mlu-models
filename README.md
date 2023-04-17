# oneflow-cambricon-models

## 下载代码并安装

```shell
git clone --recursive https://github.com/Oneflow-Inc/oneflow-cambricon-models.git
cd oneflow-cambricon-models/libai
pip install pybind11
pip install -e .
```

## ResNet50

在 MLU 370 上使用 ResNet50 训练和推理 ImageNet，详细文档见[这里](resnet50-imagenet/README.md)。

需要先将路径切换到 oneflow-cambricon-models/resnet50-imagenet/ 下。

#### 数据集准备

需要 ImageNet 数据集，并且默认存放在 /ssd/dataset/ImageNet/extract 目录下，如果您的 ImageNet 数据集在其他路径中存放，请在脚本后面输入数据集路径来指定，详情见下方**其他选项**中描述。

#### OneFlow 对应代码修改

由于 mlu 暂时不支持 stream_touch 算子，所以需要注释掉 `oneflow/python/oneflow/framework/distribute.py` 中的这三行[代码](https://github.com/Oneflow-Inc/oneflow-cambricon/blob/73870cbecc9caf0258ca38a01a13f9544176f2e4/python/oneflow/nn/parallel/distributed.py#L189-L191)。注释掉这两行代码不会影响精度正确性。

```python
...
    def post_forward_hook(module, input, output):
        ddp_state_for_reversed_params = module._ddp_state_for_reversed_params
        for state in ddp_state_for_reversed_params.values():
            state[0], state[1] = False, False
        output = ArgsTree(output).map_leaf(
            lambda x: flow._C.select_top_n(
                convert_to_tensor_tuple([x, *ddp_state_for_reversed_params.keys()]),
                n=1,
            )[0]
        )
        # 注释掉下面这三行代码
        # buffers = list(module.buffers())
        # if len(buffers) > 0:
        #     flow._C.stream_touch(buffers)
        return output
...
```

#### 训练

训练默认使用 ImageNet 数据集，和 ResNet50 网络。

单卡训练
```shell
python3 main.py
```

多卡 DDP 训练

```shell
python3 -m oneflow.distributed.launch --nproc_per_node 4 main.py --multiprocessing-distributed
```

#### 推理

推理只需要在训练命令的基础上加上 `-e` 选项即可，此时会在 imagenet 的验证集上进行推理。

单卡推理
```shell
python3 main.py -e
```
多卡 DDP 推理
```shell
python3 -m oneflow.distributed.launch --nproc_per_node 4 main.py --multiprocessing-distributed -e
```

#### benchmark

benchmark 模式下，模型会使用固定的数据只进行前向推理，该模式的目的是测试显卡的FLOPS。默认内存排布为 NCHW。
```shell
python3 main.py --benchmark
```

#### 其他选项
- 如果需要在训练、推理或者 benchemark 时使用 NHWC 布局，可以使用 `--channels-last` 选项开启。
```shell
python3 main.py --channels-last
```
- 如果需要更改网络，可以使用 `-a` 选项。例如要训练 ResNet18，则可以运行如下命令。可以支持 flowvision.models 中的网络。

```shell
python3 main.py -a resnet18
```

- 如果 ImageNet 数据集没有放在 /ssd/dataset/ImageNet/extract 下，可以手动指定位置，例如
```shell
python3 main.py /path/to/your/dataset/extract
```
- 如果没有 ImageNet 数据集，可以使用 `--dummy` 来生成随机数据作为训练数据。
```shell
python3 main.py -a resnet50 --dummy
```
- 其他常用选项有：`-b` 设置 batch size，`-p` 设置多少 iter 输出一次日志，`--lr` 设置学习率等。详细操作可以阅读[文档](resnet50-imagenet/README.md)

---

## libai-gpt2

在 MLU 370 上使用 GPT2 微调以及推理 StableDiffusion 的咒语。

#### 推理

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
#### 训练微调

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
