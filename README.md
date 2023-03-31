## oneflow-cambricon-models

### resnet50-imagenet

在 MLU 370 上使用resnet50训练和推理imagenet，详细操作步骤见[这里](resnet50-imagenet/README.md)

### libai-gpt2

在 MLU 370 上使用 GPT2 微调以及推理 StableDiffusion 的咒语。

### 推理步骤如下

```shell
git clone --recursive https://github.com/Oneflow-Inc/oneflow-cambricon-models.git
cd oneflow-cambricon-models/libai
pip install pybind11
pip install -e .
```

libai的gpt2推理实现是在projects/MagicPrompt文件夹中，这个Magicprompt是我们自己用gpt2预训练后做推理的项目，用于将一个简单的句子转换成stable diffusion的咒语。接下来，我将详细展示跑GPT2推理的步骤：

接着，新建一个文件夹，并把 https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion/tree/main 下的4个画红色框的文件下载到这个文件夹中，假设这个文件夹路径为`/home/zhangxiaoyu/onefow_gpt2_model`

<img width="392" alt="图片" src="https://user-images.githubusercontent.com/35585791/226298362-1fc67e84-d403-41a9-b107-a53a608345e5.png">

接着，我们修改 libai/projects/MagicPrompt/configs/gpt2_inference.py 中 `/data/home/magicprompt` 为 `/home/zhangxiaoyu/onefow_gpt2_model` ，同时修改 `libai/projects/MagicPrompt/pipeline.py` 中的第99行为`model_path="/home/zhangxiaoyu/onefow_gpt2_model"`。然后我们需要从https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/libai/magicprompt/OneFlow-MagicPrompt-Stable_Diffusion.zip 这里下载模型解压到`/home/zhangxiaoyu/onefow_gpt2_model`文件夹中，这样我们就可以跑起来gpt2生成咒语的推理脚本了。

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

如果要训练多卡的数据，则只需要调整 `libai/projects/MagicPrompt/configs/gpt2_training.py` 第67行的dist即可，比如数据并行可以将其调整为：

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
