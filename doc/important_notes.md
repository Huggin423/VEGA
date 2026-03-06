### 开发背景
我正在学校的实验室服务器开发毕业论文的实验，但是由于该服务器的系统是Ubuntu18.05，导致glibc的版本过低，无法使用vscode的ssh服务或任何vibe coing插件。
而我没有root权限，只被限制于在 /root/mxy 文件夹下开发。因此使用Docker或更新服务器整体系统的方案是不可能的。

为了解决这个问题，方便我的后续实验，我将在本地开发关键代码文件，但是在实验室服务器上运行并验证。
毕业论文的实验代码在 /root/mxy/SWAB 下。由于上述原因，我没有迁移数据集文件夹 /data，模型文件夹 /model，以及模型运行数据集后的中间结果 /ptm/stats，它们的体积过大，本地机器无法支持。
假如后续需要了解完整的代码仓库，可以在 /doc/index.md 文件夹中查看，里面是在 /root/mxy/SWAB 下运行 _tree -L 4_ 的结果。
但我深知这样的不足，所以假如对未迁移文件内容有任何疑问，都可以暂停询问，我将进行说明或上传样例数据作为参考。

## 有关毕业论文的项目
我的论文题目是 __基于置信度评估的多模态预训练模型选择方法研究__ ，具体的开题报告可以在 /doc/opening_report.txt 文件查看，在此我简单介绍一下。

我的模型选择主要是针对VLM，在此之前已有一些工作，但是主要是围绕仅使用 text encoder，从而加快选择效率，代表性的方法有LOVM，SWAB（这两种方法的论文可以在 /doc 文件夹下的同名文件查看，我已将其识别为文本文件）。而VEGA（你可以在/doc/VEGA查看论文）使用到了image encoder，导师认为这方面的工作可以深挖，因此我的毕业论文项目需要同时考虑text encoder和image encoder。

虽然上述三种已有方法都没有对模型进行微调（也就是zero-shot直接根据下游任务选择最优模型），但是三种方法的 setting 各有不同。
1. LOVM仅使用 text encoder，不需要目标数据集图像，需要目标数据集文本的类别名+任务描述，依赖LLM生成的文本代理，核心约束是无图仅有文本描述。
2. SWAB是针对LOVM设定的改进方法，不需要目标数据集图像，需要类别名称，依赖LLM和开源数据集（需要开源数据集的图像统计信息迁移），核心约束是无图仅有文本描述（但利用开源图bridging）。
3. VEGA的范围更大，是无监督VLM选择。需要目标数据集图像，需要类别名称，但不依赖训练数据、标注或LLM，核心约束是有图无标签。

我的本意是基于VEGA去做一些优化，因为这种方法的使用场景更实际。但是VEGA的作者没有开源论文代码，所以我暂时使用SWAB的实验框架，一方面熟悉一下其他几篇论文的工作，另一方面看看有没有能用的数据集或模型集。
我想到的能做的优化有：
1. 基于对比思想的语义拓扑构建。因为VEGA似乎没有考虑到类别间的语义关联。
2. 无监督环境下的置信度校准。这点优化是关键，因为实际场景中的数据无真实标签，VEGA是依赖VLM生成的伪标签进行图结构中节点的初始化，这就需要考虑到VLM可能对OOD样本输出高置信度但错误的类别预测。

综上所述，我希望能利用AI辅助我完成毕业论文实验。

## 目前进展
1. 我发现SWAB虽然和VEGA的逻辑完全不同，但是像ptm_stats文件夹下的模型中间结果是可以直接拿来用的。
2. 导师建议我使用logME方法，先确认我的方法的上限，具体的代码文件在./hybrid_eval/run_oracle.py。

---

## 2026-03-01 代码框架重组

### 已完成工作

1. **新目录结构**:
   - `configs/`: 数据集和模型配置
   - `data/`: 统一数据加载接口
   - `methods/baseline/`: LogME和VEGA基线实现
   - `evaluation/`: 评估指标（Rank Correlation, Top-k Accuracy等）
   - `experiments/`: 实验运行脚本

2. **基线方法**:
   - LogME: 已实现，支持numba加速
   - VEGA: 基础框架完成，需进一步完善图匹配逻辑

3. **评估指标**:
   - Spearman/Kendall/Pearson相关系数
   - Top-k准确率
   - 加权Kendall's tau
   - MRR (Mean Reciprocal Rank)


### 数据说明

`ptm_stats/logits/` 包含约50个模型在22个数据集上的zero-shot logits：
- 可直接用于VEGA伪标签生成
- 需要配合features使用（从stats_on_hist_task整理）

### 注意事项

1. imagenet-1k和clevr_closest_object_distance的Oracle评估失败，暂不考虑
2. VEGA实现需要参考原论文完善图匹配逻辑
3. 当前框架与SWAB框架解耦，专注于"有图无标签"设定

---

## 2026-03-02 复现VEGA

### 现阶段问题

我突然意识到VEGA不能直接使用SWAB的中间结果。主要是因为SWAB的文本是caption后的特征，可能借用了LLM来扩写。而VEGA只是使用了类别的特征。当然不排除SWAB的中间结果更有利于更优的结果，但是目前我想先搞得简单一些，所以可能需要自己重新利用model中的模型去跑各个数据集data的文本特征。这样一来，需要先修改explore_data.py，增加对data和model的认识，然后重新在ptm_stats中增加一个文件夹存储这个阶段要使用的文本特征。

同时发现一个潜在的优化点，后续我再另外实现。就是文本特征其实可以用caption的平均值，但是现在为了简单一些自己重新跑一下类别的文本特征。

### 已完成工作

1. 复现VEGA。该项目已重命名为VEGA。因为3月1日只是搭建了框架，AI辅助生成的VEGA算法完全不是论文的意思，今天按照论文的内容重新复现，具体代码在 methods/baseline 文件夹下。目前methods/test_vega.py可以正常运行，7个基础测试均通过，接下来需要考虑和实际数据结合运行。

2. 当我实际在服务器上操作时，考虑到实际数据集data、模型集model、中间结果集ptm_stats的体积太大，我将完整的数据集保留在SWAB文件夹下，并建立符号链接。具体的服务器视角需要你结合参考index_update.md和index.md，其中index_update.md是服务器上的VEGA文件夹，index.md是服务器上的SWAB文件夹。

3.  进一步重组代码框架，并结合github。因为我在本地开发，所以需要先将更新的代码push到github上，再登陆服务器git pull最新的成果。期间解决了代码仓库嵌套的问题，增加了.gitmodules文件，这样就可以正常下载LogME_official仓库了。github会同步更新：https://github.com/Huggin423/VEGA.git

4. 重新利用已有的model和data跑一下文本特征，尽量符合原始VEGA的要求，后续可以尝试优化结合SWAB/LOVM中获取更优质文本特征的方式。

---

## 2026-03-03 融合新模型

### 已完成工作

1. 尝试尽量向VEGA论文的实验设置。目前数据集是完全覆盖的，但是模型比较麻烦。原论文中有三个benchmark。A是对CLIP家族模型的测试，B是对不同预训练算法获得的模型测试，C是结合不同提示词模板的测试。

---

## 2026-03-04 文本特征提取脚本优化

### 问题背景

运行 `scripts/extract_class_text_features.py` 时，发现多个模型无法成功加载，主要原因是 HuggingFace Hub 无法访问（大陆网络不稳定）。具体错误日志见 `scripts/vega_log.txt`。

### 已完成工作

1. **分离模型加载策略**：
   - `CLIP_FAMILY_MODELS`: 可直接通过 open_clip 加载的模型（laion400m 系列，17个）
   - `LAION_2B_MODELS_LOCAL`: 需要从本地缓存加载的 LAION 2B 模型（18个，包括 ConvNeXt、CoCa、ViT-H/g 等）

2. **新增本地缓存加载函数** `load_open_clip_from_cache()`：
   - 从 `SWAB/model/checkpoint/` 下的 HuggingFace 缓存格式加载模型
   - 自动查找 `snapshots` 目录

3. **实现 BLIP 模型支持**：
   - 使用 SWAB 已有的 `model/blip_class.py`
   - 从本地 checkpoint 加载：`model_base_retrieval_coco.pth`、`model_base_retrieval_flickr.pth`、`model_large_retrieval_coco.pth`、`model_large_retrieval_flickr.pth`

4. **实现 BEIT3 模型支持**：
   - 使用 SWAB 已有的 `model/beit_class.py`
   - 从本地 checkpoint 加载：`beit3_base_patch16_384_coco_retrieval.pth` 等

5. **为特殊模型添加 open_clip 方式加载**：
   ```python
   OPENCLIP_STYLE_MODELS = {
       "BioCLIP": ("ViT-B-16", "bioclip"),
       "BiomedCLIP": ("ViT-B-16", "biomedclip"),
       "QuiltNet": ("ViT-B-32", "quiltnet"),
   }
   ```

### 模型数量统计

| 类别 | 数量 | 说明 |
|------|------|------|
| Benchmark A (直接加载) | 17 | OpenAI CLIP + laion400m |
| Benchmark A (缓存加载) | 18 | LAION 2B、ConvNeXt、CoCa |
| Benchmark B | 10 | AltCLIP、SigLIP、BioCLIP 等 |
| BLIP + BEIT3 | 8 | 4 BLIP + 4 BEIT3 |
| **总计** | **53** | |

### 运行方式

```bash
# 查看配置
python scripts/extract_class_text_features.py --setup

# 运行提取
python scripts/extract_class_text_features.py

# 验证结果
python scripts/extract_class_text_features.py --verify
```

### 已生成的 class_text_feat 文件

截至目前，已成功生成 23 个模型的文本特征文件（见 `SWAB/ptm_stats/class_text_feat/`），还有部分模型因网络问题待处理。所有模型都优先从本地路径加载，避免 HuggingFace Hub 网络问题。
