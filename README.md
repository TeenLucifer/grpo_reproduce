# GRPO vs GSPO
本项目使用deepseek grpo和qwen gspo方法对qwen2.5-1.5B-Instruct模型在GSM8K数据集上的全量微调, 完整地复现了[GRPO算法](https://arxiv.org/pdf/2402.03300)和[GSPO算法](https://arxiv.org/pdf/2507.18071)进行对比, 包括旧策略采样、参考策略采样和新策略训练. 本项中搭建的分布式训练框架适合off policy方法与deepspeed结合进行LLM分布式微调.

## 训练框架
![框图](./docs/framework.png)

项目主要分为采样进程和训练进程:
* 采样进程: 旧策略轨迹采样 + 旧策略概率分布推理 + 新策略的概率分布推理
* 训练进程: deepspeed自动fork多个子进程Rank_n, 各子进程中进行数据分割及分布式训练
* 训练数据传递: 采样进程->训练进程, 用zmq实现
* 模型参数同步: 训练进程->采样进程, 用文件系统实现

## GRPO算法原理
GRPO在PPO算法的基础上改进, PPO算法为Actor-Critic网络结构, 需要同时训练Actor和Critci两个网络, 对于动辄几十亿参数的大语言模型来说训练开销太大. GRPO方法的优化点在于对同一问题采样多次, 称为一组, 用组内回报的均值来代替Critic网络, 减少了一半训练开销.

目标函数:

$$
J_{GRPO}(\theta)=E\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{\left|y_i\right|} \sum_{t=1}^{\left|y_i\right|} \min \left(w_{i, t}(\theta) \hat{A}_{i, t}, \text{clip}\left(w_{i, t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{i, t}\right)\right]
$$
$$
w_{i, t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i<t})}{\pi_{\theta_{old}}(y_{i,t} \mid x, y_{i<t})}
$$
$$
\hat{A}_{i, t}=\hat{A}_i=\frac{r(x, y_i)-\text{mean}(\{r(x, y_i)\}_{i=1}^G)}{\text{std}(\{r(x, y_i)\}_{i=1}^G)}
$$

其中$w_{i,t}$表示token级别的重要性采样, $\hat{A}_{i}$表示序列的组内回报值:

目标函数和Loss函的核心代码实现:
```python
ref_policy_log_probs_ = ref_policy_log_probs[:, prefix_len-1:] # 参考策略概率分布
old_policy_log_probs_ = old_policy_log_probs[:, prefix_len-1:] # 旧策略概率分布
new_policy_log_probs_ = new_policy_log_probs[:, prefix_len-1:] # 新策略概率分布
attention_mask_       = attention_mask[:, prefix_len:]

importance_ratio = torch.exp(new_policy_log_probs_ - old_policy_log_probs_) # 重要性采样
cliped_ratio = torch.clip(importance_ratio, 1 - clip_epsilon, 1 + clip_epsilon) # 相似度裁剪
importance_term = importance_ratio * advantages
clip_term = cliped_ratio * advantages

kl_term = torch.exp(ref_policy_log_probs_ - new_policy_log_probs_) - (ref_policy_log_probs_ - new_policy_log_probs_) - 1 # kl散度

objective_function = torch.min(importance_term, clip_term) - kl_beta * kl_term # 目标函数
per_token_loss = -objective_function # loss函数

loss = ((per_token_loss * attention_mask_).sum(dim=1) / attention_mask_.sum(dim=1)).mean() # batch的均值作为最终loss(只统计有效token的loss)
```

## GSPO算法原理
GSPO算法是Qwen团队最新提出的RLHF算法, 在GRPO算法基础上进行改进, 设计动机是为了解决GRPO算法序列级别的reward与token级别的重要性采样值颗粒度不对齐导致的不稳定性问题. GSPO算法的改进点为把重要性采样部分调整为序列级别. 带来了两点优势:
* 降低token方差, 训练过程更为稳定, 用几何均值计算序列重要性采样, 能够有效缩小token的方差, 使训练过程更加稳定.
* 对于MoE架构模型, 不再需要routing replay, 因为序列重要性天然包含对专家路由的边缘积分, 专家路由与生成模型的联合概率分布变为边缘概率分布, 可以直接进行重要性采样.

$$
J_{GSPO}(\theta)=E\left[\frac{1}{G} \sum_{i=1}^G \min \left(s_i(\theta) \hat{A}_i, \text{clip}\left(s_i(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_i\right)\right]
$$
$$
s_i(\theta)=\left(\frac{\pi_\theta\left(y_i \mid x\right)}{\pi_{\theta_{\text {old }}}\left(y_i \mid x\right)}\right)^{\frac{1}{\left|y_i\right|}}=\exp \left(\frac{1}{\left|y_i\right|} \sum_{t=1}^{\left|y_i\right|} \log \frac{\pi_\theta\left(y_{i, t} \mid x, y_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(y_{i, t} \mid x, y_{i,<t}\right)}\right)
$$

其中$s_i(\theta)$表示序列重要性采样, 与序列组内回报$\hat{A}_i$颗粒度是对齐的.

目标函数和Loss函数的核心代码实现:
```python
batch_size = ref_policy_log_probs.shape[0]

# 取生成部分的概率分布
ref_policy_log_probs_ = ref_policy_log_probs[:, prefix_len-1:] # token_0裁剪了, 因此需要裁剪的长度为prefix_len-1
old_policy_log_probs_ = old_policy_log_probs[:, prefix_len-1:]
new_policy_log_probs_ = new_policy_log_probs[:, prefix_len-1:]
attention_mask_       = attention_mask[:, prefix_len:]         # attention_mask维度中token_0的位置没裁剪, 因此需要裁剪的长度为prefix_len

# 计算有效序列, 遮掩pad_token
valid_seq_len = attention_mask_.sum(dim=1)
new_old_log_probs_ = (new_policy_log_probs_ - old_policy_log_probs_) * attention_mask_
ref_new_log_probs_ = (ref_policy_log_probs_ - new_policy_log_probs_) * attention_mask_

# 序列级别的重要性采样
importance_ratio = torch.exp(new_old_log_probs_.sum(dim=1) / valid_seq_len).view(batch_size, 1) # batch_size * 1
cliped_ratio = torch.clip(importance_ratio, 1 - clip_epsilon, 1 + clip_epsilon) # batch_size * 1
importance_term = importance_ratio * advantages # batch_size * 1
clip_term = cliped_ratio * advantages # batch_size * 1

kl_term = torch.exp(ref_new_log_probs_.sum(dim=1) / valid_seq_len) - (ref_new_log_probs_.sum(dim=1) / valid_seq_len) - 1
kl_term = kl_term.view(batch_size, 1)

objective_function = torch.min(importance_term, clip_term) - kl_beta * kl_term
sequence_loss = -objective_function

# 批次平均损失作为总损失
loss = sequence_loss.mean()
```

## 数据集
GSM8K数据集是由8.5K个高质量的小学数学问题组成的语言模型训练数据集. 每个问题包含"question"和"answer"两个字段, answer中给出了问题的推理过程和最终的答案. 单个数据示例如下所示:

```
question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```

### 对话格式
在提示词中要求模型回复中需要包含思考过程和答案
* 思考过程需要用标签\<think\>(思考过程)\</think\>标记
* 答案需要用标签\<answer\>(答案)\</answer\>标记

### 奖励函数
* 答案奖励: 答案正确奖励+1, 错误奖励-1
* 格式奖励: 格式正确奖励+1.25, 错误奖励-1

## 效果展示
* 参考模型: qwen2.5-1.5B-Instruct
* 目标模型: qwen2.5-1.5B-Instruct
* 硬件配置: 3 × AutoDL vGPU-32G (GPU0/1用于训练, GPU2用于采样)
* 训练步数: 200 steps (60min)

![GRPO vs GSPO](./docs/grpo_vs_gspo.png)
准确率评估包含答案和格式两部分:

* GSPO算法在50个训练步左右基本稳定并到达峰值, 答案准确率为0.6左右, 格式准确率为0.99左右
* GRPO算法在120个训练步左右基本稳定并到达峰值, 答案准确率为0.6左右, 格式准确率为0.99左右

从结果来看GSPO训练速度明显优于GRPO, 消耗更少的时间达到稳定状态. 从模型特性来解释, GSPO模型训练时方差更小, 在矫正输出分布时有更强的确定性能够快速调整, 宏观上体现为更快得收敛至稳定值. 训练至200步后两种方法训练的结果基本接近, 应该是达到模型极限.

![entropy_comparison](./docs/entropy_comparison.png)

也可以从熵的角度来对比训练结果, 熵表示概率分布多样化的程度, 分布越分散熵越大. 从上面图中可以看到, 在200步的训练过程中, GSPO相比于GRPO的输出序列平均熵降低更快. 说明在微调任务中GSPO能够更快逐渐趋于某个人类偏好. 单纯看熵并不能直接说明方法的优劣, 只能说明分布的分散程度, 可以用熵结合reward来对比评估方法.

### LoRA
本项目应用了LoRA技术进行显存优化, 相对于全量微调显存占用极大减少, 梯度及优化器参数约为全量的3%. 单卡batch_size=4, 全量微调和LoRA微调的显存占用情况为:
<table>
<tr>
<td><img src="./docs/low_peak_full_bs8.png" alt="full" width="400"/></td>
<td><img src="./docs/low_peak_lora_bs8.png" alt="lora" width="400"/></td>
</tr>
<tr>
<td align="center">全量微调显存占用</td>
<td align="center">LoRA微调显存占用</td>
</tr>
</table>

## 项目部署
```python
# config.yaml
# 用grpo算法训练
training:
  use_gspo: false
# 用gspo算法训练
training:
  use_gspo: true
```

```bash
# 依赖安装
pip install -r requirements.txt
# GSM8K数据集下载
git clone https://huggingface.co/datasets/openai/gsm8k
# Qwen2.5-1.5B-Instruct模型下载(huggingface)
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
# Qwen2.5-1.5B-Instruct模型下载(modelscope)
git clone https://www.modelscope.cn/Qwen/Qwen2.5-1.5B-Instruct.git
# 启动采样进程
python sampling_worker.py
# 启动训练进程
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 training_worker.py
```

## 踩坑记录
* 工程上实现的生成序列包含人为填充的pad token, 需要将pad token去掉计算有效序列的概率分布, 防止引入无意义的概率噪声.
* AutoDL vGPU进行分布式训练时后台通信不能用默认的nccl, 需要改为gloo, nccl仅支持物理GPU的通信.
* off policy方法涉及到旧策略、新策略等多个模型, 通常采样与训练分布不同进程中, 采样数据传输到训练进程后需要手动进行数据并行, 给deepspeed fork的各子进程手动分配数据, 否则会造成单卡数据过多且重复.
* 从训练进程同步模型参数至采样进程时仅在主进程中传递即可, 否则会重复传递造成资源浪费.
* LLM进行RLHF训练微调时若需要同步模型参数, 可以用文件系统实现. LLM通常参数量较大, 若用网络传递速度较慢且容易失败.
* 在训练模式下前向传播时显存占用过高, 显存占用会出现一个尖峰, 非常容易OOM, 解决方法是在加载模型时开启Gradient Checkpoint `model.gradient_checkpointing_enable()`. 具体表现如下图所示, 开启前峰值显存约为17GB左右, 开启后5GB左右, 有效地防止了OOM的现象, 同时也允许更大的batch_size.
* LoRA微调时需要更大的步长, 比如: 2e-4. 用LoRA训练时参数被约束在一个低秩子空间, 在受限的空间中找到最优解相对于全量微调需要更多的迭代步骤, 因此可以考虑更大的步长加快收敛.

<table>
<tr>
<td><img src="./docs/high_peak_full_bs8.png" alt="high_peak" width="400"/></td>
<td><img src="./docs/low_peak_full_bs8.png" alt="low_peak" width="400"/></td>
</tr>
<tr>
<td align="center">未开启Gradient Checkpointing</td>
<td align="center">开启Gradient Checkpointing</td>
</tr>
</table>

## 参考资料
本项目基于以下优秀项目实现, 在此进行感谢
* [GRPO论文](https://arxiv.org/pdf/2402.03300) 提供了理论基础, 组内相对奖励的设计极大的减少了训练开销.
* [GSPO论文](https://arxiv.org/pdf/2507.18071) 提供了理论基础, 序列重要性采样与序列reward进行颗粒度对齐稳定了训练过程并提升训练效率.
* [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) 用清晰的代码逻辑实现了GRPO方法, 并且手搓transformer网络、qwen模型结构和AdamW, 是一份非常优秀的示例代码.
* [simple_GRPO](https://github.com/lsdefine/simple_GRPO) 提供了采样-训练双进程实现的思路, 并且以非常简介的形式复现了GRPO方法.
* [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) 提供了高质量的qwen系列预训练模型.
* [GSM8K](https://huggingface.co/datasets/openai/gsm8k) 提供了高质的问答数据.