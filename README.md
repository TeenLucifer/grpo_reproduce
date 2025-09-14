# GRPO方法复现
微调qwen2.5-3B-Instruct模型, 用GRPO算法

## GRPO算法实现步骤
$$
J(\theta) = E_{\pi_\theta}\left[\min\left(\frac{P_\theta(a|s)}{P_{\theta'}(a|s)}\hat{A}_{\theta'}(s, a), \text{clip}\left(\frac{P_\theta(a|s)}{P_{\theta'}(a|s)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_{\theta'}(s, a)\right) - \beta D_{KL}(P_\theta, P_{ref}) \right]
$$

其中, $\hat{A}_{\theta'}(s, a)$为分组计算的回报值估算的优势, 用$\mathbf{r}$表示组内回报值, 具体计算方式为:
$$
\hat{A}_{\theta'}(s, a) = \mathbf{r} - \frac{\text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$

创建3个进程, 主进程用于训练, 其余两个一个用于旧策略采样数据, 一个用于参考策略推理

两块卡, A用于训练, B用于部署参考模型, 分为训练和推理两个程序运行

1. 训练程序: 分为pytorch训练新策略模型和vllm部署推理旧策略模型两个线程, 线程A1运行pytorch训练部分, 线程A2运行vllm部署推理旧策略为pytorch训练提供数据

2. 推理程序: 用vllm部署参考模型, 用于生成训练的参考分布, 通过fastapi的形式提供接口给训练程序调用

问答格式:
回答格式中需要包含思考过程和答案, 答案仅需要包含数字<think></think>, <answer></answer>

### importance sampling和kl divergence实现过程
.logits方法会获取通过前面token预测得到后一个token的概率分布, 而不是续写整个句子

最终loss会经过形状归一化, 归一化过程可导且可进行反向传播, 因此重要性采样和kl散度计算时可以用张量

1. $\pi_{old}$采样得到一批轨迹, 包括query和answer
2. 拼接query和answer, 得到trajectory
3. trajectory过一遍$\pi_{old}$获取每个token的概率分布
4. trajectory过一遍$\pi_{new}$获取每个token的概率分布
5. trajectory过一遍$\pi_{ref}$获取每个token的概率分布
6. $\pi_{old}$和$\pi_{new}$的token概率分布比值为重要性采样值(对应元素的token概率比值, 结果为一个张量)
7. $\pi_{new}$和$\pi_{ref}$的token概率分布比值计算KL散度

## GSM8K数据集
GSM8K数据集是由8.5K个高质量的小学数学问题组成的语言模型训练数据集. 每个问题包含"question"和"answer"两个字段, answer中给出了问题的推理过程和最终的答案. 数据实例如下所示:

```
question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```

## Reward Function设计
1. 答案奖励: 答案正确奖励+1, 错误奖励-1
2. 格式奖励: 格式正确奖励+1.25, 错误奖励-1

## 部署步骤
```bash
# GSM8K数据集下载
git clone https://huggingface.co/datasets/openai/gsm8k
# Qwen2.5-3B-Instruct模型下载
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
# Qwen2.5-7B-Instruct模型下载
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

## 显存估算
目标模型Qwen2.5-3B-Instruct, 参考模型Qwen2.5-7B-Instruct

FP16/BF16全量微调情况下需要60~70GB左右显存, LoRA微调需要30~40GB左右显存
### 参考模型部署显存
以Qwen2.5-7B-Instruct FP16/BF16为例

总显存开销 = 参数显存 + KV Cache + 激活值 + 框架开销 = 16~18GB

### 目标模型全量微调显存
| 配置                                  | FP32精度  | FP16/BF16精度 |
|---------------------------------------|---------|-------------|
| 模型参数(1份模型参数)                    | 12GB    | 6GB         |
| 梯度(1份模型参数)                       | 12GB    | 6GB         |
| AdamW优化器(F32 2份模型参数, 1阶矩+2阶矩) | 24GB    | 24GB        |
| 激活值等                               | 5-15GB  | 5-15GB      |
| 总计                                   | 53-63GB | 41-51GB     |
### 目标模型LoRA微调显存
LoRA旁路矩阵的参数量估算:
$$
n_{LoRA} = n_{total} \frac{2r}{d_{model}}
$$
$n_{LoRA}$表示LoRA的旁路矩阵, $n_{total}$表示模型的总参数量, $r$表示秩(旁路矩阵的维度), $d_{model}$表示模型的隐藏层维度.
对于Qwen2.5-3B模型来说$d_{model}=2048$, 以$r=32$为例, 用LoRA方法微调的显存估算为:
| 配置                                       | FP32精度   | FP16/BF16精度 |
|-------------------------------------------|------------|-------------|
| 模型参数(1份模型参数+1份旁路矩阵参数)          | 12GB+0.4GB | 6GB+0.2GB         |
| 梯度(1份旁路矩阵参数)                        | 0.4GB      | 0.2GB         |
| AdamW优化器(F32 2份旁路矩阵参数, 1阶矩+2阶矩) | 0.75GB      | 0.75GB        |
| 激活值等                                   | 5-15GB      | 5-15GB      |
| 总计                                       | 19-29GB    | 12-22GB     |

## 待完善:
0. 验证目标策略和采样策略的参数同步
1. 正确率评估
2. 达到某个条件结束训练
3. 全量微调结果查看
4. LoRA方案

启动命令
```bash
# 采样
python sampling.py

# 分布式训练
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 training_worker.py
```

## 参考资料
- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero)
- [simple_GRPO](https://github.com/lsdefine/simple_GRPO)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)