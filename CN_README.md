![](./figures/宣传0.jpg)

# 项目简称：Alpaca-CoT

# 项目标题：Evolving Alpaca: An Empirical Study on Instruction Tuning for Large Language Models

## 1. 项目定位

ChatGPT的出现验证了大型语言模型(LLM)在通用人工智能(AGI)上的潜力。基于LLaMA[1]等Large Language Models(LLMs)的instruction-tuning研究(如，Alpaca [2])大幅度加速了复现ChatGPT的进程。**Alpaca-CoT**希望在这个研究方向上做出适度的贡献，以推进LLMs的开源进程、降低LLMs研究和使用成本。

具体来说，**Alpaca-CoT**项目旨在探究如何更好地通过instruction-tuning的方式来诱导Large Language Models（LLMs）具备类似ChatGPT的交互和instruction-following能力。为此，我们广泛收集了不同类型的instruction（尤其是CoT数据集），并基于LLaMA给出了深入细致的实证研究，以供未来工作参考。


热烈欢迎您向我们提供任何未被本项目收集的instruction-tuning及各类tasks数据集（或其来源）。我们将：
- 将这些数据收录并进行统一格式化处理；
- 用这些数据集instruct fine-tune LLaMA模型（未来将集成更多LLMs），并开源其checkpoint；
- 进行广泛的实证研究以探究新收录的数据集的作用。
我们希望我们的项目能够为大型语言模型的开源过程做出适度的贡献，并降低NLP研究人员上手研究LLMs的门槛。

## 2. 概述
