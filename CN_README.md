![Alpaca-CoT](https://github.com/PhoebusSi/alpaca-CoT/blob/main/figures/Alpaca-CoT-2.jpg)
# Evolving Alpaca: An Empirical Study on Instruction Tuning for Large Language Models (**Alpaca-CoT**)

## 1. 项目定位

ChatGPT的出现验证了大型语言模型(LLM)在通用人工智能(AGI)上的潜力。基于LLaMA[1]等Large Language Models(LLMs)的instruction-tuning研究(如，Alpaca[2])大幅度加速了复现ChatGPT的进程。**Alpaca-CoT**希望在这个研究方向上做出适度的贡献，以推进LLMs的开源进程、降低LLMs研究和使用成本。

具体来说，**Alpaca-CoT**项目旨在探究如何更好地通过instruction-tuning的方式来诱导Large Language Models（LLMs）具备类似ChatGPT的交互和instruction-following能力。为此，我们广泛收集了不同类型的instruction（尤其是Chain-of-Thought数据集），并基于LLaMA给出了深入细致的实证研究，以供未来工作参考。据我们所知，我们是首个将CoT拓展进Alpaca的工作，因此简称为"**Alpaca-CoT**"。


热烈欢迎您向我们提供任何未被本项目收集的instruction-tuning及各类tasks数据集（或其来源）。我们将：
- 将这些数据收录并进行统一格式化处理；
- 用这些数据集instruct fine-tune LLaMA模型（未来将集成更多LLMs），并开源其checkpoint；
- 进行广泛的实证研究以探究新收录的数据集的作用。
我们希望我们的项目能够为大型语言模型的开源过程做出适度的贡献，并降低NLP研究人员上手研究LLMs的门槛。

## 2. 概述

近期，LLaMA[1]显示出惊人的zero-shot和few-shot能力，仅需具有较少的参数即可和GPT-3.5性能相当（LLaMA-13B显著优于GPT-3（175B），LLaMA-65B与PaLM-540MB相当），明显降低了训练、微调和使用competitive大型语言模型的成本。最近，为了提高LLaMA的instruction-following能力，Stanford Alpaca[2]利用self-instruct[3]生成的52K Englishi nstruction-finetuning数据对LLaMA进行了微调。然而，目前该方向的研究仍然面临着以下两个挑战：
- 1. 即使LLaMA-7b依然对计算资源有着较高的要求；
- 2. 用于instruction finetuning的开源数据集不够多，且缺乏各数据类型的实证研究；

为此，我们提出了Alpaca-CoT项目，该项目结合了相关的近期前沿技术，具有以下优势：
- 该项目可以更高效、此回购包含从[此处]修改的代码(https://github.com/tloen/alpaca-lora)，通过使用[低秩自适应（LoRA）]，可以廉价高效地**_finetune LLaMA _**（与Stanford Alpaca相比，性能不会下降）(https://arxiv.org/pdf/2106.09685.pdf)[4]，[PEFT](https://github.com/huggingface/peft)和[位和字节](https://github.com/TimDettmers/bitsandbytes). LLaMA模型的“7b”、“13b”和“30b”版本可以在单个80G A100上轻松训练。


[1]: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971v1)

[2]: [Stanford Alpaca: An Instruction-following LLaMA model](https://github.com/tatsu-lab/stanford_alpaca)

[3]: [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)

[4]: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)

[5]: [FLAN: Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)

[6]: [BELLE: Bloom-Enhanced Large Language model Engine](https://github.com/LianjiaTech/BELLE)
