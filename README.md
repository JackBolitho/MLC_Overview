# MLC_Overview
This is an overview of Machine Learning Compilation (MLC), the process of optimizing machine learning models to enhance performance across various hardware platforms.

### TensorIR:
TensorIR stands for Tensor Intermediate Representation, and it refers to low-level, platform-independent representations of computational graphs or operations in machine learning frameworks, and it is used primarily in the context of optimization. 

One of the primary examples of TensorIR can be seen in the machine learning compilation framework TVM (Tensor Virtual Machine), which provides functionality to easily modify and transform primitive tensor functions (matrix multiplication, ReLU, element-wise operations), so as to take advantage of native hardware (reduce stride and maximize L1 caching, utilize GPUs or superscalars). TVM does this by providing labels for various aspects of computation, such as loop indicies, so as to provide metadata that can be used to optimize. These labels also allow for these tensor functions to be modified in certain ways, such as splitting up loop indicies, rearranging where computation occurs, or parallelizing. This is demonstrated in TensorIR.ipynb.  

### Computational Graph Abstraction:
We can represent neural networks as a computational graph, in which each node has a collection of inputs and an output which funnels directly into the input of the next node. This can be done using TVM's relax module, which provides functionality for easily calling TVM primitive tensor functions. Every node in a computational graph should be "side-effect free", which means that it solely takes inputs and produces outputs, without allocating memory or altering global states. This feature of computational graph nodes allows them to be modular and reorderable.

### Additional Resources:

This is a link to wonderful lectures by CMU professor Tianqi Chen on the topic:
[https://mlc.ai/summer22/schedule]

This is a link to notes on said lectures:
[https://mlc.ai/chapter_introduction/index.html]

This is a link to a MLC LLM (Machine Learning Compilation for Large Language Models) tutorial: 
[https://llm.mlc.ai/docs/get_started/introduction.html#introduction-to-mlc-llm]

This is a link to a paper which uses MLC LLM in the context of generative music transformers: [https://arxiv.org/pdf/2411.09625]

This is a link to the GitHub of the MLC LLM Framework: [https://github.com/mlc-ai/mlc-llm]

This is a link to the implementation of the GPT-2 architecture with the MLC LLM framework: [https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/gpt2/gpt2_model.py]

This is a link to a directed graph representing the tvm class hierarchy: [https://tvm.apache.org/docs/reference/api/doxygen/inherits.html]


# How To Inference With MLC-LLM
MLC-LLM is a framework designed to make ML Chatbots more accessible. All parts of the framework are geared towards chatbots, which is great if you want to make a chatbot. The process for inferencing any MLC-LLM model starts downloading MLC-LLM from here: [https://llm.mlc.ai/docs/install/mlc_llm.html]. 

Once you download the MLC-LLM package, in order to create a model, we must first start by converting model weights (Pytorch, ONNX, etc.) to the MLC-LLM format. This is described here: [https://llm.mlc.ai/docs/compilation/convert_weights.html]

After that, compile the model functions into a .dylib, which can be found here: [https://llm.mlc.ai/docs/compilation/compile_models.html]

### Converting Weights and the Chatbot in Python
In order to use the python chatbot:

#### Convert the weights to an MLC acceptable format:
```bash
mlc_llm convert_weight ./JordanAI-bassAndChords-v0.1-pytorch --quantization q0f16 -o ./JordanAI-bassAndChords-v0.1-MLCLLM
```

#### Generate the MLC config:
```bash
mlc_llm gen_config ./JordanAI-bassAndChords-v0.1-pytorch --quantization q0f16 --conv-template LM -o ./JordanAI-bassAndChords-v0.1-MLCLLM
```

#### Compile into a .dylib binary given the weights and config:
```bash
mlc_llm compile ./JordanAI-bassAndChords-v0.1-MLCLLM/mlc-chat-config.json --device metal:0 -o ./JordanAI-bassAndChords-v0.1-MLCLLM/MLCModel.dylib
```

These commands and their syntax are listed here: [https://llm.mlc.ai/docs/compilation/compile_models.html#compile-command-specification] 

Then, we can create the MLC Engine with our given model directory and .dylib model binary. For the MLCEngine to work, the model directory must include a tokenizer.json, which determines how data is split up into tokens. We do not need to tokenize, since all of our MIDI inputs are already tokens.

The tokenizer we used as a test was found from https://huggingface.co/mlc-ai/mlc-chat-stanford-crfm-music-small-800k-q0f16/blob/main/tokenizer.json, which paired every possible vocabulary element with an adjacent token. This works because the model has the same vocabulary as ours. 

```python
from mlc_llm import MLCEngine
engine = MLCEngine(model="./JordanAI-bassAndChords-v0.1-MLCLLM", model_lib="./JordanAI-bassAndChords-v0.1-MLCLLM/MLCModel.dylib")
```

Once we have the engine, we can enact the chat completion to get the next predicted token. Content refers to the token that we input, and max tokens is the maximum number of tokens we generate from the response.

```python
response = engine.chat.completions.create(
    messages=[{"role": "user", "content": "EVENT_TIME_0"}] ,
    max_tokens=1
)
print(response)
```
The major problems with this current implementation are that they require a tokenizer.json and that the output is a token, not a logit. 


