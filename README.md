# MLC_Overview
This is an overview of Machine Learning Compilation (MLC), the process of optimizing machine learning models to enhance performance across various hardware platforms.

### TensorIR:
TensorIR stands for Tensor Intermediate Representation, and it refers to low-level, platform-independent representations of computational graphs or operations in machine learning frameworks, and it is used primarily in the context of optimization. 

One of the primary examples of TensorIR can be seen in the machine learning compilation framework TVM (Tensor Virtual Machine), which provides functionality to easily modify and transform primitive tensor functions (matrix multiplication, ReLU, element-wise operations), so as to take advantage of native hardware (reduce stride and maximize L1 caching, utilize GPUs or superscalars). TVM does this by providing labels for various aspects of computation, such as loop indicies, so as to provide metadata that can be used to optimize. These labels also allow for these tensor functions to be modified in certain ways, such as splitting up loop indicies, rearranging where computation occurs, or parallelizing. This is demonstrated in TensorIR.ipynb.  

### Computational Graph Abstraction:
We can represent neural networks as a computational graph, in which each node has a collection of inputs and an output which funnels directly into the input of the next node. This can be done using TVM's relax module, which provides functionality for easily calling TVM primitive tensor functions. Every node in a computational graph should be "side-effect free", which means that it solely takes inputs and produces outputs, without allocating memory or altering global states. This feature of computational graph nodes allows them to be modular and reorderable.

## Additional Resources:

This is a link to wonderful lectures by CMU professor Tianqi Chen on the topic:
[https://mlc.ai/summer22/schedule]

This is a link to notes on said lectures:
[https://mlc.ai/chapter_introduction/index.html]

This is a link to a MLC LLM (Machine Learning Compilation for Large Language Models) tutorial: 
[https://llm.mlc.ai/docs/get_started/introduction.html#introduction-to-mlc-llm]
