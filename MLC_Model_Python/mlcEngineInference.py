from mlc_llm import MLCEngine

# Create engine
model = "JordanAI-bassAndChords-v0.1-MLCLLM"
model_lib = "JordanAI-bassAndChords-v0.1-MLCLLM/MLCModel.dylib"
engine = MLCEngine(model, model_lib=model_lib)

# Run chat completion in OpenAI API.
# messages=[{"role": "user", "content": "What is the meaning of life?"}],
for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "Very cool!"}],
    model=model,
    stream=True,
    max_tokens=10,
    seed=100
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")

engine.terminate()