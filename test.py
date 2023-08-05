# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
# from transformers import LlamaForCausalLM, LlamaTokenizer
from llama_cpp import Llama
from langchain.llms import LlamaCpp


if __name__ == "__main__":
    # meta-llama/Llama-2-7b
    # embeddings = HuggingFaceEmbeddings(model_name="meta-llama/Llama-2-7b")

    # tokenizer = LlamaTokenizer.from_pretrained("llama/from_hf/tokenizer.model")
    # tokens = np.array([tokenizer.encode("Hello this is a test")]).astype(int)
    # model = LlamaForCausalLM.from_pretrained("llama/llama-2-7b_bin/")
    # print(model.forward(input_ids=tokens, output_hidden_states=True))


    # Works
    # llama = LlamaCppEmbeddings(model_path="/home/poludmik/virtual_env_project/llama/llama-2-7b-chat/ggml-model-f32.bin")
    # query_result = llama.embed_query("The fact that artificial intelligence has learned to draw is nothing. Think about what will happen when he is not accepted into the Vienna Academy of Arts.")
    # print(len(query_result)) 


    question = "Q: Why did I spawn in russia? A: "

    llm = LlamaCpp(model_path="/home/poludmik/virtual_env_project/llama/llama-2-7b-chat/ggml-model-f32.bin")
    output = llm(question, max_tokens=60, stop=["Q:", "\n"], echo=True)
    print(output)

    llm = Llama(model_path="/home/poludmik/virtual_env_project/llama/llama-2-7b-chat/ggml-model-f32.bin")
    output = llm(question, max_tokens=60, stop=["Q:", "\n"], echo=True)
    print(output)

    