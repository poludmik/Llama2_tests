# from langchain.embeddings import HuggingFaceEmbedding # Only works with sentence_transformers, Llama is not there.
import torch
from langchain.llms import HuggingFacePipeline


def get_embeddings_with_HuggingFaceModel():
    """
    Model from hugging face (...-7b-hf). Take outputs and then make average of token embeddings.
    """
    from transformers import AutoModel, AutoTokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf") #.to(device)
    sentence = "How is it going?"
    inputs = tokenizer(sentence, return_tensors="pt") #.to(device)
    outputs = model(**inputs)
    print(outputs[0].shape)


def get_answer_from_HuggingFaceModel():
    """
    To get the text response from a model that is downloaded here (from HuggingFace).
    """
    from transformers import LlamaForCausalLM, LlamaTokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf") #.to(device)
    sentence = "Q: I have two blue apples and three yellow bananas. What color are my apples? A: "
    inputs = tokenizer(sentence, return_tensors="pt") #.to(device)
    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(" ".join(result))


def get_embeddings_with_LlamaCpp_library(): # needs a local .bin model
    from langchain.embeddings import LlamaCppEmbeddings
    llama = LlamaCppEmbeddings(model_path="/home/poludmik/virtual_env_project/llama/from_hf_2-7b-chat/ggml-model-q4_0.bin", n_ctx=2048) # n_ctx is context window, max length of input
    query_result = llama.embed_query("The fact that artificial intelligence has learned to draw is nothing. Think about what will happen when he is not accepted into the Vienna Academy of Arts.")
    print(len(query_result))


def get_answer_with_Llama_from_llama_cpp_library(): # needs a local .bin model
    from llama_cpp import Llama
    question = "Q: Why did I spawn in russia? A: "
    llm = Llama(model_path="/home/poludmik/virtual_env_project/llama/llama-2-7b-chat/ggml-model-f32.bin")
    output = llm(question, max_tokens=60, stop=["Q:", "\n"], echo=True)
    print(output)


def get_answer_with_langchain_llm_LlamaCpp(): # needs a local .bin model
    from langchain.llms import LlamaCpp
    question = "Q: Why did I spawn in russia? A: "
    llm = LlamaCpp(model_path="/home/poludmik/virtual_env_project/llama/llama-2-7b-chat/ggml-model-f32.bin")
    output = llm(question, max_tokens=60, stop=["Q:", "\n"], echo=True)
    print(output)


def get_answer_with_HuggingFacePipeline():
    from langchain.llms import HuggingFacePipeline
    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10},
        )
    # and use it in RetrievalQA...
    


if __name__ == "__main__":
    # meta-llama/Llama-2-7b
    # embeddings = HuggingFaceEmbeddings(model_name="meta-llama/Llama-2-7b-hf")


    