from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from retriever import retrieve_relevant_chunks


MODEL_NAME = "microsoft/BioGPT-Large" 
MAX_TOKENS = 1024



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()


def format_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"


def generate_answer(query: str) -> str:
    chunks = retrieve_relevant_chunks(query)
    prompt = format_prompt(query, chunks)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            top_k=5,
            temperature=0.3
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()

