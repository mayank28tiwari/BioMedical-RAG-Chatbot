import gradio as gr
import webbrowser
from embd import load_to_chroma
from textgenerator import generate_answer
from imagegenerator import BiomedicalImageGenerator



generator = BiomedicalImageGenerator()


def rag_response_pipeline(query, edit_prompt=""):

    chunks = load_to_chroma(query)
    answer = generate_answer(query)

    if edit_prompt.strip():
        image = generator.edit_image(generator.generate_image(query), edit_prompt)
    else:
        image = generator.generate_image(query)

    image_path = generator.save_image(image)


    return answer, image_path, "\n\n".join(chunks)


def launch_app():
    with gr.Blocks(title="Biomedical RAG Chatbot") as demo:
        gr.Markdown("## 🧬 Biomedical Assistant with Text & Image Generation")

        with gr.Row():
            question = gr.Textbox(label="Enter your biomedical question", lines=2, placeholder="e.g. How does mRNA vaccine work?")
            edit_prompt = gr.Textbox(label="Image Edit Prompt (Optional)", placeholder="e.g. Show the structure in 3D")

        with gr.Row():
            submit = gr.Button("Generate Answer + Image")

        with gr.Row():
            answer = gr.Textbox(label="Generated Answer")
            retrieved = gr.Textbox(label="Retrieved Context (Top Chunks)")

        image_output = gr.Image(label="Generated Biomedical Image")

        submit.click(
            fn=rag_response_pipeline,
            inputs=[question, edit_prompt],
            outputs=[answer, image_output, retrieved]
        )

    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, share=False)
    # webbrowser.open("http://127.0.0.1:7860")

    webbrowser.open("http://127.0.0.1:7860")





if __name__ == "__main__":
    launch_app()
