# app.py (Place this in the root of your project)

import gradio as gr
import asyncio # Keep asyncio import, might be needed by Gradio internals

print("app.py: Script execution started.")
print(f"app.py: Gradio version: {gr.__version__}")

# Define the simple function
def greet(name):
    if not name:
        return "Please enter a name!"
    return f"Hello, {name} from Gradio {gr.__version__} (loaded via entrypoint)!"

# Define the Gradio Blocks interface directly
# Assign it to a variable (conventionally 'demo' or 'app', but any name works)
minimal_demo = gr.Blocks()
with minimal_demo:
    gr.Markdown("## Super Simple Gradio-lite Test (Loaded via entrypoint)")
    name_input = gr.Textbox(label="Enter your name", placeholder="e.g., Pyodide User")
    output_greeting = gr.Textbox(label="Greeting", interactive=False)
    greet_button = gr.Button("Greet")
    greet_button.click(fn=greet, inputs=name_input, outputs=output_greeting)

print("app.py: Blocks object 'minimal_demo' created.")

# NOTE: DO NOT call minimal_demo.launch() or gr.mount_gradio_app() here.
# Gradio-lite handles the mounting when using the `entrypoint` attribute.
# The Gradio Blocks instance (`minimal_demo`) just needs to be defined globally.