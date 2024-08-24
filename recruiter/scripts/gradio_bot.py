"""
This the bot with same functionality as the discord bot but with gradio instead of discord.
This is to show off the bot with less time and effort.
And this could be greatful while testing the bot.
"""

import gradio as gr
from recruiter.nlqs.workflow import main_workflow

def main():
    # Gradio interface
    with gr.Blocks(title="ResumeDB") as demo:
        gr.Markdown("# ResumeDB")

        chatbot = gr.Chatbot([], elem_id="chatbot", height=700)
        msg = gr.Textbox(show_copy_button=True)
        btn = gr.Button("Send")

        # clear = gr.ClearButton([msg, chatbot])

        msg.submit(main_workflow, [msg, chatbot], [msg, chatbot])
        btn.click(main_workflow, [msg, chatbot], [msg, chatbot])

    # Launch the interface
        try:
            demo.launch(debug=True, share=True)
        except:
            demo.launch(debug=True)

        return chatbot
