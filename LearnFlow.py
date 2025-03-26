import gradio as gr
from openai import OpenAI
import os
import time
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("API_KEY")

# System prompt
default_system_prompt = """You are a kind and respectful career planning advisor..."""

MODEL_NAME = "gpt-4o"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
# ✅ 在此填写你的 OpenAI API Key

# 将 history 从 OpenAI 格式转为 Gradio 的 [[user, assistant], ...] 格式
def convert_to_chatbot_format(history):
    result = []
    user_msg = None
    for msg in history:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant":
            result.append([user_msg, msg["content"]])
            user_msg = None
    return result

# 支持流式的聊天预测
def predict(message, history):
    client = OpenAI(api_key=API_KEY)
    messages = [{"role": "system", "content": default_system_prompt}]
    messages.extend(history if history else [])
    messages.append({"role": "user", "content": message})

    start_time = time.time()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        stream=True
    )

    full_message = ""
    first_chunk_time = None
    last_yield_time = None

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            full_message += chunk.choices[0].delta.content
            current_time = time.time()
            if last_yield_time is None or (current_time - last_yield_time >= 0.25):
                chatbot_display = convert_to_chatbot_format(history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": full_message}
                ])
                yield chatbot_display, history
                last_yield_time = current_time

    if full_message:
        total_time = time.time() - start_time
        full_message += f" (First Chunk: {first_chunk_time:.2f}s, Total: {total_time:.2f}s)"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": full_message})
        yield convert_to_chatbot_format(history), history

# 固定消息按钮逻辑
def send_fixed_message(fixed_message):
    def inner(history):
        return predict(fixed_message, history)
    return inner

with gr.Blocks(css="""
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@700&display=swap');

#title {
    text-align: center;
    margin-top: 40px;
    margin-bottom: 30px;
    font-size: 80px;
    font-family: 'Fredoka', sans-serif;
    font-weight: 700;
}

#title .gradient-text {
    background: linear-gradient(90deg, #ff6ec4, #7873f5, #4ac29a);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
}

button.small-colored {
    font-size: 14px !important;
    padding: 8px 12px !important;
    border-radius: 8px;
    font-weight: bold;
    background-color: #a8e6cf;
    color: #000000;
    border: none;
    margin-bottom: 10px;
    width: 100%;
}

footer { display: none !important; }

#spinner {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    font-family: 'Fredoka', sans-serif;
    font-size: 20px;
    font-weight: bold;
    gap: 20px;
}

.spinner-circle {
  border: 6px solid #f3f3f3;
  border-top: 6px solid #ff6ec4;
  border-right: 6px solid #7873f5;
  border-bottom: 6px solid #4ac29a;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  animation: spin 1s linear infinite;
}

.typing-loader {
  display: inline-block;
}

.typing-loader .loader-text {
  color: #b19cd9;
  font-size: 24px;
}

.typing-loader .dot {
  animation: blink 1.4s infinite both;
  font-size: 32px;
  font-weight: bold;
  color: #4ac29a;
}

.typing-loader .dot:nth-child(2) {
  animation-delay: 0.2s;
}
.typing-loader .dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@keyframes blink {
  0% { opacity: 0.2; }
  20% { opacity: 1; }
  100% { opacity: 0.2; }
}

label {
    font-size: 18px !important;
}
input, select, textarea {
    font-size: 16px !important;
}
input[type="radio"] + span, select option {
    font-size: 14px !important;
}

div[data-testid="chatbot"] > div:first-child {
    font-size: 18px !important;
    font-weight: 600;
    color: #6a5acd;
}

div[data-testid="chatbot"] svg {
    display: none !important;
}

#subject_checkboxes input[type="checkbox"] + span {
    font-size: 14px !important;
}

#sidebar {
    background-color: #fff8f5;
    border-right: 2px solid #ffa500;
    padding-top: 20px;
    align-items: center;
    gap: 10px;
    height: 100%;
}

#interest-tab,
#program-tab {
    width: 100%;
    border: none;
    background: none;
    font-size: 14px;
    font-weight: bold;
    text-align: center;
    padding: 12px 5px;
    cursor: pointer;
    color: gray;
}

#interest-tab.selected,
#program-tab.selected {
    color: #ff6600 !important;
    background-color: #fff2ec !important;
    border-left: 4px solid #ff6600 !important;
}

#school-card {
    background-color: #fdfdfd;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
    margin: 8px 0;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr); /* 固定最多 5 列 */
    gap: 16px;
    margin-top: 20px;
}

.school-card {
    background-color: #ffffff;
    border-radius: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    padding: 16px;
    font-family: 'Fredoka', sans-serif;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 8px;
}

.school-card strong {
    font-size: 18px;
    font-weight: 700;
    color: #333333;
}

.school-card span {
    font-size: 16px;
    color: #444;
}

""") as demo:

    selected_tab = gr.State(value="interest")

    gr.HTML("""
    <div id="title">
        🎓 <span class="gradient-text">Career Planner Bot</span>
    </div>
    """)

    with gr.Column(visible=True) as intro_page:
        grade = gr.Dropdown(choices=["9th and below", "10th", "11th", "12th"], label="Grade")
        school = gr.Textbox(label="High School")
        province = gr.Textbox(label="Province")
        gender = gr.Radio(choices=["Female", "Male", "Other"], label="Gender")
        interests = gr.CheckboxGroup(
            label="Subjects you’re most interested in (optional)",
            choices=["Math", "Science", "History", "Literature", "Art", "Computer Science", "Economics", "Engineering", "Psychology", "Biology", "Other"],
            elem_id="subject_checkboxes"
        )
        start_button = gr.Button("Start Chat")

    with gr.Column(visible=False) as loading_page:
        gr.HTML("""
        <div id="spinner">
            <div class="spinner-circle"></div>
            <div class="typing-loader">
                <span class="loader-text">Preparing your personalized assistant</span>
                <span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
            </div>
        </div>
        """)

    with gr.Row(visible=False) as main_chat:
        with gr.Column(scale=0, min_width=80, elem_id="sidebar"):
            interest_tab = gr.Button("🎯 Interest", elem_id="interest-tab", elem_classes=["selected"])
            program_tab = gr.Button("🎓 Program", elem_id="program-tab")

        with gr.Column(scale=4, elem_id="main-content") as main_area:
            with gr.Column(visible=True) as interest_page:
                with gr.Row():
                    with gr.Column(scale=1, min_width=160):
                        suggest_btn = gr.Button("Suggest Majors", elem_classes="small-colored")
                        school_btn = gr.Button("Recommend Schools", elem_classes="small-colored")
                        ps_btn = gr.Button("Write Personal Statement", elem_classes="small-colored")
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(label="👩‍🎓 Sophie")
                        msg = gr.Textbox(
                            placeholder="Feel free to share your interests, goals, or concerns. I'm here to help you plan your path!",
                            label="Input",
                            lines=1
                        )
                        state = gr.State([])

                        msg.submit(fn=predict, inputs=[msg, state], outputs=[chatbot, state])
                        suggest_btn.click(fn=send_fixed_message("Can you suggest some college majors based on my interests?"),
                                          inputs=[state], outputs=[chatbot, state])
                        school_btn.click(fn=send_fixed_message("Can you recommend some schools based on my profile?"),
                                         inputs=[state], outputs=[chatbot, state])
                        ps_btn.click(fn=send_fixed_message("Can you help me write a personal statement for university application?"),
                                     inputs=[state], outputs=[chatbot, state])

            with gr.Column(visible=False) as program_page:
                gr.Markdown("## 🎓 Saved Programs")

                # 输入区域
                with gr.Column():
                    with gr.Row():
                        school_name = gr.Textbox(label="School Name", placeholder="e.g. Ohio Northern University")
                        program_name = gr.Textbox(label="Program Name", placeholder="e.g. Computer Science")
                    add_button = gr.Button("➕ Add Program Card")

                # 卡片展示容器（HTML）
                card_display = gr.HTML("")  # 用 HTML 渲染自定义网格
                school_cards = gr.State([])
                delete_trigger = gr.Text(visible=False, label=None, elem_id="delete_trigger")

                def delete_school_card(delete_index, cards):
                    cards = cards or []
                    delete_index = int(delete_index)
                    if 0 <= delete_index < len(cards):
                        cards.pop(delete_index)

                    html = "<div class='card-grid'>"
                    for i, (s, p) in enumerate(cards):
                        html += f"""
                        <div class='school-card' id='card-{i}'>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong>🏫 {s}</strong>
                                <button class='delete-btn' data-index='{i}' style='background: none; border: none; font-size: 18px; cursor: pointer;'>❌</button>
                            </div>
                            <span>📘 {p}</span>
                        </div>
                        """
                    html += "</div>"
                    return html, cards

                delete_trigger.change(
                    fn=delete_school_card,
                    inputs=[delete_trigger, school_cards],
                    outputs=[card_display, school_cards]
                )

                def add_school_card(school, program, cards):
                    cards = cards or []
                    cards.append((school, program))

                    html = "<div class='card-grid'>"
                    for i, (s, p) in enumerate(cards):
                        html += f"""
                        <div class='school-card' id='card-{i}'>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong>🏫 {s}</strong>
                                <button class='delete-btn' data-index='{i}' style='background: none; border: none; font-size: 18px; cursor: pointer;'>❌</button>
                            </div>
                            <span>📘 {p}</span>
                        </div>
                        """
                    html += "</div>"

                    return html, cards

                add_button.click(
                    fn=add_school_card,
                    inputs=[school_name, program_name, school_cards],
                    outputs=[card_display, school_cards]
                )

    def switch_to_interest():
        return (
            gr.update(visible=True),  # interest_page
            gr.update(visible=False),  # program_page
            gr.update(elem_classes=["selected"]),  # highlight Interest
            gr.update(elem_classes=[]),            # unhighlight Program
            "interest"
        )

    def switch_to_program():
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(elem_classes=["selected"]),
            gr.update(elem_classes=[]),
            "program"
        )

    interest_tab.click(fn=switch_to_interest, inputs=[],
                       outputs=[interest_page, program_page, interest_tab, program_tab, selected_tab])
    program_tab.click(fn=switch_to_program, inputs=[],
                      outputs=[interest_page, program_page, program_tab, interest_tab, selected_tab])

    def start_chat_fn(g, s, p, gen, interests_list):
        if interests_list:
            interest_str = f"You mentioned you're interested in subjects like {', '.join(interests_list)}."
        else:
            interest_str = "You haven't selected any favorite subjects yet."

        welcome = (
            f"Welcome! You're in grade {g}, from {s} in {p}, and identify as {gen}. "
            f"{interest_str} Let's begin your planning journey!"
        )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            [["assistant", welcome]],
            [{"role": "user", "content": welcome}]
        )

    def loading_to_main():
        time.sleep(1)
        return gr.update(visible=False), gr.update(visible=True)

    start_button.click(
        fn=start_chat_fn,
        inputs=[grade, school, province, gender, interests],
        outputs=[intro_page, loading_page, main_chat, chatbot, state]
    ).then(
        fn=loading_to_main,
        inputs=[],
        outputs=[loading_page, main_chat]
    )

demo.launch()

gr.HTML("""
<script>
document.addEventListener('click', function(event) {
    if (event.target && event.target.matches('.delete-btn')) {
        const index = event.target.getAttribute('data-index');
        const wrapper = document.querySelector('[id^="delete_trigger"]');
        const input = wrapper ? wrapper.querySelector('input') : null;
        if (input) {
            input.value = index;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
});
</script>
""")