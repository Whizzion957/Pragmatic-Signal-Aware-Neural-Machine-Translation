"""
demo.py — Interactive Demo (ipywidgets) (v2)
==============================================
Provides an interactive translation demo with sarcasm detection,
dual translation comparison, sentiment analysis, and pragmatic
divergence score.
"""

import torch
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output


def create_demo(sarcasm_pipe, sentiment_pipe, base_model, base_tokenizer,
                prag_model, prag_tokenizer, embed_model=None):
    """
    Create and display an interactive translation demo.
    """
    from translator import translate_single
    from evaluate import get_sentiment, compute_pragmatic_divergence
    from detector import predict_sarcasm

    # ─── UI Components ─────────────────────────────────────────────
    title = widgets.HTML(
        value="""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; margin-bottom: 15px;">
            <h2 style="color: white; margin: 0; font-family: 'Segoe UI', sans-serif;">
                🧠 Pragmatic Signal-Aware Neural Machine Translation
            </h2>
            <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">
                English → Hindi translation with sarcasm awareness
            </p>
        </div>
        """
    )

    input_box = widgets.Textarea(
        value="oh great, another monday morning, just what i needed",
        placeholder="Enter an English sentence...",
        description="",
        layout=widgets.Layout(width="100%", height="80px"),
        style={"description_width": "0px"},
    )

    input_label = widgets.HTML(
        value='<b style="font-size: 14px;">📝 Enter English text:</b>'
    )

    translate_btn = widgets.Button(
        description="🔄 Translate",
        button_style="primary",
        layout=widgets.Layout(width="200px", height="40px"),
        style={"font_weight": "bold"},
    )

    output_area = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="15px",
            border_radius="10px",
            min_height="200px",
        )
    )

    # ─── example buttons ───────────────────────────────────────────
    examples = [
        "oh great, another monday morning, just what i needed",
        "wow what a fantastic service, waited only 2 hours!",
        "sure, because standing in the rain is so much fun",
        "i love the beautiful sunset at the beach",
        "yeah right, like i totally needed more homework today",
    ]

    example_label = widgets.HTML(
        value='<b style="font-size: 13px; color: #555;">💡 Try these examples:</b>'
    )

    example_buttons = []
    for ex in examples:
        btn = widgets.Button(
            description=ex[:50] + ("..." if len(ex) > 50 else ""),
            layout=widgets.Layout(width="auto"),
            button_style="",
            tooltip=ex,
        )

        def make_handler(text):
            def handler(b):
                input_box.value = text
            return handler

        btn.on_click(make_handler(ex))
        example_buttons.append(btn)

    examples_box = widgets.VBox([
        example_label,
        widgets.HBox(example_buttons[:3], layout=widgets.Layout(gap="5px")),
        widgets.HBox(example_buttons[3:], layout=widgets.Layout(gap="5px")),
    ])

    # ─── translate handler ─────────────────────────────────────────
    def on_translate(b):
        with output_area:
            clear_output(wait=True)
            text = input_box.value.strip()
            if not text:
                display(HTML("<p style='color: red;'>⚠️ Please enter some text.</p>"))
                return

            display(HTML("<p style='color: #888;'>⏳ Processing...</p>"))
            clear_output(wait=True)

            # Step 1: Sarcasm detection
            sarc_result = predict_sarcasm(text, sarcasm_pipe)
            is_sarcastic = sarc_result["is_sarcastic"]
            control_token = "<SARCASTIC>" if is_sarcastic else "<LITERAL>"

            # Step 2: Translations
            baseline_trans = translate_single(text, base_model, base_tokenizer)
            pragmatic_trans = translate_single(
                text, prag_model, prag_tokenizer,
                control_token=control_token,
            )

            # Step 3: Sentiment analysis
            src_sentiment = get_sentiment(text, sentiment_pipe)
            base_sentiment = get_sentiment(baseline_trans, sentiment_pipe)
            prag_sentiment = get_sentiment(pragmatic_trans, sentiment_pipe)

            base_match = "✅" if src_sentiment == base_sentiment else "❌"
            prag_match = "✅" if src_sentiment == prag_sentiment else "❌"

            # Step 4: Pragmatic Divergence Score (if available)
            pds_html = ""
            if embed_model is not None:
                base_pds = compute_pragmatic_divergence([text], [baseline_trans], embed_model)
                prag_pds = compute_pragmatic_divergence([text], [pragmatic_trans], embed_model)
                pds_html = f"""
                <tr>
                    <td style="padding: 10px; font-weight: bold;">
                        Pragmatic Sim. (PDS)
                    </td>
                    <td style="padding: 10px;">
                        {base_pds['mean_similarity']:.4f}
                    </td>
                    <td style="padding: 10px; background: #f0f7ff;">
                        {prag_pds['mean_similarity']:.4f}
                    </td>
                </tr>
                """

            # Color coding
            sarc_color = "#e74c3c" if is_sarcastic else "#27ae60"
            sarc_icon = "🎭" if is_sarcastic else "📝"

            html = f"""
            <div style="font-family: 'Segoe UI', sans-serif;">
                <div style="background: {'#ffeaea' if is_sarcastic else '#eaffea'};
                            padding: 12px; border-radius: 8px; margin-bottom: 10px;
                            border-left: 4px solid {sarc_color};">
                    <b>{sarc_icon} Pragmatic Signal:</b>
                    <span style="color: {sarc_color}; font-weight: bold;">
                        {sarc_result['label']}
                    </span>
                    (confidence: {sarc_result['score']:.1%})
                </div>

                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">
                        </th>
                        <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">
                            📤 Baseline NMT
                        </th>
                        <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">
                            🧠 Pragmatic-Aware NMT
                        </th>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: bold; border-bottom: 1px solid #eee;">
                            Translation
                        </td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; font-size: 16px;">
                            {baseline_trans}
                        </td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; font-size: 16px;
                                   background: #f0f7ff;">
                            {pragmatic_trans}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: bold; border-bottom: 1px solid #eee;">
                            Sentiment
                        </td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">
                            {base_sentiment} {base_match}
                        </td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; background: #f0f7ff;">
                            {prag_sentiment} {prag_match}
                        </td>
                    </tr>
                    {pds_html}
                </table>

                <div style="margin-top: 10px; padding: 8px; background: #f8f9fa;
                            border-radius: 5px; font-size: 12px; color: #666;">
                    Source sentiment: <b>{src_sentiment}</b> |
                    Control token: <code>{control_token}</code>
                </div>
            </div>
            """
            display(HTML(html))

    translate_btn.on_click(on_translate)

    # ─── layout and display ────────────────────────────────────────
    demo_widget = widgets.VBox([
        title,
        input_label,
        input_box,
        widgets.HBox([translate_btn], layout=widgets.Layout(justify_content="center")),
        examples_box,
        widgets.HTML(value="<hr style='margin: 10px 0;'>"),
        output_area,
    ], layout=widgets.Layout(
        max_width="800px",
        padding="15px",
    ))

    display(demo_widget)
    print("✅ Demo ready! Enter text and click 'Translate'.")


if __name__ == "__main__":
    print("demo.py must be run inside a Jupyter/Colab notebook.")
    print("Import create_demo() and call it with the required pipelines.")
