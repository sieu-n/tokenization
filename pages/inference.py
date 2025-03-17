import io
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import transformers

from pages.common import create_colored_text_html

# Dictionary of tokenizers with their model paths
TOKENIZERS = {
    "LLaMA 3": "baseten/Meta-Llama-3-tokenizer",
    "DeepSeek": "deepseek-ai/deepseek-coder-7b-base",
    "T5": "t5-base",
    "BERT": "bert-base-uncased",
    "GPT-2": "gpt2",
}


def get_tokenizer(tokenizer_name):
    """Get tokenizer using AutoTokenizer"""
    if tokenizer_name in TOKENIZERS:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZERS[tokenizer_name])
    else:
        return transformers.AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_text(tokenizer_name, text):
    """Tokenize text and return tokens and decoded tokens"""
    tokenizer = get_tokenizer(tokenizer_name)
    if not tokenizer:
        raise RuntimeError("Tokenizer not found")

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded_tokens = [tokenizer.decode(t) for t in tokens]
    return tokens, decoded_tokens


def generate_token_length_plot(decoded_tokens):
    """Generate a plot of token lengths using Seaborn"""
    # Create a DataFrame for easier Seaborn plotting
    token_lengths = [len(token.encode("utf-8")) for token in decoded_tokens]
    df = pd.DataFrame(
        {"Token Index": range(len(token_lengths)), "Token Length": token_lengths}
    )

    # Set the style and create the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Create a bar plot with Seaborn
    sns.barplot(x="Token Index", y="Token Length", data=df, palette="viridis")

    # Customize the plot
    plt.title("Token Length Distribution", fontsize=16)
    plt.xlabel("Token Index", fontsize=12)
    plt.ylabel("Token Length", fontsize=12)

    # Rotate x-axis labels if there are many tokens
    if len(token_lengths) > 10:
        plt.xticks(rotation=45, ha="right")

    # Add summary statistics
    plt.text(
        0.02,
        0.95,
        f"Mean Length: {np.mean(token_lengths):.2f}\n"
        + f"Max Length: {max(token_lengths)}\n"
        + f"Min Length: {min(token_lengths)}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    plt.close()
    return buf


def main():
    st.title("🔤 Tokenizer playground")
    st.markdown(
        """
    Explore how different tokenizers break down text into tokens!
    """
    )

    # Tokenizer selection
    model = st.query_params.get("model", "LLaMA 3")
    selected_tokenizer = st.selectbox(
        "Select Tokenizer",
        list(TOKENIZERS.keys()) + ["✨ Custom"],
        index=list(TOKENIZERS.keys()).index(model),
    )

    if selected_tokenizer == "✨ Custom":
        selected_tokenizer = st.text_input("Huggingface name:")
        if not selected_tokenizer:
            return

    # Text input
    input_text = st.text_area(
        "Enter text to tokenize",
        st.query_params.get(
            "text", "Explore how different tokenizers break down text into tokens!🚀"
        ),
        height=200,
    )
    if not input_text:
        return

    # Perform tokenization
    with st.spinner("Tokenizing text..."):
        tokens, decoded_tokens = tokenize_text(selected_tokenizer, input_text)

    st.markdown(f"Characters: `{len(input_text)}`\tTokens: `{len(tokens)}`")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(
        [
            "Colored Tokens",
            "Token Details",
            "Token Length Analysis",
        ]
    )

    with tab1:
        # Display color-coded tokens
        st.markdown(create_colored_text_html(decoded_tokens), unsafe_allow_html=True)

    with tab2:
        # Detailed token information
        st.markdown("### Token Details")
        token_df = [{"Token ID": tokens, "Token": decoded_tokens}]
        st.dataframe(token_df)

        # Show raw token IDs
        st.markdown("### Raw Token IDs")
        st.write(tokens)

    with tab3:
        # Token length analysis
        st.markdown("### Token Length Distribution")
        token_length_plot = generate_token_length_plot(decoded_tokens)
        st.image(token_length_plot)


main()
