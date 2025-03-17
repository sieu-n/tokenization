import os
import sys

import streamlit as st

# ".../tokenization"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    pg = st.navigation(
        [
            st.Page("inference.py", title="🔍 Inference"),
            st.Page("train_bpe.py", title="📈 Train BPE Tokenizer"),
            st.Page("hash_embedding.py", title="🧠 Hash Embeddings"),
        ]
    )
    st.sidebar.title("Navigation")
    pg.run()
