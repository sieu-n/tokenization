import time
from collections import defaultdict

import pandas as pd
import streamlit as st

from libs.basic import BasicTokenizer
from libs.regex import RegexTokenizer
from pages.common import create_colored_text_html

SAMPLE_TEXT_TRAIN_BPE = "Byte pair encoding (also known as BPE, or digram coding) is an algorithm, first described in 1994 by Philip Gage, for encoding strings of text into smaller strings by creating and using a translation table. A slightly-modified version of the algorithm is used in large language model tokenizers. The original version of the algorithm focused on compression. It replaces the highest-frequency pair of bytes with a new byte that was not contained in the initial dataset. A lookup table of the replacements is required to rebuild the initial dataset. The modified version builds 'tokens' (units of recognition) that match varying amounts of source text, from single characters (including single digits or single punctuation marks) to whole words (even long compound words)."
SAMPLE_TEXT_TRAIN_CODE = """
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        merge_sort(left)
        merge_sort(right)
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

arr = [38, 27, 43, 3, 9, 82, 10]
print("Merge Sort:", merge_sort(arr.copy()))
print("Quick Sort:", quick_sort(arr.copy()))
"""
INFERENCE_TEXT = """The original BPE algorithm operates by iteratively replacing the most common contiguous sequences of characters in a target text with unused 'placeholder' bytes. The iteration ends when no sequences can be found, leaving the target text effectively compressed. Decompression can be performed by reversing this process, querying known placeholder terms against their corresponding denoted sequence, using a lookup table. In the original paper, this lookup table is encoded and stored alongside the compressed text. """
DEFAULT_INFERENCE_TEXT = (
    "Explore how different tokenizers break down text into tokens!🚀"
)
pseudocode = """# actual code
    def train(self, text, vocab_size, verbose=False):
        ...

        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()"""


def format_word_freq(word_freq):
    formatted = {}
    for seq, freq in word_freq.items():
        formatted["".join(seq).replace("</w>", "⏹")] = freq
    return formatted


def format_pair_counts(pair_counts):
    formatted = {}
    for pair, count in pair_counts.items():
        formatted[f"{pair[0]}{pair[1]}"] = count
    return dict(sorted(formatted.items(), key=lambda x: x[1], reverse=True))


def init_state():
    tokenizer = BasicTokenizer()
    text = "aaabdaaabac"
    st.session_state.update({"trained": False, "train_logs": [], "tokenizer": None})


def main():
    if "tokenizer" not in st.session_state:
        init_state()

    st.title("BPE playground")

    st.markdown(
        """
    This page demonstrates 1) how training of a Byte Pair Encoding (BPE) works. 2) how BPE tokenizes a new word. Both in a step-by-step example.
    BPE is used in many language models like GPT and BERT to create tokens from text.
    """
    )

    # train ==========================
    example_texts = {
        "Simple example": "aaabdaaabac",
        "Short passage": SAMPLE_TEXT_TRAIN_BPE,
        "Python code": SAMPLE_TEXT_TRAIN_CODE,
        "Custom": "",
    }

    text_option = st.selectbox("*Select example text:", list(example_texts.keys()))

    if text_option == "Custom":
        text = st.text_area("Enter your training text:", height=150)
    else:
        text = example_texts[text_option]
        st.text_area("Training text:", text, height=150, disabled=True)

    num_merges = st.slider("*Select # merges:", min_value=1, max_value=100, value=10)
    vocab_size = 256 + num_merges
    st.markdown(
        f"`{num_merges}` merges + `256` default items will result in -> `{vocab_size}` vocabulary."
    )

    st.divider()
    model_type = st.radio("Select type", ["BasicTokenizer", "RegexTokenizer"], index=0)
    print(model_type)
    st.info(
        "`RegexTokenizer` splits on whitespace and punctuation as well as special characters. It is the real tokenizerrs used to reproduce gpt tokenizers, while the implementation of `BasicTokenizer` is straightforward for learning purposes."
    )

    def _train_tokenizer():
        print("Training BPE...")
        if not text:
            st.error("Please enter some text to train the tokenizer.")
            return
        with st.spinner("Training BPE..."):
            if model_type == "BasicTokenizer":
                st.session_state["tokenizer"] = BasicTokenizer()
            elif model_type == "RegexTokenizer":
                st.session_state["tokenizer"] = RegexTokenizer()
            else:
                st.error("Invalid model type.")
                return
            train_logs = st.session_state["tokenizer"].train(
                text, vocab_size, verbose=True
            )
            st.session_state["train_logs"] = train_logs
            st.session_state["trained"] = True

    st.button("Train BPE!", on_click=_train_tokenizer)

    if st.session_state.get("trained", False) == False:
        return
    st.divider()

    # step-by-step viewer ====================
    st.subheader("Training Process")

    # warning
    if len(st.session_state["train_logs"]) != num_merges:
        st.warning(
            f"Number of merges was too big. Only {len(st.session_state['train_logs'])} merges were performed."
        )
        num_merges = len(st.session_state["train_logs"])

    tokenizer = st.session_state["tokenizer"]
    tokens = tokenizer.encode(text)
    tokenized_text = tokenizer.decode_individual(tokens)

    if st.checkbox("Show step-by-step information", value=True):
        st.divider()
        step = st.slider("*Training step:", 1, num_merges, 1) - 1
        current_step = st.session_state["train_logs"][step]
        st.markdown(
            f"**1. Initial tokenization (`ids`):** `{len(text)}` characters, `{len(current_step['init-ids'])}` tokens\n\n{'/'.join(tokenizer.decode_individual(current_step['init-ids']))}"
        )

        st.markdown("**2. Pair counts:**")
        with st.expander("`get_stats`"):
            st.code(
                """def get_stats(ids, counts=None):
    '''
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    '''
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts
    
stats = get_stats(ids)"""
            )
        stats_decoded = {
            f"{tokenizer.decode([pair[0]])}__{tokenizer.decode([pair[1]])}": count
            for pair, count in current_step["stats"].items()
        }
        if st.checkbox("Show top-10", value=True):
            # show top 10 sorted
            st.write(
                {
                    tk: stats_decoded[tk]
                    for tk in sorted(
                        stats_decoded, key=stats_decoded.get, reverse=True
                    )[:10]
                }
            )
        else:
            st.write(stats_decoded)

        st.markdown(
            f"**3. Most frequent pair:** `{tokenizer.decode([current_step['pair'][0]])} (#{current_step['pair'][0]}) + {tokenizer.decode([current_step['pair'][1]])} (#{current_step['pair'][1]}) -> {tokenizer.decode([tokenizer.merges[current_step['pair']]])}`"
        )
        st.code("pair = max(stats, key=stats.get)")

        vocab_list = [[tk.decode(errors="replace")] for tk in current_step["vocab"]][
            256:
        ]
        merges_decoded = {
            (tokenizer.decode([t1]), tokenizer.decode([t2])): tokenizer.decode([merged])
            for (t1, t2), merged in current_step["merges"].items()
        }
        if step == num_merges - 1:
            # edge case for last step
            decode_with_next = tokenizer.decode_individual(tokenizer.encode(text))
        else:
            decode_with_next = tokenizer.decode_individual(
                st.session_state["train_logs"][step + 1]["init-ids"]
            )
        st.markdown(
            f"""**4. Update data.**
- merges: `{merges_decoded}`
- vocab: `{vocab_list}`
- tokenization for next iter (`ids`): `{'/'.join(decode_with_next)}`"""
        )
        with st.expander("`merge`"):
            st.code(
                '''def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

idx = 256 + i
merges[pair] = idx
vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
ids = merge(ids, pair, idx)'''
            )

        st.divider()
        st.code(pseudocode)
        st.divider()

    # overview ===========================

    if st.checkbox("Show overview", value=True):
        st.subheader("Results")

        st.markdown(
            f"Starting with `{len(text)}` characters, tokenized into `{len(tokens)}` tokens."
        )

        # Create tabs for the different visualizations
        tab1, tab2 = st.tabs(["Merge Rules", "Vocabulary"])

        with tab1:
            merge_df = pd.DataFrame(
                {
                    "idx": [merged for _, merged in tokenizer.merges.items()],
                    "From": [
                        f"{tokenizer.decode([pair[0]])}, {tokenizer.decode([pair[1]])}"
                        for pair, _ in tokenizer.merges.items()
                    ],
                    "To": [
                        tokenizer.decode([merged])
                        for _, merged in tokenizer.merges.items()
                    ],
                }
            )
            st.dataframe(merge_df, use_container_width=True, hide_index=True)
        with tab2:
            vocab_df = pd.DataFrame(
                {
                    "Token": tokenizer.vocab.keys(),
                    "str": [
                        tk.decode(errors="replace") for tk in tokenizer.vocab.values()
                    ],
                    "Length": [len(token) for token in tokenizer.vocab.values()],
                }
            )
            st.dataframe(vocab_df, use_container_width=True, hide_index=True)

    # inference ===========================
    st.subheader("Run inference")
    default_input_text = {
        "Simple example": "aabcbaaab",
        "Short passage": INFERENCE_TEXT,
    }
    input_text = st.text_area(
        "Enter a word to tokenize:",
        default_input_text.get(text_option, DEFAULT_INFERENCE_TEXT),
        height=300,
    )
    if not input_text:
        return
    encode_logs = []
    tokens = tokenizer.encode(input_text, encode_logs=encode_logs)
    decoded_tokens = tokenizer.decode_individual(tokens)

    st.markdown(f"Characters: `{len(input_text)}`\tTokens: `{len(tokens)}`")
    st.markdown(create_colored_text_html(decoded_tokens), unsafe_allow_html=True)

    if st.checkbox("Show steps", value=True, key="show_steps"):
        st.divider()
        step = st.slider("*Inference step:", 1, len(encode_logs), 1) - 1
        current_step = encode_logs[step]

        st.markdown(
            f"**1. Initial tokenization (`ids`):** `{len(input_text)}` characters, `{len(current_step['init-ids'])}` tokens\n\n{'/'.join(tokenizer.decode_individual(current_step['init-ids']))}"
        )

        st.markdown("2. pair with **lowest merge index** (most trivial pair): ")
        st.code(
            "stats = get_stats(ids)\npair = min(stats, key=lambda p: self.merges.get(p, float('inf')))"
        )
        stats_decoded = {
            f"{tokenizer.decode([pair[0]])}__{tokenizer.decode([pair[1]])}": tokenizer.merges.get(
                pair, float("inf")
            )
            for pair, count in current_step["stats"].items()
        }
        if st.checkbox("Show top-10", value=True, key="show_top10-2"):
            # show top 10 sorted
            st.write(
                {
                    tk: stats_decoded[tk]
                    for tk in sorted(stats_decoded, key=stats_decoded.get)[:10]
                }
            )
        else:
            st.write(stats_decoded)

        st.markdown(
            f"**3. Merged token:** `{tokenizer.decode([current_step['pair'][0]])} + {tokenizer.decode([current_step['pair'][1]])} -> {tokenizer.decode([tokenizer.merges[current_step['pair']]])}`"
        )
        st.divider()
        st.code(
            """# actual code
while len(ids) >= 2:
    stats = get_stats(ids)
    pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
    idx = self.merges[pair]
    ids = merge(ids, pair, idx)
    
return ids"""
        )
        st.divider()


main()
