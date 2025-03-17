import hashlib

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from pages.common import add_bmc_footer

st.title("Hash Embeddings Visualization")
st.markdown(
    """
This interactive demo helps you understand how hash embeddings work compared to traditional word embeddings and the hashing trick.
*It's best viewed on wide mode*

*For more information, have a look at my article: [What are Hash embeddings?](https://sieunpark77.medium.com/what-are-hash-embeddings-217759886d0)*"""
)

# Sidebar parameters
st.sidebar.header("Parameters")
num_words = st.sidebar.slider("Number of words", 5, 20, 10)
embedding_dim = st.sidebar.slider("Embedding dimension", 2, 20, 6)

# Generate random words if not provided
word_input = st.sidebar.text_area(
    "Custom words (one per line)",
    "dog\ncat\nmouse\nhouse\nelephant\nbird\ntiger\nbanana\napple\ncar",
)
words = [w.strip() for w in word_input.split("\n") if w.strip()][:num_words]

st.sidebar.divider()
num_hash_functions = st.sidebar.slider("Number of hash functions (k)", 1, 5, 2)
num_buckets = st.sidebar.slider("Number of buckets (B)", 2, 20, 8)


# Helper functions
def hash_function(word, seed, num_buckets):
    """Hash a word to an integer between 0 and num_buckets-1"""
    return int(hashlib.md5(f"{word}{seed}".encode()).hexdigest(), 16) % num_buckets


def get_standard_embedding(word, vocab_size, embedding_dim):
    """Simulate a standard embedding lookup (deterministic for demo)"""
    # Use word hash as seed for reproducibility
    np.random.seed(int(hashlib.md5(word.encode()).hexdigest(), 16) % 10000)
    return np.random.randn(embedding_dim)


def get_hashing_trick_embedding(word, num_buckets, embedding_dim):
    """Get embedding using the hashing trick"""
    bucket = hash_function(word, 0, num_buckets)
    # Use bucket as seed for reproducibility
    np.random.seed(bucket)
    return np.random.randn(embedding_dim)


def get_hash_embedding(word, num_buckets, embedding_dim, num_hash_functions):
    """Get hash embedding using multiple hash functions and importance parameters"""
    component_vectors = []
    # Generate component vectors for each hash function
    for i in range(num_hash_functions):
        bucket = hash_function(word, i, num_buckets)
        np.random.seed(bucket)
        component_vectors.append(np.random.randn(embedding_dim))

    # Generate importance parameters (also deterministic for demo)
    np.random.seed(int(hashlib.md5(word.encode()).hexdigest(), 16) % 10000)
    importance_params = np.random.rand(num_hash_functions)
    importance_params = importance_params / importance_params.sum()  # Normalize

    # Weighted sum of component vectors
    hash_embedding = np.zeros(embedding_dim)
    for i in range(num_hash_functions):
        hash_embedding += importance_params[i] * component_vectors[i]

    return hash_embedding, component_vectors, importance_params


# Display explanation===============================================================
with st.expander("Overview - explanation", expanded=True):
    st.markdown(
        """
    **1. Traditional Embeddings**
    - Every word has its own unique vector
    - Parameters: `vocab_size × embedding_dim`

    **2. Hashing Trick**
    - Maps words to buckets using a hash function
    - Multiple words can share the same vector (collisions)
    - Parameters: `num_buckets × embedding_dim`

    **3. Hash Embeddings**
    - Use k different hash functions to select k component vectors for each word
    - Learn importance parameters to weight these vectors
    - Final representation is the weighted sum of component vectors
    - Parameters: `num_buckets × embedding_dim + vocab_size × k`
    """
    )

# Show all three embedding methods===============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Traditional Embeddings")
    standard_embeddings = {}
    for word in words:
        standard_embeddings[word] = get_standard_embedding(
            word, len(words), embedding_dim
        )

    # Create DataFrame for displaying
    standard_df = pd.DataFrame(
        {
            "Word": list(standard_embeddings.keys()),
            "Vector": [str(v.round(2)) for v in standard_embeddings.values()],
        }
    )
    st.dataframe(standard_df, use_container_width=True)

    # Parameters calculation
    st.markdown(f"**Parameters: {len(words) * embedding_dim}**")

with col2:
    st.subheader("Hashing Trick")
    hashing_trick_embeddings = {}
    hashing_trick_buckets = {}
    for word in words:
        bucket = hash_function(word, 0, num_buckets)
        hashing_trick_buckets[word] = bucket
        hashing_trick_embeddings[word] = get_hashing_trick_embedding(
            word, num_buckets, embedding_dim
        )

    # Create DataFrame for displaying
    hashing_df = pd.DataFrame(
        {
            "Word": list(hashing_trick_embeddings.keys()),
            "Bucket": [
                hashing_trick_buckets[w] for w in hashing_trick_embeddings.keys()
            ],
            "Vector": [str(v.round(2)) for v in hashing_trick_embeddings.values()],
        }
    )
    st.dataframe(hashing_df, use_container_width=True)

    # Check for collisions
    bucket_counts = pd.Series(list(hashing_trick_buckets.values())).value_counts()
    collisions = bucket_counts[bucket_counts > 1]

    # Parameters calculation
    st.markdown(f"**Parameters: {num_buckets * embedding_dim}**")

if len(collisions) > 0:
    st.warning(
        f"Collisions detected for `Hashing trick`! {len(collisions)} buckets have multiple words."
    )

with col3:
    st.subheader("Hash Embeddings")
    hash_embeddings = {}
    hash_buckets = {}
    hash_importance = {}

    for word in words:
        embedding, components, importance = get_hash_embedding(
            word, num_buckets, embedding_dim, num_hash_functions
        )
        hash_embeddings[word] = embedding

        # Store which buckets were used for each hash function
        word_buckets = []
        for i in range(num_hash_functions):
            word_buckets.append(hash_function(word, i, num_buckets))
        hash_buckets[word] = word_buckets
        hash_importance[word] = importance

    # Create DataFrame for displaying
    hash_df = pd.DataFrame(
        {
            "Word": list(hash_embeddings.keys()),
            "Buckets": [str(hash_buckets[w]) for w in hash_embeddings.keys()],
            "Importance": [
                str(hash_importance[w].round(2)) for w in hash_embeddings.keys()
            ],
            "Vector": [str(v.round(2)) for v in hash_embeddings.values()],
        }
    )
    st.dataframe(hash_df, use_container_width=True)

    # Parameters calculation
    st.markdown(
        f"**Parameters: {num_buckets * embedding_dim + len(words) * num_hash_functions}**"
    )

# Visualize all embeddings===============================================================
st.header("Embedding Space Visualization")

# Generate 2D projections of all embeddings
if embedding_dim > 2:
    # Combine all vectors for PCA
    vectors = []
    labels = []
    methods = []

    for word in words:
        vectors.append(standard_embeddings[word])
        labels.append(word)
        methods.append("Standard")

        vectors.append(hashing_trick_embeddings[word])
        labels.append(word)
        methods.append("Hashing Trick")

        vectors.append(hash_embeddings[word])
        labels.append(word)
        methods.append("Hash Embedding")

    # Apply PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(np.array(vectors))

    # Create visualization dataframe
    viz_df = pd.DataFrame(
        {
            "x": vectors_2d[:, 0],
            "y": vectors_2d[:, 1],
            "Word": labels,
            "Method": methods,
        }
    )
else:
    # Direct 2D visualization
    viz_data = []
    for word in words:
        viz_data.append(
            {
                "x": standard_embeddings[word][0],
                "y": standard_embeddings[word][1],
                "Word": word,
                "Method": "Standard",
            }
        )
        viz_data.append(
            {
                "x": hashing_trick_embeddings[word][0],
                "y": hashing_trick_embeddings[word][1],
                "Word": word,
                "Method": "Hashing Trick",
            }
        )
        viz_data.append(
            {
                "x": hash_embeddings[word][0],
                "y": hash_embeddings[word][1],
                "Word": word,
                "Method": "Hash Embedding",
            }
        )
    viz_df = pd.DataFrame(viz_data)

# Create a scatter plot for each method
col1, col2, col3 = st.columns(3)

with col1:
    standard_chart = (
        alt.Chart(viz_df[viz_df["Method"] == "Standard"])
        .mark_text()
        .encode(x="x", y="y", text="Word", tooltip=["Word", "x", "y"])
        .properties(width=300, height=300, title="Standard Embeddings")
        .interactive()
    )

    st.altair_chart(standard_chart, use_container_width=True)

with col2:
    hashing_chart = (
        alt.Chart(viz_df[viz_df["Method"] == "Hashing Trick"])
        .mark_text()
        .encode(x="x", y="y", text="Word", tooltip=["Word", "x", "y"])
        .properties(width=300, height=300, title="Hashing Trick Embeddings")
        .interactive()
    )

    st.altair_chart(hashing_chart, use_container_width=True)

with col3:
    hash_emb_chart = (
        alt.Chart(viz_df[viz_df["Method"] == "Hash Embedding"])
        .mark_text()
        .encode(x="x", y="y", text="Word", tooltip=["Word", "x", "y"])
        .properties(width=300, height=300, title="Hash Embeddings")
        .interactive()
    )

    st.altair_chart(hash_emb_chart, use_container_width=True)


# Word detail viewer===============================================================
st.header("Detailed Word View")
selected_word = st.selectbox("Select a word to examine", [None] + words)

if selected_word:
    # Get embeddings for selected word
    std_emb = standard_embeddings[selected_word]
    hash_trick_emb = hashing_trick_embeddings[selected_word]
    hash_emb, components, importance = get_hash_embedding(
        selected_word, num_buckets, embedding_dim, num_hash_functions
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hash Embedding Construction")
        # Display the hash functions and buckets
        hash_detail_df = pd.DataFrame(
            {
                "Hash Function": [f"H{i+1}" for i in range(num_hash_functions)],
                "Bucket": [
                    hash_function(selected_word, i, num_buckets)
                    for i in range(num_hash_functions)
                ],
                "Importance": importance.round(3),
                "Component Vector": [
                    str(components[i].round(2)) for i in range(num_hash_functions)
                ],
            }
        )
        st.dataframe(hash_detail_df, use_container_width=True)

        # Show final vector
        st.markdown(f"**Final Hash Embedding Vector:** {hash_emb.round(3)}")

        # Formula explanation
        st.markdown(
            """
        ### Formula
        The hash embedding is calculated as:
        
        **vector = importance₁ × component₁ + importance₂ × component₂ + ... + importanceₖ × componentₖ**
        
        This weighted combination allows the model to choose which hash functions are most useful.
        """
        )

    with col2:
        st.subheader("Embedding Comparison")
        # Create a comparison of the three embeddings
        comparison_df = pd.DataFrame(
            {
                "Method": ["Standard Embedding", "Hashing Trick", "Hash Embedding"],
                "Vector": [
                    str(std_emb.round(3)),
                    str(hash_trick_emb.round(3)),
                    str(hash_emb.round(3)),
                ],
            }
        )
        st.dataframe(comparison_df, use_container_width=True)

        # If embedding dimension is 2, visualize directly, otherwise use PCA
        if embedding_dim == 2:
            # Create a direct visualization
            plot_data = pd.DataFrame(
                {
                    "x": [std_emb[0], hash_trick_emb[0], hash_emb[0]],
                    "y": [std_emb[1], hash_trick_emb[1], hash_emb[1]],
                    "Method": ["Standard", "Hashing Trick", "Hash Embedding"],
                }
            )
        else:
            # Use PCA to visualize in 2D
            all_vectors = np.vstack([std_emb, hash_trick_emb, hash_emb])
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(all_vectors)
            plot_data = pd.DataFrame(
                {
                    "x": transformed[:, 0],
                    "y": transformed[:, 1],
                    "Method": ["Standard", "Hashing Trick", "Hash Embedding"],
                }
            )

        # Create scatter plot with Altair
        chart = (
            alt.Chart(plot_data)
            .mark_circle(size=200)
            .encode(x="x", y="y", color="Method", tooltip=["Method", "x", "y"])
            .properties(
                width=400,
                height=300,
                title=f"2D Visualization of Embeddings for '{selected_word}'",
            )
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)


# Add summary at the bottom
# Add this code to the bottom of the Streamlit app before the final summary

# Add this code to the bottom of the Streamlit app before the final summary

st.markdown(
    """
---
### Key Benefits of Hash Embeddings

1. **Fewer Parameters**: Hash embeddings require significantly fewer parameters than standard embeddings.
   - Standard: vocab_size × dimension
   - Hash Embeddings: num_buckets × dimension + vocab_size × num_hash_functions (where num_buckets << vocab_size)
"""
)

# Create expandable section for realistic comparison
with st.expander(
    "Compare Parameter Efficiency with Realistic Dimensions", expanded=True
):
    # Input fields for realistic values
    col1, col2, col3 = st.columns(3)

    with col1:
        vocab_size = st.number_input(
            "Vocabulary Size",
            min_value=1000,
            max_value=10000000,
            value=500000,
            step=10000,
            help="Number of unique tokens in vocabulary",
        )

    with col2:
        real_embedding_dim = st.number_input(
            "Embedding Dimension",
            min_value=10,
            max_value=10000,
            value=300,
            step=50,
            help="Dimension of embedding vectors",
        )

    with col3:
        real_num_buckets = st.number_input(
            "Number of Buckets",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Number of hash buckets",
        )

    real_num_hash_functions = st.slider(
        "Number of Hash Functions",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of hash functions used in hash embeddings",
    )

    # Calculate parameter counts
    standard_params = vocab_size * real_embedding_dim
    hashing_trick_params = real_num_buckets * real_embedding_dim
    hash_embedding_params = (
        real_num_buckets * real_embedding_dim + vocab_size * real_num_hash_functions
    )

    # Calculate savings
    savings_vs_standard = (1 - hash_embedding_params / standard_params) * 100

    # Format large numbers with commas
    def format_number(num):
        return f"{num:,}"

    # Create a comparison table
    comparison_data = {
        "Method": ["Standard Embeddings", "Hashing Trick", "Hash Embeddings"],
        "Formula": [
            "vocab_size × embedding_dim",
            "num_buckets × embedding_dim",
            "num_buckets × embedding_dim + vocab_size × num_hash_functions",
        ],
        "Parameters": [
            format_number(standard_params),
            format_number(hashing_trick_params),
            format_number(hash_embedding_params),
        ],
        "Memory (MB)": [
            f"{standard_params * 4 / (1024*1024):.2f}",
            f"{hashing_trick_params * 4 / (1024*1024):.2f}",
            f"{hash_embedding_params * 4 / (1024*1024):.2f}",
        ],
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

    # Create a bar chart to visualize the parameter counts
    chart_data = pd.DataFrame(
        {
            "Method": ["Standard\nEmbeddings", "Hashing\nTrick", "Hash\nEmbeddings"],
            "Parameters": [
                standard_params,
                hashing_trick_params,
                hash_embedding_params,
            ],
        }
    )

    # Add parameter breakdown for hash embeddings
    st.subheader("Hash Embeddings Parameter Breakdown")
    hash_emb_component_params = real_num_buckets * real_embedding_dim
    hash_emb_importance_params = vocab_size * real_num_hash_functions

    # Create pie chart data
    pie_data = pd.DataFrame(
        {
            "Component": ["Component Vectors", "Importance Parameters"],
            "Parameters": [hash_emb_component_params, hash_emb_importance_params],
        }
    )

    pie_chart = (
        alt.Chart(pie_data)
        .mark_arc()
        .encode(
            theta="Parameters", color="Component", tooltip=["Component", "Parameters"]
        )
        .properties(
            width=400, height=400, title="Hash Embeddings Parameter Distribution"
        )
    )

    st.altair_chart(pie_chart, use_container_width=True)

    # Add insights
    st.markdown(
        f"""
    ### Key Insights
    
    - **Memory Savings**: Hash embeddings use **{savings_vs_standard:.1f}%** fewer parameters than standard embeddings with these settings
    - **Quality vs Efficiency Trade-off**: 
        - With {real_num_buckets:,} buckets (compared to {vocab_size:,} words), hash embeddings can maintain quality while reducing memory
        - The {real_num_hash_functions} hash functions allow each word to have a unique representation despite sharing component vectors
    - **Scalability**: As vocabulary grows, the parameter efficiency advantage of hash embeddings increases
    
    **Useful for**
    - **Edge Applications**: When deploying NLP models on memory-constrained devices
    - **Large Vocabularies**: When working with multiple languages or specialized domains
    - **Handling Rare Words**: Hash embeddings can represent unseen words effectively
    """
    )

st.markdown(
    """
2. **Collision Handling**: Unlike the hashing trick, hash embeddings can differentiate between words that hash to the same bucket through learned importance parameters.

3. **No Dictionary Required**: Hash embeddings can work without a pre-defined vocabulary, making them suitable for online learning scenarios.

4. **Implicit Regularization**: Hash embeddings have been shown to have a regularizing effect that can improve model performance.
"""
)

st.markdown(
    """
---
This demo was created to help understand the concepts in the paper:  
"Hash Embeddings for Efficient Word Representations" by Dan Svenstrup, Jonas Meinertz Hansen, and Ole Winther (2017)


*For more information, have a look at my article: [What are Hash embeddings?](https://sieunpark77.medium.com/what-are-hash-embeddings-217759886d0)*

"""
)

add_bmc_footer()