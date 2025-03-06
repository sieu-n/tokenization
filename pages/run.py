import streamlit as st


def page_2():
    st.title("Page 2")


if __name__ == "__main__":
    pg = st.navigation(
        [
            st.Page(
                "🔍_Inference.py",
            ),
            st.Page("📈_Train_BPE.py"),
        ]
    )
    st.sidebar.title("Navigation")
    pg.run()
