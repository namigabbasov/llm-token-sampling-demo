import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(layout="wide", page_title="LLM Token Sampling Playground")



### Utility functions


def softmax(logits, temp=1.0):
    z = np.array(logits) / float(temp)
    z = z - np.max(z)
    exps = np.exp(z)
    return exps / exps.sum()

def apply_top_k(probs, k):
    if k is None or k <= 0 or k >= len(probs):
        return probs
    idx = np.argsort(probs)[::-1]
    keep = idx[:k]
    mask = np.zeros_like(probs, dtype=bool)
    mask[keep] = True
    newp = np.where(mask, probs, 0.0)
    s = newp.sum()
    return newp / (s if s > 0 else 1.0)

def apply_top_p(probs, p):
    if p is None or p >= 1.0:
        return probs
    idx = np.argsort(probs)[::-1]
    sorted_probs = probs[idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    keep = idx[:cutoff]
    mask = np.zeros_like(probs, dtype=bool)
    mask[keep] = True
    newp = np.where(mask, probs, 0.0)
    s = newp.sum()
    return newp / (s if s > 0 else 1.0)

def build_chart(vocab, probs):
    df = pd.DataFrame({"token": vocab, "probability": probs})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("token:N", sort=None, title="Token"),
            y=alt.Y("probability:Q", title="Probability"),
            tooltip=["token", alt.Tooltip("probability:Q", format=".4f")]
        )
        .properties(height=300)
    )
    return chart



### MAIN UI


st.title("🔍 LLM Token Sampling Playground")
st.write("""
This interactive tool demonstrates how Large Language Models select **one token at a time**
based on probability distributions.  
""")



### SHARP vs FLAT PRESETS


st.header("Token Sampling Controls (General Playground)")

with st.sidebar:
    st.header("Settings")

    preset = st.selectbox(
        "Preset vocabulary / logits",
        ["Sharp (high-consistency)", "Flat (low-consistency)", "Custom"],
    )

    if preset == "Sharp (high-consistency)":
        vocab = ["America","Americans","Africa","Asia","Europe","Other1","Other2"]
        logits = [8.0, 0.5, -1.0, -1.5, -2.0, -3.0, -4.0]

    elif preset == "Flat (low-consistency)":
        vocab = ["fairness","bias","ethical","deep","machine","trust","model","learning"]
        logits = [1.2,1.1,1.05,1.0,0.95,1.0,0.9,0.85]

    else:
        txt = st.text_area("Enter vocabulary (one per line):",
            "token1\ntoken2\ntoken3")
        vocab = [t.strip() for t in txt.splitlines() if t.strip()]
        logits = [1.0] * len(vocab)

    st.subheader("Sampling Method")
    sampling_method = st.radio("Method", ["Greedy", "Temperature", "Top-k", "Top-p"])
    temperature = st.slider("Temperature", 0.01, 2.0, 1.0, 0.01)
    top_k = st.slider("Top-k", 1, max(1, len(vocab)), 5)
    top_p = st.slider("Top-p", 0.01, 1.0, 0.9, 0.01)



### General PMF visualization


st.subheader("Probability Distribution (General Playground)")
probs = softmax(logits, temp=temperature)
st.altair_chart(build_chart(vocab, probs), use_container_width=True)

st.write("Top tokens:")
top_idx = np.argsort(probs)[::-1][:10]
for i in top_idx:
    st.write(f"- {vocab[i]}: {probs[i]:.4f}")



### DEMONSTRATION: Sharp fact vs Flat citation


st.markdown("---")
st.header("Demonstration: Why Facts Are Stable but Citations Hallucinate")


### UK CAPITAL TEST (SHARP)


st.write("""
## Test 1 — High-Consistency Fact  
**Question:** *What is the capital of the United Kingdom?*  
We simulate a **sharp** probability distribution.
""")

if st.button("Run UK Capital Test"):
    vocab_demo = ["London", "Manchester", "Birmingham", "Cardiff", "Edinburgh", "Other1"]
    logits_demo = [10, 1, 0.5, 0.2, 0.1, -2]

    probs_demo = softmax(logits_demo, temp=1.0)
    chosen = vocab_demo[np.argmax(probs_demo)]

    st.subheader("PMF — Sharp Distribution (Capital of UK)")
    st.altair_chart(build_chart(vocab_demo, probs_demo), use_container_width=True)

    st.success(f"**Generated token:** {chosen}")
    st.info("Because the PMF is extremely sharp, the LLM chooses **London every time**, under all sampling settings.")



### CITATION TEST


st.write("""
## 🟠 Test 2 — Low-Consistency Citation Task  
**Question:** *Give me an APA citation for Smith (2019).*  
We simulate a **flat** probability distribution.
""")

if st.button("Run Citation Test"):
    vocab_cite = ["fairness", "bias", "ethical", "trust", "modeling", "AI", "learning"]
    logits_cite = [1.02, 1.01, 1.03, 0.99, 1.00, 1.02, 1.01]  # very flat

    probs_cite = softmax(logits_cite, temp=1.0)
    chosen = np.random.choice(vocab_cite, p=probs_cite)

    st.subheader("PMF — Flat Distribution (Citation Task)")
    st.altair_chart(build_chart(vocab_cite, probs_cite), use_container_width=True)

    st.warning(f"**Generated token:** {chosen}")
    st.info("""
Because the distribution is **flat**, the model is unsure.  
Each click produces a **different token**, just like citation hallucinations.
""")



### END


st.markdown("---")
st.write("Created to demonstrate LLM token sampling, sharp vs flat distributions, and hallucination mechanisms.")

