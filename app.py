# app.py
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(layout="wide", page_title="LLM Token Sampling Playground")

# -------------------------
# Utilities
# -------------------------
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
    # find smallest set with cumulative >= p
    cutoff = np.searchsorted(cumsum, p) + 1
    keep = idx[:cutoff]
    mask = np.zeros_like(probs, dtype=bool)
    mask[keep] = True
    newp = np.where(mask, probs, 0.0)
    s = newp.sum()
    return newp / (s if s > 0 else 1.0)

def sample_from(probs, method="temperature"):
    # probs should already be normalized
    return np.random.choice(len(probs), p=probs)

def build_chart(vocab, probs):
    df = pd.DataFrame({"token": vocab, "probability": probs})
    df["rank"] = df["probability"].rank(method="first", ascending=False).astype(int)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("token:N", sort=None, title="Token"),
            y=alt.Y("probability:Q", title="Probability"),
            color=alt.condition(
                alt.datum.probability == df["probability"].max(),
                alt.value("#4c78a8"),
                alt.value("#8da0cb"),
            ),
            tooltip=["token", alt.Tooltip("probability:Q", format=".4f")],
        )
        .properties(height=320)
    )
    return chart

# -------------------------
# App UI
# -------------------------
st.title("LLM Token Sampling Playground")
st.write(
    "Interactive demo: set token logits (or use presets), adjust temperature / top-k / top-p, "
    "and observe how the PMF and sampled token change. Great for teaching softmax & sampling."
)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    preset = st.selectbox(
        "Preset vocabulary / logits",
        options=["Sharp (high-consistency)", "Flat (low-consistency)", "Custom"],
    )

    default_vocab_sharp = ["America", "Americans", "Africa", "Asia", "Europe", "Other1", "Other2", "Other3"]
    default_logits_sharp = [8.0, 0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0]

    default_vocab_flat = ["fairness", "bias", "ethical", "deep", "machine", "trust", "model", "learning"]
    default_logits_flat = [1.2, 1.0, 1.05, 0.8, 0.7, 0.65, 0.6, 0.5]

    if preset == "Sharp (high-consistency)":
        vocab = default_vocab_sharp.copy()
        initial_logits = default_logits_sharp.copy()
    elif preset == "Flat (low-consistency)":
        vocab = default_vocab_flat.copy()
        initial_logits = default_logits_flat.copy()
    else:
        # custom: user provides newline-separated tokens
        txt = st.text_area(
            "Enter vocabulary (one token per line)",
            value="United\nStates\nof\nAmerica\nAmericans\nAfrica\nAsia\nEurope",
            height=200,
        )
        vocab = [t.strip() for t in txt.splitlines() if t.strip()]
        # create default logits (small random)
        initial_logits = list(np.linspace(1.0, 0.1, len(vocab)).tolist()) if len(vocab) > 0 else []

    st.markdown("---")
    st.subheader("Sampling settings")
    sampling_method = st.radio("Sampling method", ["Greedy (argmax)", "Temperature", "Top-k", "Top-p"], index=1)
    temperature = st.slider("Temperature (for Temperature & Top-k/p pipelines)", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
    top_k = st.slider("Top-k (k tokens)", min_value=1, max_value=max(1, len(vocab)), value=min(5, len(vocab)))
    top_p = st.slider("Top-p (nucleus)", min_value=0.01, max_value=1.0, value=0.9, step=0.01)

    st.markdown("---")
    st.subheader("Generation controls")
    n_steps = st.number_input("Generate steps (auto)", min_value=1, max_value=50, value=1, step=1)
    random_seed = st.number_input("Random seed (optional, 0 = random)", min_value=0, value=0, step=1)
    if random_seed != 0:
        np.random.seed(int(random_seed))

    st.markdown("---")
    st.write("Tip: use the sliders below to tweak logits for each token.")

# Main: logit sliders
st.header("Token logits")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("Adjust logits for each token (higher = more likely before softmax).")
    logits = []
    max_tokens_for_sliders = 40
    if len(vocab) > max_tokens_for_sliders:
        st.warning(f"Large vocab ({len(vocab)}). Only first {max_tokens_for_sliders} tokens get sliders.")
    for i, token in enumerate(vocab[:max_tokens_for_sliders]):
        default = initial_logits[i] if i < len(initial_logits) else 0.0
        val = st.slider(f"{token}", min_value=-10.0, max_value=12.0, value=float(default), step=0.1, key=f"logit_{i}")
        logits.append(val)
    # For tokens beyond slider limit, fill zeros
    if len(vocab) > max_tokens_for_sliders:
        logits += [0.0] * (len(vocab) - max_tokens_for_sliders)

with col2:
    st.markdown("Quick actions")
    if st.button("Randomize logits"):
        logits = list(np.random.normal(loc=0.0, scale=1.0, size=len(vocab)))
        # Rerun - hack: write to session state for persistence
        for i in range(min(len(vocab), max_tokens_for_sliders)):
            st.session_state[f"logit_{i}"] = float(logits[i])
        st.experimental_rerun()
    if st.button("Reset logits to preset"):
        for i in range(min(len(vocab), max_tokens_for_sliders)):
            default = initial_logits[i] if i < len(initial_logits) else 0.0
            st.session_state[f"logit_{i}"] = float(default)
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Current generation")
    if "generated" not in st.session_state:
        st.session_state.generated = []
    if "last_probs" not in st.session_state:
        st.session_state.last_probs = None

    generate = st.button("Generate Next Token")
    clear = st.button("Clear Generated Text")

    if clear:
        st.session_state.generated = []
        st.session_state.last_probs = None

# If not enough tokens, show a message
if len(vocab) == 0:
    st.error("No tokens in vocabulary. Enter tokens in the sidebar (Custom) or choose a preset.")
    st.stop()

# Compute probabilities from logits & the chosen sampling pipeline
probs = softmax(logits, temp=temperature)

# Apply sampling pipeline
probs_for_sampling = probs.copy()
if sampling_method == "Greedy (argmax)":
    # greedy: probability mass concentrated on argmax for demonstration (we still sample deterministically)
    argmax = np.argmax(probs_for_sampling)
    probs_for_sampling = np.zeros_like(probs_for_sampling)
    probs_for_sampling[argmax] = 1.0
elif sampling_method == "Top-k":
    probs_for_sampling = apply_top_k(probs_for_sampling, top_k)
elif sampling_method == "Top-p":
    probs_for_sampling = apply_top_p(probs_for_sampling, top_p)
# Temperature case already included by passing temp earlier; Top-k/top-p use that tempered probs

# Show chart and top tokens
st.subheader("Probability Mass Function (PMF)")
chart = build_chart(vocab, probs)
st.altair_chart(chart, use_container_width=True)

# Show top tokens in a small table
top_n = min(10, len(vocab))
top_idx = np.argsort(probs)[::-1][:top_n]
top_df = pd.DataFrame({
    "rank": np.arange(1, top_n + 1),
    "token": np.array(vocab)[top_idx],
    "probability": probs[top_idx]
})
st.table(top_df.style.format({"probability": "{:.4f}"}))

# Sampling / generation logic
if generate:
    chosen_index = None
    if sampling_method == "Greedy (argmax)":
        chosen_index = int(np.argmax(probs))
    else:
        # For methods that modify the candidate set, apply top-k/p then sample
        candidate_probs = probs.copy()
        if sampling_method == "Top-k":
            candidate_probs = apply_top_k(candidate_probs, top_k)
        elif sampling_method == "Top-p":
            candidate_probs = apply_top_p(candidate_probs, top_p)
        # ensure normalized
        s = candidate_probs.sum()
        if s <= 0:
            candidate_probs = probs.copy()
            s = candidate_probs.sum()
        normalized = candidate_probs / s
        chosen_index = int(np.random.choice(len(vocab), p=normalized))

    chosen_token = vocab[chosen_index]
    st.session_state.generated.append(chosen_token)
    st.session_state.last_probs = probs
    st.experimental_rerun()

# Auto-generate N steps (bulk)
if st.button(f"Auto-generate {n_steps} steps"):
    for _ in range(n_steps):
        # Recompute probs each step (in a real LLM the logits would change based on new context; here we keep static for clarity)
        probs_step = softmax(logits, temp=temperature)
        if sampling_method == "Greedy (argmax)":
            idx = int(np.argmax(probs_step))
        else:
            cand = probs_step.copy()
            if sampling_method == "Top-k":
                cand = apply_top_k(cand, top_k)
            elif sampling_method == "Top-p":
                cand = apply_top_p(cand, top_p)
            s = cand.sum()
            if s <= 0:
                cand = probs_step.copy()
                s = cand.sum()
            idx = int(np.random.choice(len(vocab), p=cand / s))
        st.session_state.generated.append(vocab[idx])
    st.experimental_rerun()

# Show generated text
st.subheader("Generated sequence")
st.text(" ".join(st.session_state.generated) if st.session_state.generated else "(empty)")

# Show last probs raw if present
if st.session_state.last_probs is not None:
    st.subheader("Last computed probabilities (raw)")
    df_raw = pd.DataFrame({"token": vocab, "probability": st.session_state.last_probs})
    st.dataframe(df_raw.style.format({"probability": "{:.6f}"}), height=240)
