# LLM Token Sampling Playground

An interactive **Streamlit app** for demonstrating how Large Language Models select their next token using:

- Logits  
- Softmax  
- Probability Mass Functions (PMFs)  
- Temperature sampling  
- Top-k sampling  
- Top-p (nucleus) sampling  
- Greedy decoding  

Students can adjust sliders, visualize distributions, and see how “sharp” vs. “flat” probability landscapes influence **hallucinations**.

---

## Live Demo

**https://llm-token-sampling-demo.streamlit.app/**

---

## Repository Contents

- `app.py` — Main Streamlit application  
- `requirements.txt` — Dependencies for deployment  

---

## 📌 Important Note: AI Literacy Disclaimer

This playground is for **educational / AI literacy** purposes only. The demos (for example, the UK-capital test always producing **London**) are **simplified illustrations** of how probability shapes LLM behavior and do **not** fully represent how real large language models operate in all cases.

- Some behaviors are intentionally **deterministic or simplified** to make concepts (sharp vs. flat probability distributions, sampling, hallucination risk) easier to see and discuss.  
- The app teaches **intuition** about model confidence and sampling; it is **not** a complete or exact simulation of production LLM internals or training data.  
- Use these examples to build understanding and critical thinking about model outputs, but do not rely on them as a technical specification or proof of how any specific model will behave in every scenario.

