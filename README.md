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

**https://YOUR-APP-URL.streamlit.app**

---

## Repository Contents

- `app.py` — Main Streamlit application  
- `requirements.txt` — Dependencies for deployment  

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
