# Research Assistant AI Agent

An intelligent research automation tool that takes a user query, expands it, gathers diverse knowledge from trusted online sources like youtube, semantic scholar, wikipedia and summarizes the findings using Meta LLAMA 3.1.

## What It Does

- User enters a **research query**.
- The query is expanded into **five diverse sub-queries** using an LLM.
- Each sub-query is processed through:
  - **YouTube Tool** – Extracts relevant video transcripts
  - **Semantic Scholar Tool** – Retrieves academic papers and citations
  - **Wikipedia Tool** – Fetches detailed encyclopedia entries
- The collected content is aggregated and passed to an LLM for **summarization and insight generation**.

## 🧠 Powered By
- **Together AI**
- **Meta LLaMA 3.1**
- **LangChain** – Agent orchestration and tool integration
- **YouTube Search APIs**, **Semantic Scholar API**, **Wikipedia API**

## 📂 Folder Structure

```
research-assistant-agent/
├── notebook/
│   └── research.ipynb        # Exploratory notebook for agent logic
├── home.py                   # UI or CLI entry (optional)
├── ideation.py               # Query expansion module
├── research_collection3.py   # Tools for content collection
├── main.py                   # Orchestrates full pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## ▶️ How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/research-assistant-agent.git
   cd research-assistant-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API keys**
   - Add your API keys (OpenAI, Together, Semantic Scholar, YouTube) in an `.env` file or directly in the code.

4. **Run the app**
   ```bash
   streamlit run main.py
   ```

5. **Input your query** and get a detailed, multi-source research summary.

## 📄 License

MIT License – Free to use and extend.

---

Let me know if you'd like to add usage examples, architecture diagrams, or citation formats in the output.
