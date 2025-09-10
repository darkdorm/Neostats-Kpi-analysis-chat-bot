# NeoStats — AI KPI Analysis Chatbot

🚀 **NeoStats** is an AI-powered KPI analysis chatbot built with **Streamlit**, **OpenAI/Groq LLMs**, **Plotly**, and **PostgreSQL**. It helps you analyze business data, detect trends and anomalies, forecast future KPIs, and interact with your data via a chat interface.

---

## Features

- Upload CSV/Excel files or use the sample dataset
- Interactive KPI dashboard: Revenue, Profit, Customers, Churn Rate
- Trend visualization using Plotly
- Detect anomalies in KPIs (e.g., churn spikes)
- Forecast future metrics with Prophet (up to 5 years)
- Generate AI-powered business insights
- Chat with your data in **English, Tamil, Telugu, or Kannada**
- Export insights as PDF
- Save KPI data to PostgreSQL
- Build RAG index for advanced data querying

---

## Demo

> **Run locally:**  
```bash
streamlit run app.py
Setup Instructions
1. Clone the repository
git clone https://github.com/your-username/Neostats-Kpi-analysis-chat-bot.git
cd Neostats-Kpi-analysis-chat-bot

2. Create a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Setup .env file

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_db_name
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4
GROQ_MODEL=groq-model


Important: Do not commit .env to GitHub. Add it to .gitignore for security.

5. Run the app
streamlit run app.py

Usage

Upload your CSV/Excel file or use the sample dataset.

View KPIs and interactive dashboards.

Explore trends, anomalies, and forecasts.

Ask questions in the chat (select language in sidebar).

Export insights as PDF or save data to PostgreSQL.

Folder Structure
├── app.py                 # Main Streamlit app
├── config/
│   └── config.py          # Configuration loader
├── models/
│   └── llm.py             # LLM interface
├── utils/
│   ├── rag_utils.py       # RAG index & query
│   ├── websearch.py       # Web search helper
│   └── pdf_export.py      # PDF export helper
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignored files (including .env)
└── README.md              # This file

Notes

PostgreSQL is optional; the app can run without it.

Chatbot supports multiple languages: English, Tamil, Telugu, Kannada.

For deployment (Streamlit Cloud, Heroku), use environment variables for API keys instead of uploading .env.

License

This project is for educational/demo purposes.


---

If you want, I can also **add a short “Project Description” paragraph** at the top so it’s perfect for submitting as part of your PPT or GitHub repo.  

Do you want me to do that?
