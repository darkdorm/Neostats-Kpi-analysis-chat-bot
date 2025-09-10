import streamlit as st
from config.config import CONFIG
from models.llm import get_llm
from utils.rag_utils import read_table, build_index_for_dataframe, rag_query, simple_kpi_extraction
from utils.websearch import search_web
from utils.pdf_export import export_pdf
import traceback
import pandas as pd
import numpy as np
from openai import OpenAI
from groq import Groq
import plotly.graph_objects as go
from sqlalchemy import create_engine
from prophet import Prophet

# ----------------- Initialize clients -----------------
openai_client = OpenAI(api_key=CONFIG.get("OPENAI_API_KEY"))
groq_client = Groq(api_key=CONFIG.get("GROQ_API_KEY"))

# ----------------- Helper functions -----------------
def get_time_column(df):
    for col in ["Quarter", "Month", "Year"]:
        if col in df.columns:
            return col
    return None

def ask_llm(system_prompt, user_message, max_tokens=500, temperature=0.2):
    provider = CONFIG.get("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        response = openai_client.chat.completions.create(
            model=CONFIG.get("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    elif provider == "groq":
        response = groq_client.chat.completions.create(
            model=CONFIG.get("GROQ_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    else:
        return "❌ Unsupported LLM provider. Please set LLM_PROVIDER to 'openai' or 'groq'."

def save_to_postgres(df, table_name="kpi_data"):
    try:
        user = CONFIG.get("POSTGRES_USER")
        password = CONFIG.get("POSTGRES_PASSWORD")
        host = CONFIG.get("POSTGRES_HOST")
        port = CONFIG.get("POSTGRES_PORT")
        db = CONFIG.get("POSTGRES_DB")

        if None in [user, password, host, port, db]:
            st.error("PostgreSQL credentials are missing in CONFIG/.env")
            return

        port = int(port)
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        st.success(f"Data saved to PostgreSQL table `{table_name}`.")
    except Exception as e:
        st.error(f"Failed to save to PostgreSQL: {e}")

def prepare_prophet_df(df, col, time_col):
    df_prop = df[[time_col, col]].dropna().rename(columns={time_col: 'ds', col: 'y'})
    if time_col == "Quarter":
        def q_to_date(q):
            try:
                parts = str(q).replace(" ", "-").split("-")
                qnum, year = int(parts[0][1:]), int(parts[1])
                month = (qnum - 1) * 3 + 1
                return pd.Timestamp(year=year, month=month, day=1)
            except: return pd.NaT
        df_prop['ds'] = df_prop['ds'].apply(q_to_date)
    elif time_col == "Month":
        df_prop['ds'] = pd.to_datetime(df_prop['ds'].str.replace(" ", "-"), format="%b-%Y", errors='coerce')
    elif time_col == "Year":
        df_prop['ds'] = pd.to_datetime(df_prop['ds'].astype(str) + "-01-01", errors='coerce')
    return df_prop.dropna()

def forecast_prophet(df_prop):
    if df_prop.shape[0] < 2:
        return None
    m = Prophet()
    m.fit(df_prop)
    future = m.make_future_dataframe(periods=60, freq="M")
    forecast = m.predict(future)
    return forecast

def plot_forecast(df_prop, forecast, col_name):
    chart_type = st.session_state.get('forecast_chart_type', 'Line Chart')
    fig = go.Figure()

    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(x=df_prop['ds'], y=df_prop['y'], mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode="lines", name="Forecast"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode="lines",
                                 line=dict(dash='dot'), name="Upper"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode="lines",
                                 line=dict(dash='dot'), name="Lower"))
    elif chart_type == "Bar Chart":
        fig.add_trace(go.Bar(x=df_prop['ds'], y=df_prop['y'], name="Actual"))
        fig.add_trace(go.Bar(x=forecast['ds'], y=forecast['yhat'], name="Forecast"))
    elif chart_type == "Area Chart":
        fig.add_trace(go.Scatter(x=df_prop['ds'], y=df_prop['y'], mode="lines", fill='tozeroy', name="Actual"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode="lines", fill='tozeroy', name="Forecast"))
    elif chart_type == "Scatter Chart":
        fig.add_trace(go.Scatter(x=df_prop['ds'], y=df_prop['y'], mode="markers", name="Actual"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode="markers", name="Forecast"))

    fig.update_layout(
        title=f"{col_name} Forecast ({chart_type})",
        xaxis_title="Date",
        yaxis_title=col_name,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="NeoStats — AI KPI Assistant", layout="wide")

# ----------------- Project Heading -----------------
st.markdown("""
<style>
.hero-header {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(90deg, #2563eb, #9333ea);
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.hero-header h1 { margin:0; font-size:2.5rem; }
.hero-header p { margin:0; font-size:1.2rem; opacity:0.85; }
</style>
<div class="hero-header">
    <h1>🚀 NeoStats — AI KPI Assistant</h1>
    <p>Analyze trends, forecast growth, and gain actionable insights</p>
</div>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("NeoStats — KPI Assistant")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])
    use_sample = st.checkbox("Use sample dataset instead")
    mode = st.radio("Response mode", ["Concise", "Detailed"])

    st.markdown("📊 Graph Display Options")
    trend_chart_type = st.selectbox("Select Trend Chart Type", ["Line Chart", "Bar Chart", "Area Chart", "Scatter Chart"])
    forecast_chart_type = st.selectbox("Select Forecast Chart Type", ["Line Chart", "Bar Chart", "Area Chart", "Scatter Chart"])
    st.session_state['trend_chart_type'] = trend_chart_type
    st.session_state['forecast_chart_type'] = forecast_chart_type

    st.divider()
    st.markdown("⚙️ Model & Keys")
    st.write(f"LLM Provider: {CONFIG.get('LLM_PROVIDER')}")
    st.write("Tip: set API keys in your .env file")

# ----------------- Session Initialization -----------------
if "index_name" not in st.session_state: st.session_state.index_name = None
if "df" not in st.session_state: st.session_state.df = None
if "messages" not in st.session_state: st.session_state.messages = []
if "insights" not in st.session_state: st.session_state.insights = None

# ----------------- Sample Dataset -----------------
if use_sample and uploaded is None:
    st.info("Using sample dataset")
    df = pd.DataFrame({
        "Quarter": ["Q1-2023","Q2-2023","Q3-2023","Q4-2023","Q1-2024","Q2-2024","Q3-2024","Q4-2024"],
        "Revenue": [120000,135000,128000,140000,150000,160000,155000,170000],
        "Profit": [30000,35000,32000,37000,40000,42000,41000,45000],
        "Customers": [200,220,215,230,240,260,255,270],
        "ChurnRate": [0.05,0.045,0.048,0.042,0.04,0.038,0.041,0.037]
    })
    st.session_state.df = df
    uploaded = True

# ----------------- File Upload -----------------
if uploaded is not None and uploaded is not True:
    try:
        df = read_table(uploaded, uploaded.name)
        st.session_state.df = df
        st.success(f"Loaded {uploaded.name} with {df.shape[0]} rows.")
    except Exception as e:
        st.error("Failed to read file: " + str(e))
        st.error(traceback.format_exc())

# ----------------- Main Display -----------------
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    st.dataframe(df.head(10))
    time_col = get_time_column(df)
    kpis = simple_kpi_extraction(df)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    def format_growth(val):
        if val is None: return "N/A"
        color = "#0F3D23" if val >= 0 else "#7F1D1D"
        symbol = "▲" if val >= 0 else "▼"
        return f'<span style="color:{color}; font-weight:bold;">{symbol} {abs(val*100):.2f}%</span>'
    def kpi_card(title, value, delta, bg_color="#f0f4f8", icon_color="#000000"):
        st.markdown(f"""
        <div style="
            background:{bg_color};
            border-radius:16px;
            padding:1.5rem;
            text-align:center;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            margin-bottom:10px;">
            <h4 style="margin:0; color:{icon_color}; font-size:1.1rem;">{title}</h4>
            <h2 style="margin:5px 0; color:#111827; font-size:2rem;">{value}</h2>
            <p style="margin:0; font-size:1rem;">{delta}</p>
        </div>""", unsafe_allow_html=True)

    with col1: kpi_card("💰 Revenue", f"${kpis.get('Revenue',{}).get('last',0):,.0f}", format_growth(kpis.get('Revenue',{}).get('growth_fraction')), "#4A94E2")
    with col2: kpi_card("📈 Profit", f"${kpis.get('Profit',{}).get('last',0):,.0f}", format_growth(kpis.get('Profit',{}).get('growth_fraction')), "#8CF0AD")
    with col3: kpi_card("👥 Customers", f"{kpis.get('Customers',{}).get('last',0):,.0f}", format_growth(kpis.get('Customers',{}).get('growth_fraction')), "#C392F0")
    with col4:
        churn_val = kpis.get('ChurnRate',{}).get('last',None)
        churn_display = f"{churn_val*100:.2f}%" if churn_val else "N/A"
        kpi_card("🔄 Churn Rate", churn_display, format_growth(kpis.get('ChurnRate',{}).get('growth_fraction')), "#E497A4")

    # Trends Over Time
    st.markdown("### 📈 Trends Over Time")
    if time_col:
        df_plot = df.copy()
        if time_col == "Month":
            df_plot["Date"] = pd.to_datetime(df_plot["Month"].str.replace(" ", "-"), format="%b-%Y", errors="coerce")
        elif time_col == "Quarter":
            def q_to_date(q):
                try:
                    parts = str(q).replace(" ", "-").split("-")
                    qnum, year = int(parts[0][1:]), int(parts[1])
                    month = (qnum - 1) * 3 + 1
                    return pd.Timestamp(year=year, month=month, day=1)
                except: return pd.NaT
            df_plot["Date"] = df_plot["Quarter"].apply(q_to_date)
        elif time_col == "Year":
            df_plot["Date"] = pd.to_datetime(df_plot["Year"].astype(str) + "-01-01", errors='coerce')
        df_plot = df_plot.dropna(subset=["Date"]).reset_index(drop=True)

        if len(df_plot) > 200:
            df_plot = df_plot.set_index("Date").sort_index().resample("M").mean(numeric_only=True).reset_index()

        chart_type = st.session_state.get('trend_chart_type', 'Line Chart')
        fig = go.Figure()
        if chart_type == "Line Chart":
            fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Revenue"], mode="lines+markers", name="Revenue"))
            fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Profit"], mode="lines+markers", name="Profit"))
        elif chart_type == "Bar Chart":
            fig.add_trace(go.Bar(x=df_plot["Date"], y=df_plot["Revenue"], name="Revenue"))
            fig.add_trace(go.Bar(x=df_plot["Date"], y=df_plot["Profit"], name="Profit"))
        elif chart_type == "Area Chart":
            fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Revenue"], mode="lines", fill='tozeroy', name="Revenue"))
            fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Profit"], mode="lines", fill='tozeroy', name="Profit"))
        elif chart_type == "Scatter Chart":
            fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Revenue"], mode="markers", name="Revenue"))
            fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Profit"], mode="markers", name="Profit"))
        fig.update_layout(title=f"Revenue & Profit Trends ({chart_type})", xaxis_title="Date", yaxis_title="Amount", template="plotly_white", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    # Forecast
    st.markdown("### 🔮 Forecast Next 5 Years")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        df_prop = prepare_prophet_df(df, col, time_col)
        if df_prop is not None and not df_prop.empty:
            forecast = forecast_prophet(df_prop)
            if forecast is not None:
                plot_forecast(df_prop, forecast, col)
# ----------------- Anomalies -----------------
    report_df = locals().get("df_plot", df)
    if "ChurnRate" in report_df.columns and "Date" in report_df.columns:
        alerts = []
        churn_s = report_df["ChurnRate"].reset_index(drop=True)
        for i in range(1, len(churn_s)):
            if not pd.isna(churn_s.iloc[i-1]) and churn_s.iloc[i] > churn_s.iloc[i-1] * 1.2:
                alerts.append((report_df["Date"].iloc[i], churn_s.iloc[i]))
        if alerts:
            for t, v in alerts:
                st.error(f"⚠️ Churn spike detected on {pd.to_datetime(t).strftime('%Y-%m-%d')}: {v*100:.2f}%")
        else:
            st.success("✅ No churn anomalies detected.")

    # ----------------- Auto Insights -----------------
    try:
        prompt = f"Given this data:\n{df.to_string(index=False)}\n\nList 3 key business insights in plain English."
        insights = ask_llm("You are a business analyst.", prompt, max_tokens=200, temperature=0.3)
        st.session_state.insights = insights
        st.markdown(f"**Insights:** {insights}")
    except Exception as e:
        st.warning("Could not generate auto insights. Check API key or quota.")
        st.error(str(e))

    # ----------------- Export PDF & Build RAG -----------------
    exp_col1, exp_col2, exp_col3 = st.columns([1,1,1])
    with exp_col1:
        if st.button("📤 Export Insights as PDF"):
            if st.session_state.insights:
                filename = "kpi_report.pdf"
                export_pdf(kpis, st.session_state.insights, st.session_state.messages, filename)
                with open(filename, "rb") as f:
                    st.download_button("⬇️ Download Report", f, file_name=filename, mime="application/pdf")
            else:
                st.warning("No insights available to export.")

    with exp_col2:
        if st.button("Build RAG index from this file"):
            try:
                info = build_index_for_dataframe(df, index_name="index_demo")
                st.session_state.index_name = info["index_name"]
                st.success(f"Built index `{info['index_name']}` with {info['num_docs']} chunks.")
            except Exception as e:
                st.error("Failed to build index: " + str(e))
                st.error(traceback.format_exc())

    with exp_col3:
        if st.button("💾 Save to PostgreSQL"):
            save_to_postgres(df)

# ----------------- Chat Section -----------------
st.divider()
st.subheader("💬 Chat with your data")

# Language selection
lang = st.sidebar.selectbox(
    "Chat Response Language",
    ["English", "Tamil", "Telugu", "Kannada"],
    index=0
)

if st.session_state.index_name is None:
    st.info("No index available. Upload a file (or use sample) and press 'Build RAG index'.")
else:
    chat_col, _ = st.columns([3,1])
    with chat_col:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(f"👤 **You:** {msg['content']}")
                else:
                    st.markdown(f"🤖 **Assistant ({lang}):** {msg['content']}")

        st.markdown("👉 Quick Questions:")
        qcols = st.columns(3)
        preset = ["Why did revenue change last period?", "How does churn affect profit?", "What actions should management take?"]
        user_input = None
        for i, q in enumerate(preset):
            if qcols[i].button(q):
                user_input = q

        typed_input = st.chat_input("Ask about your KPIs...")
        if typed_input:
            user_input = typed_input

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        hits = rag_query(st.session_state.index_name, user_input, top_k=5)
                        context_text = "\n\n".join([f"{h['text']}" for h in hits])

                        # optional web search if benchmarking
                        web_results = []
                        if any(w in user_input.lower() for w in ["compare","industry","benchmark","market"]):
                            web_results = search_web(user_input, num=2)
                        web_context = "\n".join([f"- {r['title']}: {r['snippet']} ({r['link']})" for r in web_results])

                        # system prompt with language support
                        system_prompt = (
                            f"You are a helpful business analyst assistant. "
                            f"Use the company data and web results to answer. "
                            f"Always reply in {lang}. "
                            f"Explain WHAT changed, WHY, and suggest NEXT steps in 2-3 sentences."
                        )

                        full_prompt = f"{context_text}\n\nWeb info:\n{web_context}\n\nQuestion: {user_input}"
                        ans = ask_llm(system_prompt, full_prompt, max_tokens=350, temperature=0.3)

                        st.markdown(ans)
                        st.session_state.messages.append({"role":"assistant","content":ans})
                    except Exception as e:
                        st.error("Failed to fetch answer: " + str(e))
