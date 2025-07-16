import streamlit as st
import os
import json
import datetime as dt
from typing import Dict, List, Any

from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import plotly.express as px

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFacePipeline
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

# ==============================================================================
# 1. CORE PIPELINE LOGIC
# ==============================================================================

# --------  CONFIG  --------
CONFIG = {
    "max_articles_per_ticker": 10,
    "gap_threshold_pct": 2.0,
    "summarizer_model": "gpt-4o",
    "verifier_model": "gpt-4o",
    "analyst_model": "gemini-2.5-pro",
    "fingpt_model": "FinGPT/fingpt-sentiment_llama2-13b_lora",
}

# --------  API KEYS & MODELS (with caching and deployment fix) --------
@st.cache_resource
def load_models_and_keys():
    """Load API keys from .env and initialize models once."""
    load_dotenv()
    keys = {
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN")
    }
    if not all([keys["SERPER_API_KEY"], keys["OPENAI_API_KEY"], keys["GOOGLE_API_KEY"]]):
        return None, None

    models = {
        "summarizer_llm": ChatOpenAI(model=CONFIG["summarizer_model"], temperature=0.2, api_key=keys["OPENAI_API_KEY"]),
        "verifier_llm": ChatOpenAI(model=CONFIG["verifier_model"], temperature=0.0, api_key=keys["OPENAI_API_KEY"]),
        "analyst_llm": ChatGoogleGenerativeAI(model=CONFIG["analyst_model"], temperature=0.0, model_kwargs={"response_mime_type": "application/json"}, api_key=keys["GOOGLE_API_KEY"]),
        
        # --- FIX: Removed device_map="auto" for Streamlit Cloud (CPU-only) compatibility ---
        "fingpt": HuggingFacePipeline(model_id=CONFIG["fingpt_model"], task="text-classification", return_all_scores=True)
    }
    return keys, models

# --------  STATE DEFINITION  --------
class PipelineState(Dict):
    tickers: List[str]
    start: dt.date
    end: dt.date
    news_raw: Dict[str, List[Dict]]
    news_summaries: Dict[str, List[Dict]]
    sentiment_scored: Dict[str, List[Dict]]
    prices_raw: pd.DataFrame
    finance_analysis: Dict[str, Dict]
    verification: str | Dict
    report: str
    errors: List[str]

# --------  CHAINS & PROMPTS  --------
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are FinancialNewsSummarizerGPT. Produce exactly 3 concise bullet points (max 60 tokens each) accurately preserving all numbers, dates, and financial terms."),
    ("human", "{article_text}")
])
verify_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are FactCheckGPT. Cross-reference the financial analysis with the news sentiment. If there are no major inconsistencies or surprising correlations, output 'OK'. Otherwise, provide a brief JSON list of observations or corrections."),
    ("human", "Sentiment Analysis JSON:\n```{sent_json}```\n\nFinancial Data JSON:\n```{fin_json}```")
])
aggregate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are ChiefInvestmentStrategistGPT. Combine the news sentiment, financial data, and verification notes to create a final investment score from 1 (strong sell) to 10 (strong buy) for each ticker. Explain your reasoning for each score. Return a single JSON object where keys are the tickers."),
    ("human", "Sentiment Data:\n{sent}\n\nFinancial Data:\n{fin}\n\nVerification Notes:\n{corr}")
])

# --------  NODE HELPERS --------
def fetch_news(state: PipelineState) -> PipelineState:
    serper = GoogleSerperAPIWrapper(type_="news")
    out = {}
    for t in state["tickers"]:
        try:
            query = f"\"{t}\" stock news after:{state['start']:%Y-%m-%d} before:{state['end']:%Y-%m-%d}"
            result = serper.run(query)
            if isinstance(result, str) and "error" in result.lower(): raise Exception(f"Serper API error for {t}: {result}")
            out[t] = json.loads(result).get("news", [])
        except Exception as e:
            state["errors"].append(f"Failed to fetch news for {t}: {e}")
            out[t] = []
    state["news_raw"] = out
    return state

def summarise_news(state: PipelineState, summary_chain) -> PipelineState:
    summaries = {}
    for t, articles in state["news_raw"].items():
        if not articles:
            summaries[t] = []
            continue
        batch_inputs = [{"article_text": (art.get("snippet") or art.get("title", ""))} for art in articles[:CONFIG["max_articles_per_ticker"]] if (art.get("snippet") or art.get("title"))]
        if not batch_inputs: continue
        try:
            batch_results = summary_chain.batch(batch_inputs, {"max_concurrency": 5})
            summaries[t] = [{"summary": summary_text, "link": articles[i].get("link"), "published": articles[i].get("date")} for i, summary_text in enumerate(batch_results)]
        except Exception as e:
            state["errors"].append(f"Failed to summarize news for {t}: {e}")
            summaries[t] = []
    state["news_summaries"] = summaries
    return state

def score_sentiment(state: PipelineState, fingpt) -> PipelineState:
    scored = {}
    for t, items in state["news_summaries"].items():
        scored[t] = []
        for it in items:
            try:
                probs = fingpt(it["summary"])[0]
                pmap = {d["label"].lower(): d["score"] for d in probs}
                score = pmap.get("positive", 0) - pmap.get("negative", 0)
                scored[t].append({**it, "sentiment_score": round(score, 3)})
            except Exception as e:
                state["errors"].append(f"Failed to score sentiment for an article in {t}: {e}")
    state["sentiment_scored"] = scored
    return state

def fetch_finance_data(state: PipelineState) -> PipelineState:
    try:
        end_date = state["end"] + dt.timedelta(days=1)
        df = yf.download(state["tickers"], start=state["start"], end=end_date, progress=False)
        if df.empty: raise ValueError("No data returned from Yahoo Finance.")
        state["prices_raw"] = df
    except Exception as e:
        state["errors"].append(f"Failed to download financial data for tickers {state['tickers']}: {e}")
        state["prices_raw"] = pd.DataFrame()
    return state

def analyse_finance_data(state: PipelineState) -> PipelineState:
    if state["prices_raw"].empty:
        state["finance_analysis"] = {t: {"error": "Missing price data"} for t in state["tickers"]}
        return state
    analysis = {}
    if isinstance(state['prices_raw'].columns, pd.MultiIndex):
        df_full = state["prices_raw"].stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        df_full = state['prices_raw'].copy()
        df_full['Ticker'] = state['tickers'][0]
        df_full = df_full.reset_index()
    for ticker in state["tickers"]:
        df = df_full[df_full['Ticker'] == ticker].dropna()
        if len(df) < 2:
            analysis[ticker] = {"error": "Not enough data for analysis."}
            continue
        n_day_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        largest_move = df['Close'].pct_change().abs().max()
        analysis[ticker] = {"period_return_pct": round(n_day_return * 100, 2), "largest_daily_move_pct": round(largest_move * 100, 2)}
    state["finance_analysis"] = analysis
    return state

def crosscheck(state: PipelineState, verify_chain) -> PipelineState:
    try:
        result = verify_chain.invoke({"sent_json": json.dumps(state["sentiment_scored"]), "fin_json": json.dumps(state["finance_analysis"])})
        try: state["verification"] = json.loads(result)
        except json.JSONDecodeError: state["verification"] = result
    except Exception as e:
        state["errors"].append(f"Verification step failed: {e}")
        state["verification"] = "Verification failed due to an error."
    return state

def aggregate(state: PipelineState, aggregate_chain) -> PipelineState:
    try:
        report = aggregate_chain.invoke({"sent": json.dumps(state["sentiment_scored"]), "fin": json.dumps(state["finance_analysis"]), "corr": json.dumps(state["verification"])})
        state["report"] = report
    except Exception as e:
        state["errors"].append(f"Final aggregation step failed: {e}")
        state["report"] = {"error": "Report generation failed."}
    return state

# ==============================================================================
# 2. STREAMLIT DASHBOARD UI
# ==============================================================================

st.set_page_config(layout="wide", page_title="Financial Sentiment Dashboard")
st.title("Financial News & Sentiment Analysis Dashboard")
st.markdown("An interactive dashboard to analyze financial news sentiment and price data using a multi-agent LLM pipeline.")

api_keys, models = load_models_and_keys()
if not models:
    st.error("🚨 **API Keys Not Found!** Please add `OPENAI_API_KEY`, `GOOGLE_API_KEY`, and `SERPER_API_KEY` to your Streamlit Secrets.")
    st.stop()

summary_chain = summary_prompt | models['summarizer_llm'] | StrOutputParser()
verify_chain = verify_prompt | models['verifier_llm'] | StrOutputParser()
aggregate_chain = aggregate_prompt | models['analyst_llm'] | JsonOutputParser()

@st.cache_data(show_spinner=False)
def run_pipeline(tickers, start_date, end_date):
    """Builds and runs the LangGraph pipeline, decorated to cache results."""
    g = StateGraph(PipelineState)
    g.add_node("fetch_news", fetch_news)
    g.add_node("summarise_news", lambda state: summarise_news(state, summary_chain))
    g.add_node("score_sentiment", lambda state: score_sentiment(state, models['fingpt']))
    g.add_node("fetch_finance", fetch_finance_data)
    g.add_node("analyse_finance", analyse_finance_data)
    g.add_node("crosscheck", lambda state: crosscheck(state, verify_chain))
    g.add_node("aggregate", lambda state: aggregate(state, aggregate_chain))
    g.set_entry_point("fetch_news")
    g.add_edge("fetch_news", "summarise_news"); g.add_edge("summarise_news", "score_sentiment")
    g.add_edge("fetch_news", "fetch_finance"); g.add_edge("fetch_finance", "analyse_finance")
    g.add_edge("score_sentiment", "crosscheck"); g.add_edge("analyse_finance", "crosscheck")
    g.add_edge("crosscheck", "aggregate"); g.add_edge("aggregate", END)
    
    pipeline = g.compile()
    initial_state = { "tickers": tickers, "start": start_date, "end": end_date, "errors": [] }
    return pipeline.invoke(initial_state)

st.sidebar.header("Analysis Configuration")
if 'available_tickers' not in st.session_state:
    st.session_state.available_tickers = ['NVDA', 'GOOGL', 'MSFT', 'AAPL', 'TSLA', 'AMZN', 'META']
new_ticker = st.sidebar.text_input("Add Ticker Symbol", placeholder="e.g., CRM").strip().upper()
if st.sidebar.button("Add Ticker"):
    if new_ticker and new_ticker not in st.session_state.available_tickers:
        st.session_state.available_tickers.append(new_ticker)
    elif not new_ticker: st.sidebar.warning("Please enter a ticker symbol.")
    else: st.sidebar.warning(f"{new_ticker} is already in the list.")
selected_tickers = st.sidebar.multiselect("Select Stock Tickers for Analysis", options=st.session_state.available_tickers, default=['NVDA', 'MSFT'])
today = dt.date.today()
start_date = st.sidebar.date_input("Start Date", value=today - dt.timedelta(days=7))
end_date = st.sidebar.date_input("End Date", value=today)

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    if not selected_tickers: st.warning("Please select at least one ticker.")
    elif start_date >= end_date: st.warning("Start Date must be before End Date.")
    else:
        with st.spinner("Analyzing... This may take a few minutes. Processing news, financials, and running LLM agents..."):
            st.session_state.final_state = run_pipeline(selected_tickers, start_date, end_date)

if 'final_state' in st.session_state:
    state = st.session_state.final_state
    if state.get("errors"):
        st.error("### Errors Occurred During Analysis")
        for error in state["errors"]: st.code(error)
    tab_titles = ["📈 Summary Report"] + [f"{ticker} Details" for ticker in state["tickers"]]
    tabs = st.tabs(tab_titles)
    with tabs[0]:
        st.header("Chief Investment Strategist Report")
        report_data = state.get("report")
        if report_data and "error" not in report_data: st.json(report_data)
        else: st.warning("Could not generate the final report.")
        st.subheader("Verification Notes"); st.write(state.get("verification", "No verification notes."))
    for i, ticker in enumerate(state["tickers"]):
        with tabs[i+1]:
            st.header(f"Detailed Analysis for {ticker}")
            st.subheader("Financial Overview")
            fin_analysis = state.get("finance_analysis", {}).get(ticker, {})
            if "error" in fin_analysis:
                st.warning(f"Could not perform financial analysis for {ticker}: {fin_analysis['error']}")
            else:
                col1, col2 = st.columns(2)
                col1.metric(label=f"Return ({start_date} to {end_date})", value=f"{fin_analysis.get('period_return_pct', 0)}%")
                col2.metric(label="Largest Single-Day Move", value=f"{fin_analysis.get('largest_daily_move_pct', 0)}%")
                prices_df = state.get("prices_raw")
                if prices_df is not None and not prices_df.empty:
                    ticker_prices = prices_df.loc[:, pd.IndexSlice[:, ticker]] if isinstance(prices_df.columns, pd.MultiIndex) else prices_df
                    fig = px.line(ticker_prices, y="Close", title=f"{ticker} Stock Price", labels={"Date": "Date", "Close": "Closing Price (USD)"})
                    st.plotly_chart(fig, use_container_width=True)
            st.subheader("News Sentiment Analysis")
            sentiment_data = state.get("sentiment_scored", {}).get(ticker, [])
            if not sentiment_data: st.info(f"No news articles found or processed for {ticker}.")
            for item in sentiment_data:
                with st.expander(f"**{item.get('published', 'Date N/A')}** | Score: {item.get('sentiment_score', 0)}"):
                    st.markdown(item.get("summary", "No summary available."))
                    st.link_button("Go to Article", item.get("link", "#"))
else:
    st.info("Configure your analysis in the sidebar and click 'Run Analysis' to begin.")
