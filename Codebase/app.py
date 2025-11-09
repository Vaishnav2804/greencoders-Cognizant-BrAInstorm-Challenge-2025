# app.py
import os
import uuid
from dotenv import load_dotenv
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime

# LLM model mgmt
from langchain_google_genai import ChatGoogleGenerativeAI

# Core module (with LangSmith-enabled functions and sustainability points)
from core import run_full_analysis, answer_sustainability_question
import core as core_mod  # to update core.llm when the model changes

load_dotenv()

# -------------------- Session storage --------------------
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)

_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


# -------------------- Visualization helpers (User) --------------------
def create_pie_chart(chart_data):
    """Sustainability-focused pie chart."""
    if not chart_data or "labels" not in chart_data:
        return None
    colors = ["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#c0392b", "#95a5a6"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=chart_data["labels"],
                values=chart_data["values"],
                marker=dict(colors=colors),
                textposition="inside",
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        height=450,
        title={"text": chart_data.get("title", ""), "font": {"size": 16}},
        font=dict(size=12),
    )
    return fig


def create_bar_chart(chart_data, title=""):
    """Seasonal bar chart."""
    if not chart_data or "x" not in chart_data:
        return None
    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_data["x"],
                y=chart_data["y"],
                marker=dict(
                    color=["#2ecc71", "#27ae60", "#f39c12", "#e74c3c"][
                        : len(chart_data["x"])
                    ]
                ),
                text=[f"${v:.0f}" for v in chart_data["y"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title={"text": chart_data.get("title", title), "font": {"size": 16}},
        xaxis_title="Category",
        yaxis_title="Amount ($)",
        height=400,
        font=dict(size=12),
        showlegend=False,
    )
    return fig


def create_line_chart(chart_data):
    """Trend line chart."""
    if not chart_data or "x" not in chart_data:
        return None
    fig = go.Figure(
        data=[
            go.Scatter(
                x=chart_data["x"],
                y=chart_data["y"],
                mode="lines+markers",
                fill="tozeroy",
                name="Sustainability Score",
                line=dict(color="#3498db", width=3),
                marker=dict(size=8, color="#2980b9"),
            )
        ]
    )
    fig.update_layout(
        title={"text": chart_data.get("title", "Score Trend"), "font": {"size": 16}},
        xaxis_title="Month",
        yaxis_title="Score (0-100)",
        height=400,
        font=dict(size=12),
        showlegend=False,
        yaxis=dict(range=[0, 100]),
    )
    return fig


def create_score_gauge(score):
    """Sustainability score gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Sustainability Score"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": "#2ecc71"
                    if score >= 70
                    else "#f39c12"
                    if score >= 50
                    else "#e74c3c"
                },
                "steps": [
                    {"range": [0, 33], "color": "#ffe6e6"},
                    {"range": [33, 66], "color": "#fff9e6"},
                    {"range": [66, 100], "color": "#e6ffe6"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )
    fig.update_layout(height=400, font=dict(size=12))
    return fig


# -------------------- Visualization helpers (Bank) --------------------
def create_histogram(scores: List[float], title="Score Distribution"):
    if not scores:
        return None
    fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=20, marker_color="#3498db")])
    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        xaxis_title="Score",
        yaxis_title="Users",
        height=400,
        font=dict(size=12),
        bargap=0.05,
    )
    return fig


def create_points_bar(
    points_by_category: Dict[str, float], title="Total Points by Category"
):
    if not points_by_category:
        return None
    cats = list(points_by_category.keys())
    vals = [points_by_category[c] for c in cats]
    fig = go.Figure(
        data=[
            go.Bar(
                x=cats,
                y=vals,
                marker_color="#27ae60",
                text=[f"{v:,.0f}" for v in vals],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        xaxis_title="Category",
        yaxis_title="Total Points",
        height=400,
        font=dict(size=12),
        showlegend=False,
    )
    return fig


def create_agg_line(
    x: List[str], y: List[float], title="Average Monthly Score (All Users)"
):
    if not x:
        return None
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(color="#8e44ad", width=3),
                marker=dict(size=7),
            )
        ]
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        xaxis_title="Month",
        yaxis_title="Average Score",
        height=400,
        font=dict(size=12),
        yaxis=dict(range=[0, 100]),
        showlegend=False,
    )
    return fig


# -------------------- Model management --------------------
SUPPORTED_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-pro-exp",
]


def update_core_llm(model_name: str):
    """Reinitialize core.llm with the selected model."""
    try:
        core_mod.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            max_output_tokens=None,
            timeout=60,
            max_retries=3,
        )
        return f"Model set to {model_name}"
    except Exception as e:
        return f"Failed to set model: {str(e)}"


# -------------------- User Analysis function --------------------
def analyze_credit_data(file):
    """Process and analyze card/bank data (supports credit & debit outflows)."""
    if file is None:
        return "📋 Please upload a CSV file", None, None, None, None, None, {}

    try:
        df = pd.read_csv(file.name)
        required_cols = {"date", "description", "credit", "debit"}
        if not required_cols.issubset(df.columns):
            return (
                f"❌ Missing columns: {required_cols - set(df.columns)}",
                None,
                None,
                None,
                None,
                None,
                {},
            )

        # Optional session for tracing consistency
        result = run_full_analysis(df, session_id=f"user-{uuid.uuid4().hex}")

        # Summary metrics
        score = result["score"]
        carbon = result["carbon_total"]
        spending = result["total_spending"]
        transactions = result["transaction_count"]

        # Report text
        report = f"""
# 🌿 Sustainability Report

## Overview
- **Sustainability Score:** `{score}/100`
- **Total CO₂ Equivalent:** `{carbon:.2f} kg` 🌍
- **Total Spending:** `${spending:,.2f}`
- **Transactions:** `{transactions}`

## Environmental Impact
- **Carbon per Dollar:** `{result["carbon_per_dollar"]:.4f} kg CO₂/$`
- **Trees Needed to Offset:** ~`{int(carbon / 20)}`

---

## Detailed Analysis

{result["insights"]}
"""

        return (
            report,
            create_score_gauge(score),
            create_pie_chart(result["pie_data"]),
            create_bar_chart(result["bar_data"]),
            create_line_chart(result["line_data"]),
            create_bar_chart(result["carbon_data"], "Carbon Footprint"),
            result,
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None, None, None, {}


# -------------------- Chat (User Dashboard) --------------------
def _extract_file_paths(files):
    """Normalize files from MultimodalTextbox to local file paths."""
    paths = []
    for f in files or []:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "name"):
            paths.append(f.name)
        elif isinstance(f, dict) and "path" in f:
            paths.append(f["path"])
    return paths


def chatbot_reply(message, history, analysis_state, session_state, model_name):
    """Chat with context from analysis; supports multimodal textbox with multiple files."""
    if not analysis_state or "insights" not in analysis_state:
        return "Please analyze your spending data first! 📊", session_state

    # Session ID
    session_id = session_state.get("session_id")
    if not session_id:
        session_id = f"session-{uuid.uuid4().hex}"
        session_state["session_id"] = session_id

    # Ensure model is set for this turn
    selected_model = model_name or SUPPORTED_MODELS[0]
    status = update_core_llm(selected_model)
    session_state["model_status"] = status
    session_state["model_name"] = selected_model

    # Incoming message contains text + files (images/audio/etc.)
    user_msg = message.get("text") if isinstance(message, dict) else message
    files = message.get("files", []) if isinstance(message, dict) else []
    file_paths = _extract_file_paths(files)

    if file_paths:
        responses = []
        for p in file_paths:
            responses.append(
                answer_sustainability_question(
                    user_msg or "", analysis_state, image_path=p, session_id=session_id
                )
            )
        response = "\n\n".join(responses)
    else:
        response = answer_sustainability_question(
            user_msg or "", analysis_state, image_path=None, session_id=session_id
        )

    return response, session_state


# -------------------- Bank Dashboard: Aggregation --------------------
def _validate_df(df: pd.DataFrame) -> Tuple[bool, str]:
    required_cols = {"date", "description", "credit", "debit"}
    if not required_cols.issubset(df.columns):
        return False, f"Missing columns: {required_cols - set(df.columns)}"
    return True, ""


def _aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple run_full_analysis results.
    Returns dict with aggregate metrics and structures suitable for charts.
    """
    if not results:
        return {}

    # Aggregate scalars
    total_spend = sum(r["total_spending"] for r in results)
    total_carbon = sum(r["carbon_total"] for r in results)
    # Weighted average score by spend (more fair)
    weighted_score_num = sum(r["score"] * r["total_spending"] for r in results)
    avg_score = (weighted_score_num / total_spend) if total_spend > 0 else 0.0
    avg_cpd = (
        (
            sum(r["carbon_per_dollar"] * r["total_spending"] for r in results)
            / total_spend
        )
        if total_spend > 0
        else 0.0
    )
    users_n = len(results)

    # Category spend totals
    agg_category_spend = defaultdict(float)
    for r in results:
        for cat, info in r["category_data"].items():
            agg_category_spend[cat] += float(info["amount"])
    # Prepare pie
    cats_sorted = sorted(agg_category_spend.items(), key=lambda x: x[1], reverse=True)
    pie_data = {
        "labels": [c for c, _ in cats_sorted],
        "values": [v for _, v in cats_sorted],
        "title": "Aggregate Spending by Category",
    }

    # Points by category
    points_by_category = defaultdict(float)
    for r in results:
        for cat, pts in (
            r.get("points_details", {}).get("points_by_category", {}).items()
        ):
            points_by_category[cat] += float(pts)

    # Monthly average scores: combine monthly_trends across users on "month" string
    monthly_map = defaultdict(lambda: {"score_sum": 0.0, "count": 0})
    for r in results:
        mdf = r.get("monthly_trends", None)
        if isinstance(mdf, pd.DataFrame) and not mdf.empty:
            for _, row in mdf.iterrows():
                m = str(row["month"])
                monthly_map[m]["score_sum"] += float(row["score"])
                monthly_map[m]["count"] += 1
    months = sorted(monthly_map.keys())
    avg_scores_per_month = [
        (monthly_map[m]["score_sum"] / monthly_map[m]["count"])
        if monthly_map[m]["count"] > 0
        else 0.0
        for m in months
    ]

    # Score distribution
    score_list = [float(r["score"]) for r in results]

    return {
        "users": users_n,
        "total_spending": total_spend,
        "total_carbon": total_carbon,
        "avg_score": avg_score,
        "avg_carbon_per_dollar": avg_cpd,
        "pie_data": pie_data,
        "points_by_category": dict(
            sorted(points_by_category.items(), key=lambda x: x[1], reverse=True)
        ),
        "monthly_x": months,
        "monthly_y": avg_scores_per_month,
        "scores": score_list,
    }


def bank_ingest(files, dir_path):
    """
    Ingest multiple CSVs (uploaded or from a directory), run analysis per file, and return aggregates + charts.
    """
    results = []
    errors = []

    paths = []
    # Uploaded files
    if files:
        for f in files:
            try:
                if hasattr(f, "name"):
                    paths.append(f.name)
                elif isinstance(f, dict) and "path" in f:
                    paths.append(f["path"])
                elif isinstance(f, str):
                    paths.append(f)
            except Exception:
                continue
    # Directory scan
    if dir_path:
        dir_path = dir_path.strip()
        if os.path.isdir(dir_path):
            for root, _, fnames in os.walk(dir_path):
                for fn in fnames:
                    if fn.lower().endswith(".csv"):
                        paths.append(os.path.join(root, fn))

    # De-duplicate and limit to reasonable batch
    seen = set()
    unique_paths = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            unique_paths.append(p)

    for p in unique_paths:
        try:
            df = pd.read_csv(p)
            ok, msg = _validate_df(df)
            if not ok:
                errors.append(f"{os.path.basename(p)}: {msg}")
                continue
            # Use a deterministic session id label for bank runs
            res = run_full_analysis(
                df, session_id=f"bank-{os.path.basename(p)}-{uuid.uuid4().hex[:6]}"
            )
            # Augment with filename for potential future drilldowns
            res["_source_file"] = os.path.basename(p)
            results.append(res)
        except Exception as e:
            errors.append(f"{os.path.basename(p)}: {str(e)}")

    if not results:
        err_text = "❌ No valid CSVs found."
        if errors:
            err_text += "\n\n- " + "\n- ".join(errors[:10])
        return err_text, None, None, None, None, None, ""

    agg = _aggregate_results(results)

    # Build bank report markdown
    report = f"""
# 🏦 Bank Sustainability Dashboard

## Overview
- **Users (files processed):** `{agg["users"]}`
- **Aggregate Spending:** `${agg["total_spending"]:,.2f}`
- **Aggregate CO₂:** `{agg["total_carbon"]:,.2f} kg`
- **Avg Sustainability Score (weighted):** `{agg["avg_score"]:.2f}/100`
- **Avg Carbon per Dollar:** `{agg["avg_carbon_per_dollar"]:.4f} kg CO₂/$`

---

## Notes
- Scores are weighted by user spending to reflect financial impact.
- Monthly averages are simple means across users reporting that month.
"""

    # Charts
    pie_fig = create_pie_chart(agg["pie_data"])
    points_fig = create_points_bar(agg["points_by_category"])
    monthly_line = create_agg_line(agg["monthly_x"], agg["monthly_y"])
    score_hist = create_histogram(agg["scores"])

    # Prepare downloadable artifacts (JSON + CSV)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"bank_aggregate_{ts}.json"
    csv_path = f"bank_aggregate_{ts}.csv"

    # Flatten for CSV
    flat_rows = []
    for cat, amt in agg["pie_data"]["labels"] and zip(
        agg["pie_data"]["labels"], agg["pie_data"]["values"]
    ):
        flat_rows.append({"metric": "category_spend", "category": cat, "value": amt})
    for cat, pts in agg["points_by_category"].items():
        flat_rows.append({"metric": "category_points", "category": cat, "value": pts})
    for m, s in zip(agg["monthly_x"], agg["monthly_y"]):
        flat_rows.append({"metric": "monthly_avg_score", "month": m, "value": s})
    # Header stats
    flat_rows.append({"metric": "users", "value": agg["users"]})
    flat_rows.append({"metric": "total_spending", "value": agg["total_spending"]})
    flat_rows.append({"metric": "total_carbon", "value": agg["total_carbon"]})
    flat_rows.append({"metric": "avg_score", "value": agg["avg_score"]})
    flat_rows.append(
        {"metric": "avg_carbon_per_dollar", "value": agg["avg_carbon_per_dollar"]}
    )

    try:
        # JSON
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(agg, jf, indent=2)
        # CSV
        pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
        download_links = f"✅ Artifacts ready:\n- {json_path}\n- {csv_path}"
    except Exception as e:
        download_links = f"⚠️ Could not write artifacts: {str(e)}"
        json_path, csv_path = "", ""

    return (
        report,
        pie_fig,
        points_fig,
        monthly_line,
        score_hist,
        json_path,
        download_links,
    )


# -------------------- Gradio UI (Two Dashboards) --------------------
with gr.Blocks(
    title="🌿 Sustainability Points Analyzer (User & Bank Dashboards)",
    theme=gr.themes.Soft(),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .header-title {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 2rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 2rem;
    }
    .upload-area { border: 2px dashed #2ecc71; border-radius: 8px; padding: 2rem; text-align: center; background: #f0fff4; }
    """,
) as demo:
    # Header
    with gr.Group(elem_classes="header-title"):
        gr.Markdown("""
# 🌿 Sustainability Points Analyzer
User & Bank Dashboards for Sustainable Spending Intelligence
        """)

    # State
    analysis_state = gr.State({})
    session_state = gr.State({})
    bank_state = gr.State({})

    # Global model selector
    with gr.Row():
        model_selector = gr.Dropdown(
            label="LLM Model",
            choices=SUPPORTED_MODELS,
            value=SUPPORTED_MODELS[0],
            allow_custom_value=False,
        )
        model_status = gr.Markdown("Model set to gemini-2.5-flash-lite")

    # Hook model selector change
    def _on_model_change(model_name, session_state):
        status = update_core_llm(model_name)
        session_state = session_state or {}
        session_state["model_name"] = model_name
        session_state["model_status"] = status
        return status, session_state

    model_selector.change(
        fn=_on_model_change,
        inputs=[model_selector, session_state],
        outputs=[model_status, session_state],
    )

    # ---------- USER DASHBOARD ----------
    with gr.Tab("👤 User Dashboard"):
        with gr.Tab("📊 Analysis"):
            with gr.Group():
                gr.Markdown("### 📤 Upload Your Credit or Debit Statement")
                gr.Markdown(
                    "*CSV format: date, description, credit, debit (supports DD/MM/YYYY)*"
                )
                with gr.Row():
                    file_upload = gr.File(
                        label="Choose CSV File",
                        file_types=[".csv"],
                        elem_classes="upload-area",
                    )
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary", scale=1)

            # Score gauge
            with gr.Group():
                gr.Markdown("### Your Sustainability Score")
                score_gauge = gr.Plot(label="Sustainability Gauge")

            # Main report
            with gr.Group():
                gr.Markdown("### 📋 Detailed Report")
                report_text = gr.Markdown()

            # Visualizations
            gr.Markdown("### 📈 Visualizations")
            with gr.Row():
                pie_chart = gr.Plot(label="Spending Distribution")
                seasonal_chart = gr.Plot(label="Seasonal Breakdown")

            with gr.Row():
                trend_chart = gr.Plot(label="Sustainability Trend")
                carbon_chart = gr.Plot(label="Carbon Footprint")

            # Analysis button click
            analyze_btn.click(
                fn=analyze_credit_data,
                inputs=[file_upload],
                outputs=[
                    report_text,
                    score_gauge,
                    pie_chart,
                    seasonal_chart,
                    trend_chart,
                    carbon_chart,
                    analysis_state,
                ],
            )

        with gr.Tab("💬 Ask Me Anything"):
            gr.Markdown("""
### 🤖 Sustainability Assistant
Ask questions about your spending habits, get personalized recommendations, and learn how to improve!

- Upload grocery or utility bills as images to get item-level insights and automatic JSON logging.
- You can also record short voice notes; they'll be attached to your message.
            """)

            textbox = gr.MultimodalTextbox(
                file_count="multiple",
                file_types=["image", "audio"],
                sources=["upload", "microphone"],
                label="Type your message and attach files (images/audio)...",
            )

            chatbot = gr.ChatInterface(
                fn=chatbot_reply,
                additional_inputs=[analysis_state, session_state, model_selector],
                additional_outputs=[session_state],
                type="messages",
                multimodal=True,
                textbox=textbox,
                fill_height=True,
            )

    # ---------- BANK DASHBOARD ----------
    with gr.Tab("🏦 Bank Dashboard"):
        gr.Markdown("""
### Aggregate Sustainable Patterns Across Users
Upload multiple CSV statements or point to a directory of CSVs to compute aggregate sustainability metrics.
        """)

        with gr.Row():
            bank_files = gr.Files(
                label="Upload Multiple CSVs (Optional)",
                file_count="multiple",
                file_types=[".csv"],
                elem_classes="upload-area",
            )
            dir_input = gr.Textbox(
                label="Directory Path (Optional)",
                placeholder="e.g., ./data/users",
            )

        ingest_btn = gr.Button("📥 Ingest & Aggregate", variant="primary")

        bank_report = gr.Markdown(label="Bank Report")
        with gr.Row():
            bank_pie = gr.Plot(label="Aggregate Category Spend")
            bank_points = gr.Plot(label="Total Points by Category")
        with gr.Row():
            bank_monthly = gr.Plot(label="Average Monthly Score")
            bank_hist = gr.Plot(label="Score Distribution")

        with gr.Row():
            aggregate_json = gr.File(label="Aggregate JSON")
            artifact_notes = gr.Markdown()

        def _bank_ingest_wrapper(files, dir_input):
            return bank_ingest(files, dir_input)

        ingest_btn.click(
            fn=_bank_ingest_wrapper,
            inputs=[bank_files, dir_input],
            outputs=[
                bank_report,
                bank_pie,
                bank_points,
                bank_monthly,
                bank_hist,
                aggregate_json,
                artifact_notes,
            ],
        )

if __name__ == "__main__":
    demo.launch(share=True)
