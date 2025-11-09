from __future__ import annotations
import os
import json
import base64
import hashlib
import mimetypes
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional, Union

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

load_dotenv()

# Optional LangSmith
try:
    from langsmith import traceable as _ls_traceable

    _HAS_LANGSMITH = True
except Exception:
    _HAS_LANGSMITH = False

    def _ls_traceable(*args, **kwargs):
        def _decorator(f):
            return f

        return _decorator


# ------------------- TOGGLES / ENV -------------------
def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, str(default)).strip().lower()
    return val in {"1", "true", "yes", "on"}


GOVERNMENT_MODE = _env_bool("GOVERNMENT_MODE", False)
ALLOW_PII_STORAGE = _env_bool("ALLOW_PII_STORAGE", True)
AUDIT_LOG_TO_FILE = _env_bool("AUDIT_LOG_TO_FILE", False)
AUDIT_FILE_PATH = os.getenv("AUDIT_FILE_PATH", "analysis_audit_log.jsonl")

LANGSMITH_ENABLED = _env_bool("LANGSMITH_ENABLED", False)
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "gov-sustainability-chatbot")

if LANGSMITH_ENABLED:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    if "LANGCHAIN_PROJECT" not in os.environ:
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT


def _hash_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _ls_config(
    run_name: str, session_id: str | None = None, tags: List[str] | None = None
) -> Dict[str, Any]:
    if not LANGSMITH_ENABLED:
        return {}
    meta: Dict[str, Any] = {"gov_mode": GOVERNMENT_MODE, "module": "core.py"}
    if session_id:
        meta["session_id"] = _hash_id(session_id) if GOVERNMENT_MODE else session_id
    cfg: Dict[str, Any] = {"run_name": run_name, "metadata": meta}
    if tags:
        cfg["tags"] = tags
    return cfg


# ------------------- MODEL -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_output_tokens=None,
    timeout=60,
    max_retries=3,
)


# ------------------- CONSTANTS -------------------
CATEGORY_WEIGHTS = {
    "public_transport": 95,
    "education": 90,
    "healthcare": 85,
    "groceries": 75,
    "utilities": 55,
    "restaurants": 55,
    "entertainment": 60,
    "travel": 35,
    "online_shopping": 40,
    "shopping": 45,
    "rideshare": 30,
    "gas": 15,
    "other": 50,
}

CARBON_FACTORS = {
    "gas": 400,
    "rideshare": 350,
    "travel": 250,
    "shopping": 150,
    "online_shopping": 155,
    "restaurants": 100,
    "utilities": 80,
    "entertainment": 60,
    "groceries": 50,
    "education": 30,
    "healthcare": 40,
    "public_transport": 40,
    "other": 100,
}

POINTS_POLICY = {
    "version": "1.0.0",
    "base_points_per_dollar": 1.0,
    "weight_influence": 0.6,
    "carbon_penalty_per_kg_per_dollar": 0.4,
    "min_points_per_dollar": 0.05,
    "max_points_per_dollar": 3.0,
    "low_carbon_bonus_threshold": 0.08,
    "low_carbon_bonus_rate": 0.15,
}


# ------------------- SESSION STATE -------------------
_session_counters: Dict[str, Dict[str, Any]] = {}
SESSION_CONTEXT: Dict[str, Dict[str, Any]] = {}

# New: chat history store for LangChain
_history_store: Dict[str, InMemoryChatMessageHistory] = {}


# ------------------- PROMPTS -------------------
def _build_sustainability_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """You are a financial sustainability advisor analyzing credit card spending for environmental impact.

SPENDING ANALYSIS:
{summary_text}

CARBON FOOTPRINT:
{carbon_text}

IMPROVEMENT OPPORTUNITIES:
{opportunities_text}

Provide a comprehensive sustainability report with:
1. Overall Assessment
2. Key Findings
3. Carbon Context
4. Category Breakdown
5. Seasonal Patterns
6. Top 5 Recommendations
7. Quick Wins

Use markdown formatting. Keep under 400 words. Be encouraging but honest.

Your Response:
""".strip()
    )


def _build_chat_prompt_template(system_text: str) -> ChatPromptTemplate:
    """
    A LangChain chat prompt that:
      - Uses a per-session rotating system prompt
      - Injects contextual metrics as a second system message
      - Keeps chat history via MessagesPlaceholder
      - Accepts the latest user input (input)
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            (
                "system",
                "You are a personal financial sustainability advisor. The user has uploaded their spending data.\n"
                "CONTEXT:\n"
                "Sustainability Score: {score}/100\n"
                "Total CO2: {carbon_kg:.2f} kg\n"
                "Monthly Average: ${monthly_avg:.2f}\n"
                "Top Categories: {top_categories}\n"
                "{follow_up_info}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def _build_followup_prompt_three(
    top_grocery_date: str, top_grocery_desc: str, grocery_amount: float
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        f"""You are a personal financial sustainability advisor analyzing high grocery spending.

GROCERY TRANSACTION: {top_grocery_date} - {top_grocery_desc} (${grocery_amount:.2f})

Ask exactly 3 concise, actionable follow-up questions in a single message, numbered 1–3:
1) About items (local/organic vs imported, packaged vs fresh)
2) About habits (store choices, reusable bags, bulk buying)
3) About planning (food waste frequency, meal prep)

Only output the three questions with numbers 1., 2., 3. and nothing else.

Your Response:
""".strip()
    )


# Fixed system prompt variants
PROMPT_VARIANTS = [
    lambda score: f"Your sustainability score is {score:.1f}/100.",
    lambda score: "Here is a quick summary of your sustainability and spending patterns.",
    lambda score: "Consider focusing on top-impact categories to improve your sustainability.",
    lambda score: "Ask about any category or month to get targeted suggestions.",
    lambda score: "You can download your detailed sustainability report as a PDF.",
]
PROMPT_ROTATE_START_IDX = 1


def get_next_system_prompt(session_id: str, score: float) -> str:
    ctx = SESSION_CONTEXT.setdefault(session_id or "default", {"prompt_idx": 0})
    idx = ctx["prompt_idx"]
    if idx == 0:
        prompt = PROMPT_VARIANTS[0](score)
    else:
        idx_cycle = (idx - PROMPT_ROTATE_START_IDX) % (
            len(PROMPT_VARIANTS) - 1
        ) + PROMPT_ROTATE_START_IDX
        prompt = PROMPT_VARIANTS[idx_cycle](score)
    ctx["prompt_idx"] += 1
    return prompt


# ------------------- DATA PREP -------------------
def extract_category_from_description(description: str) -> str:
    description = str(description).lower()
    category_map = {
        "groceries": [
            "grocery",
            "supermarket",
            "market",
            "whole foods",
            "trader joe",
            "costco",
            "loblaws",
        ],
        "restaurants": [
            "restaurant",
            "cafe",
            "coffee",
            "pizza",
            "burger",
            "dining",
            "ubereats",
            "doordash",
            "skiptheidishes",
        ],
        "gas": [
            "shell",
            "chevron",
            "bp",
            "exxon",
            "fuel",
            "petrol",
            "esso",
            "ultramar",
        ],
        "public_transport": [
            "metro",
            "bus",
            "train",
            "transit",
            "subway",
            "tram",
            "ticket",
            "ttc",
            "go transit",
        ],
        "rideshare": ["uber", "lyft", "taxi", "ride", "cab"],
        "shopping": [
            "mall",
            "retail",
            "store",
            "shopping",
            "walmart",
            "costco",
            "target",
        ],
        "online_shopping": ["amazon", "ebay", "etsy", "aliexpress", "shein"],
        "utilities": [
            "electric",
            "water",
            "internet",
            "bill",
            "phone",
            "utility",
            "hydro",
            "enbridge",
        ],
        "entertainment": [
            "movie",
            "cinema",
            "concert",
            "theater",
            "netflix",
            "spotify",
            "gaming",
        ],
        "healthcare": ["pharmacy", "doctor", "clinic", "hospital", "medical", "dental"],
        "education": ["school", "university", "tuition", "course", "training", "udemy"],
        "travel": [
            "hotel",
            "flight",
            "airline",
            "airbnb",
            "booking",
            "travel",
            "marriott",
        ],
    }
    for category, keywords in category_map.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "other"


def get_season(date_str: str) -> str:
    try:
        date_obj = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        m = date_obj.month
        if m in [12, 1, 2]:
            return "Winter"
        if m in [3, 4, 5]:
            return "Spring"
        if m in [6, 7, 8]:
            return "Summer"
        return "Fall"
    except Exception:
        return "Unknown"


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["category"] = df["description"].apply(extract_category_from_description)
    df["season"] = df["date"].apply(get_season)
    df = df[df["debit"] > 0]
    df["month"] = df["date"].dt.to_period("M")
    return df.sort_values("date")


def separate_by_season(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    return {s: df[df["season"] == s].copy() for s in seasons}


# ------------------- ANALYTICS -------------------
def calculate_sustainability_score(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    if "debit" not in df.columns or "category" not in df.columns:
        return 0, {}
    total_spend = df["debit"].sum()
    if total_spend == 0:
        return 0, {}
    df = df.copy()
    df["weight"] = df["category"].map(CATEGORY_WEIGHTS).fillna(50)
    score = (df["debit"] * df["weight"]).sum() / total_spend
    cat_sums = df.groupby("category")["debit"].sum().to_dict()
    return min(score, 100), cat_sums


def calculate_carbon_footprint(df: pd.DataFrame) -> Tuple[float, float]:
    df = df.copy()
    df["carbon_factor"] = df["category"].map(CARBON_FACTORS).fillna(100)
    total_carbon_g = (df["debit"] * df["carbon_factor"]).sum()
    total_carbon_kg = total_carbon_g / 1000.0
    total_spend = df["debit"].sum()
    carbon_per_dollar = total_carbon_kg / total_spend if total_spend > 0 else 0.0
    return total_carbon_kg, carbon_per_dollar


def get_category_sustainability(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    result = {}
    total_spend = df["debit"].sum()
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        spend = cat_df["debit"].sum()
        weight = CATEGORY_WEIGHTS.get(category, 50)
        carbon_factor = CARBON_FACTORS.get(category, 100)
        carbon_kg = (spend * carbon_factor) / 1000.0
        result[category] = {
            "amount": spend,
            "percentage": (spend / total_spend * 100) if total_spend > 0 else 0.0,
            "weight": weight,
            "carbon_kg": carbon_kg,
            "transactions": len(cat_df),
        }
    return dict(sorted(result.items(), key=lambda x: x[1]["amount"], reverse=True))


def get_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    monthly = []
    for month, g in df.groupby("month"):
        score, _ = calculate_sustainability_score(g)
        spend = g["debit"].sum()
        carbon, _ = calculate_carbon_footprint(g)
        monthly.append(
            {
                "month": str(month),
                "score": round(score, 2),
                "spending": spend,
                "carbon_kg": carbon,
                "transactions": len(g),
            }
        )
    return pd.DataFrame(monthly)


def get_seasonal_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for season, sdf in separate_by_season(df).items():
        if len(sdf) == 0:
            continue
        score, _ = calculate_sustainability_score(sdf)
        total_carbon, cpd = calculate_carbon_footprint(sdf)
        result[season] = {
            "score": round(score, 2),
            "spending": sdf["debit"].sum(),
            "carbon_kg": round(total_carbon, 2),
            "carbon_per_dollar": round(cpd, 4),
            "transactions": len(sdf),
            "avg_transaction": round(sdf["debit"].mean(), 2),
        }
    return result


def get_improvement_opportunities(
    df: pd.DataFrame, top_n: int = 5
) -> List[Dict[str, Any]]:
    cat_data = get_category_sustainability(df)
    opp = []
    for cat, data in cat_data.items():
        weight = CATEGORY_WEIGHTS.get(cat, 50)
        if weight < 60:
            opp.append(
                {
                    "category": cat,
                    "current_spend": data["amount"],
                    "current_percentage": data["percentage"],
                    "sustainability_weight": weight,
                    "carbon_kg": data["carbon_kg"],
                    "improvement_potential": 100 - weight,
                }
            )
    return sorted(opp, key=lambda x: x["current_spend"], reverse=True)[:top_n]


def _points_per_dollar_for_category(cat: str) -> float:
    cfg = POINTS_POLICY
    base = cfg["base_points_per_dollar"]
    w_inf = cfg["weight_influence"]
    weight = CATEGORY_WEIGHTS.get(cat, 50) / 100.0
    carbon_g_per_dollar = CARBON_FACTORS.get(cat, 100)
    carbon_kg_per_dollar = carbon_g_per_dollar / 1000.0
    raw = (
        base * ((1.0 - w_inf) + w_inf * weight)
        - cfg["carbon_penalty_per_kg_per_dollar"] * carbon_kg_per_dollar
    )
    if carbon_kg_per_dollar <= cfg["low_carbon_bonus_threshold"]:
        raw *= 1.0 + cfg["low_carbon_bonus_rate"]
    return max(cfg["min_points_per_dollar"], min(cfg["max_points_per_dollar"], raw))


@_ls_traceable(name="calculate_sustainability_points")
def calculate_sustainability_points(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    if df.empty:
        return 0.0, {
            "points_by_category": {},
            "points_per_dollar_by_category": {},
            "policy_version": POINTS_POLICY["version"],
        }
    td = df.copy()
    td["ppd"] = td["category"].apply(_points_per_dollar_for_category)
    td["points"] = td["debit"] * td["ppd"]
    points_by_category = td.groupby("category")["points"].sum().to_dict()
    ppd_by_category = td.groupby("category")["ppd"].mean().to_dict()
    total_points = float(td["points"].sum())
    details = {
        "points_by_category": dict(
            sorted(points_by_category.items(), key=lambda x: x[1], reverse=True)
        ),
        "points_per_dollar_by_category": dict(
            sorted(ppd_by_category.items(), key=lambda x: x[1], reverse=True)
        ),
        "policy_version": POINTS_POLICY["version"],
    }
    return total_points, details


# ------------------- LLM INSIGHTS -------------------
@_ls_traceable(name="generate_sustainability_insights")
def generate_sustainability_insights(analysis_result: Dict[str, Any]) -> str:
    df = analysis_result["df"]
    score = analysis_result["score"]
    category_data = get_category_sustainability(df)
    seasonal_data = get_seasonal_analysis(df)
    opportunities = get_improvement_opportunities(df)
    carbon_total, carbon_per_dollar = calculate_carbon_footprint(df)

    summary_text = (
        f"**Overall Score:** {score:.1f}/100\n"
        f"**Total Spending:** ${df['debit'].sum():.2f}\n"
        f"**Number of Transactions:** {len(df)}\n"
        f"**Average Transaction:** ${df['debit'].mean():.2f}\n\n"
        "**Top Spending Categories:**\n"
        + "".join(
            f"- {cat.title()}: ${data['amount']:.2f} ({data['percentage']:.1f}%)\n"
            for cat, data in list(category_data.items())[:5]
        )
    )
    carbon_text = (
        f"**Total CO2 Equivalent:** {carbon_total:.2f} kg\n"
        f"**Carbon Intensity:** {carbon_per_dollar:.4f} kg CO2/$\n"
        f"**Trees Needed to Offset:** ~{int(carbon_total / 20)}\n"
        "**Seasonal Breakdown:**\n"
        + "".join(
            f"- {season}: {data['carbon_kg']:.2f} kg CO2\n"
            for season, data in seasonal_data.items()
        )
    )
    opportunities_text = "**Top Improvement Areas:**\n" + "".join(
        f"- {opp['category'].title()}: {opp['improvement_potential']:.0f}% more sustainable alternative available\n"
        for opp in opportunities
    )

    prompt = _build_sustainability_prompt()
    msgs = prompt.format_messages(
        summary_text=summary_text,
        carbon_text=carbon_text,
        opportunities_text=opportunities_text,
    )
    try:
        response = llm.invoke(
            msgs, config=_ls_config(run_name="insights_llm", tags=["insights"])
        ).content
    except Exception as e:
        response = f"Error generating insights: {str(e)}"
    return response


# ------------------- CHAT HISTORY HELPERS -------------------
def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    sid = session_id or "default"
    if sid not in _history_store:
        _history_store[sid] = InMemoryChatMessageHistory()
    return _history_store[sid]


def _detect_mime_type(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime or not mime.startswith("image/"):
        return "image/jpeg"
    return mime


def _encode_image_path_to_data_url(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        mime = _detect_mime_type(image_path)
        return f"data:{mime};base64,{img_b64}"
    except Exception:
        return None


def _encode_image_bytes_to_data_url(
    image_bytes: bytes, mime_hint: str = "image/png"
) -> str:
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_hint};base64,{img_b64}"


def _image_blocks(
    image_paths: Optional[List[str]] = None,
    image_bytes_list: Optional[List[bytes]] = None,
    mime_hint: str = "image/png",
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if image_paths:
        for p in image_paths:
            if p and os.path.exists(p):
                data_url = _encode_image_path_to_data_url(p)
                if data_url:
                    blocks.append({"type": "image_url", "image_url": {"url": data_url}})
    if image_bytes_list:
        for b in image_bytes_list:
            if b:
                data_url = _encode_image_bytes_to_data_url(b, mime_hint=mime_hint)
                blocks.append({"type": "image_url", "image_url": {"url": data_url}})
    return blocks


def _initialize_session_counters(session_id: str):
    if session_id not in _session_counters:
        _session_counters[session_id] = {
            "total_chats": 0,
            "followup_count": 0,
            "followup_active": False,
            "followup_target_date": None,
            "followup_target_desc": None,
            "followup_target_amount": 0.0,
            "followup_once_done": False,
        }


def _should_ask_followup(session_id: str, question: str) -> bool:
    if session_id not in _session_counters:
        return False
    session = _session_counters[session_id]
    if session.get("followup_once_done"):
        return False
    total_chats = session["total_chats"]
    followup_count = session["followup_count"]
    if total_chats >= 10:
        session["total_chats"] = 0
        session["followup_count"] = 0
        session["followup_active"] = False
        session["followup_once_done"] = False
        return False
    if followup_count >= 3:
        return False
    q = question.lower()
    rel = ["grocery", "bill", "receipt", "spending", "purchase", "shopping", "cost"]
    if not any(k in q for k in rel):
        return False
    if any(
        p in q
        for p in ["don't have", "no receipt", "can't remember", "not sure", "unknown"]
    ):
        session["followup_active"] = False
        return False
    return True


def _build_langchain_chat_runnable(
    session_id: str,
    score: float,
    carbon_kg: float,
    monthly_avg: float,
    top_categories: str,
    follow_up_info: str,
) -> RunnableWithMessageHistory:
    system_text = get_next_system_prompt(session_id or "default", score)
    prompt = _build_chat_prompt_template(system_text)
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=lambda sid: _get_session_history(sid),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    # Attach constant fields to avoid passing them every time
    # The chain expects these variables, so we’ll partially apply via assign.
    # In LC Core, you can .bind() on prompt variables, but here we keep invocation clear.
    # Return the runnable; the caller will pass variables.
    return chain_with_history


# ------------------- CHAT ANSWER -------------------
@_ls_traceable(name="answer_sustainability_question")
def answer_sustainability_question(
    question: str,
    analysis_result: Dict[str, Any],
    image_path: Optional[str] = None,
    session_id: str = None,
    image_paths: Optional[List[str]] = None,
    image_bytes_list: Optional[List[bytes]] = None,
    image_mime_hint: str = "image/png",
) -> str:
    """
    Enhanced to:
      - Use LangChain chat history per session
      - Keep first system prompt unique, rotate later prompts
      - Support multimodal image attachments (paths or bytes)
      - Preserve follow-up logic and store these turns in history
    """
    df = analysis_result["df"]
    score = analysis_result["score"]
    carbon_total, _ = calculate_carbon_footprint(df)
    category_data = get_category_sustainability(df)

    top_cats = ", ".join(
        [
            f"{cat.title()} (${data['amount']:.0f})"
            for cat, data in list(category_data.items())[:3]
        ]
    )
    monthly_avg = df.groupby("month")["debit"].sum().mean() if not df.empty else 0.0

    if session_id:
        _initialize_session_counters(session_id)
        _session_counters[session_id]["total_chats"] += 1

    grocery_df = df[df["category"] == "groceries"]
    top_grocery_date = None
    top_grocery_desc = None
    grocery_amount = 0.0
    follow_up_info = ""

    if not grocery_df.empty and grocery_df["debit"].sum() > 100:
        top_idx = grocery_df["debit"].idxmax()
        top_grocery_date = df.loc[top_idx, "date"].strftime("%Y-%m-%d")
        top_grocery_desc = df.loc[top_idx, "description"]
        grocery_amount = float(grocery_df["debit"].max())

    should_followup = False
    if session_id and _should_ask_followup(session_id, question):
        should_followup = True
        s = _session_counters[session_id]
        s["followup_active"] = True
        s["followup_count"] += 1
        s["followup_target_date"] = top_grocery_date
        s["followup_target_desc"] = top_grocery_desc
        s["followup_target_amount"] = grocery_amount

    # If follow-up is triggered, generate the 3 numbered questions and record them in history
    if should_followup and top_grocery_date:
        followup_prompt = _build_followup_prompt_three(
            top_grocery_date, top_grocery_desc, grocery_amount
        )
        followup_text = followup_prompt.format()
        response = llm.invoke(
            followup_text,
            config=_ls_config(
                run_name="followup_llm", session_id=session_id, tags=["followup"]
            ),
        ).content
        context_summary = (
            f"Sustainability Score: {score:.1f}/100 | Top Categories: {top_cats}"
        )
        final = f"{context_summary}\n\n{response}"

        # Persist this exchange into history
        if session_id:
            hist = _get_session_history(session_id)
            hist.add_message(HumanMessage(content=question))
            hist.add_message(AIMessage(content=final))

            _session_counters[session_id]["followup_active"] = True
            _session_counters[session_id]["followup_once_done"] = True
        return final
    else:
        if top_grocery_date and _session_counters.get(session_id, {}).get(
            "followup_active", False
        ):
            follow_up_info = f"Note: High grocery spending detected on {top_grocery_date} (${grocery_amount:.2f}). Upload receipt for detailed analysis."
        else:
            follow_up_info = ""

        try:
            # If any images are present, handle multimodal path with manual history replay
            has_any_images = bool(
                image_path
                or (image_paths and len(image_paths) > 0)
                or (image_bytes_list and len(image_bytes_list) > 0)
            )
            if has_any_images:
                # Consolidate images
                consolidated_paths = list(image_paths or [])
                if image_path:
                    consolidated_paths.append(image_path)

                # Build messages: system + replayed history + new human with text+images
                system_text = get_next_system_prompt(session_id or "default", score)
                hist = _get_session_history(session_id or "default")
                # Compose human text with context block similar to the template
                context_block = (
                    "You are a personal financial sustainability advisor. The user has uploaded their spending data.\n"
                    f"CONTEXT:\nSustainability Score: {score}/100\n"
                    f"Total CO2: {carbon_total:.2f} kg\n"
                    f"Monthly Average: ${monthly_avg:.2f}\n"
                    f"Top Categories: {top_cats}\n"
                    f"{follow_up_info}"
                )
                user_text = f"{context_block}\n\n{question}"

                content_blocks = [{"type": "text", "text": user_text}]
                content_blocks.extend(
                    _image_blocks(
                        image_paths=consolidated_paths,
                        image_bytes_list=image_bytes_list,
                        mime_hint=image_mime_hint,
                    )
                )

                message_list = [SystemMessage(content=system_text)]
                # Replay past history
                for m in hist.messages:
                    message_list.append(m)
                # Add new human with text + images
                human_msg = HumanMessage(content=content_blocks)
                message_list.append(human_msg)

                response_msg = llm.invoke(
                    message_list,
                    config=_ls_config(
                        run_name="chat_with_image_llm",
                        session_id=session_id,
                        tags=["chat", "image"],
                    ),
                )
                response = response_msg.content

                # Append to history
                hist.add_message(human_msg)
                hist.add_message(AIMessage(content=response))

                # Optional storage of image base64 if allowed (store only first for brevity)
                if ALLOW_PII_STORAGE:
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "date": top_grocery_date
                        if top_grocery_date
                        else datetime.now().strftime("%Y-%m-%d"),
                        "question": question,
                        "response": response,
                        "transaction_desc": top_grocery_desc,
                        "gov_mode": GOVERNMENT_MODE,
                    }
                    # DO NOT store raw images unless explicitly required; we keep metadata only.
                    fn = "grocery.json"
                    data_list = []
                    if os.path.exists(fn):
                        try:
                            with open(fn, "r") as rf:
                                data_list = json.load(rf)
                        except Exception:
                            data_list = []
                    data_list.append(entry)
                    with open(fn, "w") as wf:
                        json.dump(data_list, wf, indent=4)
                return response
            else:
                # Text-only path via RunnableWithMessageHistory
                chat_runnable = _build_langchain_chat_runnable(
                    session_id or "default",
                    score=score,
                    carbon_kg=carbon_total,
                    monthly_avg=monthly_avg,
                    top_categories=top_cats,
                    follow_up_info=follow_up_info,
                )
                result_msg = chat_runnable.invoke(
                    {
                        "score": f"{score:.1f}",
                        "carbon_kg": carbon_total,
                        "monthly_avg": monthly_avg,
                        "top_categories": top_cats,
                        "follow_up_info": follow_up_info,
                        "input": question,
                    },
                    config={"configurable": {"session_id": session_id or "default"}},
                )
                # result_msg is an AIMessage; return its content
                return getattr(result_msg, "content", str(result_msg))
        except Exception as e:
            return f"I encountered an error: {str(e)}"


# ------------------- VISUALIZATION DATA -------------------
def generate_pie_data(df: pd.DataFrame) -> Dict[str, Any]:
    cat = df.groupby("category")["debit"].sum().sort_values(ascending=False)
    return {
        "labels": cat.index.tolist(),
        "values": cat.values.tolist(),
        "title": "Spending Distribution by Category",
    }


def generate_bar_data(df: pd.DataFrame) -> Dict[str, Any]:
    order = ["Winter", "Spring", "Summer", "Fall"]
    s = df.groupby("season")["debit"].sum()
    s = s.reindex([x for x in order if x in s.index])
    return {
        "x": s.index.tolist(),
        "y": s.values.tolist(),
        "title": "Spending by Season",
    }


def generate_line_data(df: pd.DataFrame) -> Dict[str, Any]:
    m = get_monthly_trends(df)
    return {
        "x": m["month"].tolist(),
        "y": m["score"].tolist(),
        "title": "Sustainability Score Trend",
    }


def generate_carbon_data(df: pd.DataFrame) -> Dict[str, Any]:
    cat = get_category_sustainability(df)
    keys = list(cat.keys())
    vals = [cat[k]["carbon_kg"] for k in keys]
    return {"x": keys, "y": vals, "title": "Carbon Footprint by Category (kg CO2)"}


# ------------------- CONCISE SUMMARY -------------------
def generate_concise_summary(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    df = analysis_result["df"]
    score = analysis_result["score"]
    carbon_total = analysis_result["carbon_total"]
    points_total = analysis_result["points_total"]
    category_data = analysis_result["category_data"]

    if score >= 75:
        rating = "Excellent"
    elif score >= 60:
        rating = "Good"
    elif score >= 45:
        rating = "Fair"
    else:
        rating = "Needs Improvement"

    summary_text = (
        f"Your sustainability score is {score:.1f}/100 ({rating}). "
        f"You earned {points_total:,.0f} points on ${df['debit'].sum():,.2f} spend. "
        f"Estimated footprint: {carbon_total:.2f} kg CO2 (~{int(carbon_total / 20)} trees)."
    )

    top_rows = []
    for cat, data in list(category_data.items())[:5]:
        top_rows.append(
            {
                "category": cat.title(),
                "amount": f"${data['amount']:,.2f}",
                "percentage": f"{data['percentage']:.1f}%",
                "carbon_kg": f"{data['carbon_kg']:.2f}",
            }
        )

    return {
        "summary_text": summary_text,
        "top_categories": top_rows,
        "key_metrics": {
            "score": f"{score:.1f}",
            "total_spending": f"${df['debit'].sum():,.2f}",
            "carbon_total": f"{carbon_total:.2f} kg",
            "points_total": f"{points_total:,.0f}",
            "transactions": len(df),
        },
    }


# ------------------- PDF REPORT -------------------
def generate_pdf_report(
    analysis_result: Dict[str, Any], output_path: str = None
) -> str:
    if output_path is None:
        output_path = (
            f"sustainability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=1.0 * inch,
        bottomMargin=0.75 * inch,
    )
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1a5490"),
        alignment=1,
        spaceAfter=24,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2c5f8d"),
        spaceBefore=12,
        spaceAfter=8,
    )

    df = analysis_result["df"]
    score = analysis_result["score"]
    carbon_total = analysis_result["carbon_total"]
    carbon_per_dollar = analysis_result["carbon_per_dollar"]
    category_data = analysis_result["category_data"]
    seasonal_data = analysis_result["seasonal_data"]
    monthly_trends = analysis_result["monthly_trends"]
    opportunities = analysis_result["opportunities"]
    points_total = analysis_result["points_total"]
    insights = analysis_result.get("insights", "")

    story.append(Paragraph("Sustainability Analysis Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", section_style))
    summary_data = [
        ["Metric", "Value"],
        ["Sustainability Score", f"{score:.1f}/100"],
        ["Total Spending", f"${df['debit'].sum():,.2f}"],
        ["Transactions", f"{len(df):,}"],
        ["Carbon Footprint", f"{carbon_total:.2f} kg CO2"],
        ["Carbon Intensity", f"{carbon_per_dollar:.4f} kg/$"],
        ["Sustainability Points", f"{points_total:,.0f}"],
        ["Trees to Offset", f"~{int(carbon_total / 20)}"],
    ]
    t = Table(summary_data, colWidths=[3 * inch, 3 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5f8d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    # Category Breakdown
    story.append(Paragraph("Category Breakdown", section_style))
    cat_rows = [["Category", "Amount", "% of Total", "Carbon (kg)", "Txns"]]
    for cat, data in category_data.items():
        cat_rows.append(
            [
                cat.title(),
                f"${data['amount']:,.2f}",
                f"{data['percentage']:.1f}%",
                f"{data['carbon_kg']:.2f}",
                f"{data['transactions']}",
            ]
        )
    t2 = Table(
        cat_rows, colWidths=[2 * inch, 1.2 * inch, 1 * inch, 1.1 * inch, 0.7 * inch]
    )
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5f8d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 0.2 * inch))

    # Seasonal Analysis
    story.append(Paragraph("Seasonal Analysis", section_style))
    season_rows = [["Season", "Score", "Spending", "Carbon (kg)", "Txns"]]
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        if season in seasonal_data:
            d = seasonal_data[season]
            season_rows.append(
                [
                    season,
                    f"{d['score']:.1f}",
                    f"${d['spending']:,.2f}",
                    f"{d['carbon_kg']:.2f}",
                    f"{d['transactions']}",
                ]
            )
    t3 = Table(
        season_rows,
        colWidths=[1.5 * inch, 1 * inch, 1.5 * inch, 1.3 * inch, 0.7 * inch],
    )
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5f8d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )
    story.append(t3)
    story.append(Spacer(1, 0.2 * inch))

    # Monthly Trends
    story.append(Paragraph("Monthly Trends", section_style))
    m_rows = [["Month", "Score", "Spending", "Carbon (kg)", "Txns"]]
    for _, r in monthly_trends.iterrows():
        m_rows.append(
            [
                r["month"],
                f"{r['score']:.1f}",
                f"${r['spending']:,.2f}",
                f"{r['carbon_kg']:.2f}",
                f"{r['transactions']}",
            ]
        )
    t4 = Table(
        m_rows, colWidths=[1.5 * inch, 1 * inch, 1.5 * inch, 1.3 * inch, 0.7 * inch]
    )
    t4.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5f8d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )
    story.append(t4)
    story.append(PageBreak())

    # Opportunities
    story.append(Paragraph("Improvement Opportunities", section_style))
    opp_rows = [["Category", "Current Spend", "% of Total", "Potential Gain"]]
    for o in opportunities:
        opp_rows.append(
            [
                o["category"].title(),
                f"${o['current_spend']:,.2f}",
                f"{o['current_percentage']:.1f}%",
                f"{o['improvement_potential']:.0f}%",
            ]
        )
    if len(opp_rows) == 1:
        opp_rows.append(["—", "—", "—", "—"])
    t5 = Table(opp_rows, colWidths=[2 * inch, 1.5 * inch, 1.2 * inch, 1.2 * inch])
    t5.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5f8d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )
    story.append(t5)
    story.append(Spacer(1, 0.2 * inch))

    # Insights
    story.append(Paragraph("AI-Generated Insights", section_style))
    cleaned = insights.replace("**", "").replace("##", "").replace("#", "")
    for line in cleaned.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))

    doc.build(story)
    return output_path


# ------------------- MAIN PIPELINE -------------------
def _append_audit(entry: Dict[str, Any], session_id: str | None = None) -> None:
    if not AUDIT_LOG_TO_FILE:
        return
    safe_entry = dict(entry)
    if session_id:
        safe_entry["session_hash"] = (
            _hash_id(session_id) if GOVERNMENT_MODE else session_id
        )
    safe_entry["ts"] = datetime.now().isoformat()
    with open(AUDIT_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_entry, ensure_ascii=False) + "\n")


@_ls_traceable(name="run_full_analysis")
def run_full_analysis(
    df: pd.DataFrame, *, session_id: str | None = None, generate_pdf: bool = True
) -> Dict[str, Any]:
    df = preprocess_dataset(df)
    score, _ = calculate_sustainability_score(df)
    carbon_total, carbon_per_dollar = calculate_carbon_footprint(df)
    category_data = get_category_sustainability(df)
    seasonal_data = get_seasonal_analysis(df)
    monthly_trends = get_monthly_trends(df)
    points_total, points_details = calculate_sustainability_points(df)

    result = {
        "df": df,
        "score": round(score, 2),
        "carbon_total": round(carbon_total, 2),
        "carbon_per_dollar": round(carbon_per_dollar, 4),
        "category_data": category_data,
        "seasonal_data": seasonal_data,
        "monthly_trends": monthly_trends,
        "total_spending": float(df["debit"].sum()),
        "transaction_count": int(len(df)),
        "pie_data": generate_pie_data(df),
        "bar_data": generate_bar_data(df),
        "line_data": generate_line_data(df),
        "carbon_data": generate_carbon_data(df),
        "opportunities": get_improvement_opportunities(df),
        "points_total": round(points_total, 2),
        "points_details": points_details,
        "points_policy_version": POINTS_POLICY["version"],
    }

    result["insights"] = generate_sustainability_insights(result)
    result["concise_summary"] = generate_concise_summary(result)
    if generate_pdf:
        result["pdf_path"] = generate_pdf_report(result)

    _append_audit(
        {
            "summary": {
                "score": result["score"],
                "carbon_total": result["carbon_total"],
                "carbon_per_dollar": result["carbon_per_dollar"],
                "points_total": result["points_total"],
                "policy_version": result["points_policy_version"],
            },
            "top_categories": list(category_data.keys())[:5],
        },
        session_id=session_id,
    )
    return result


# ------------------- PUBLIC HELPERS -------------------
def get_system_prompt_for_session(
    session_id: str, analysis_result: Dict[str, Any]
) -> str:
    """
    Returns the next system prompt per the requirement:
    - First time per session: only the sustainability score.
    - Subsequent times: rotate through different fixed prompts.
    """
    score = float(analysis_result.get("score", 0.0))
    return get_next_system_prompt(session_id or "default", score)
