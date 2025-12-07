# nl2sql.py
"""
Improved Natural Language -> SQL translator with sandboxed SQLite execution.

Usage:
- Place in your Streamlit app directory.
- Must be wired as: from nl2sql import render_nl2sql
- Call render_nl2sql() from your router.

Notes:
- Heuristic-based, single-table focused.
- Use 'Run in sandbox' to execute generated SQL against the selected DataFrame (in-memory SQLite).
"""

from typing import List, Dict, Tuple, Optional
import streamlit as st
import pandas as pd
import re
import sqlite3
import io
from difflib import SequenceMatcher

# ---------- Helpers ----------

def _detect_tables_in_session() -> Dict[str, pd.DataFrame]:
    tables = {}
    for k, v in st.session_state.items():
        if isinstance(v, pd.DataFrame):
            tables[k] = v
    # common explicit keys
    for key in ("__uploaded_df__", "__cleaned_df__"):
        if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
            tables.setdefault(key, st.session_state[key])
    return tables

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower() or "")

def _best_column_match(token: str, columns: List[str], threshold: float = 0.55) -> Optional[str]:
    """
    Fuzzy match token -> best column using SequenceMatcher. Return None if below threshold.
    """
    if not token or not columns:
        return None
    token_norm = _norm(token)
    best = None
    best_score = 0.0
    for c in columns:
        score = SequenceMatcher(None, token_norm, _norm(c)).ratio()
        # boost exact last-segment matches (e.g., 'income' -> 'applicant_income')
        last = c.split("_")[-1].lower()
        if token.lower() == last:
            score = max(score, 0.95)
        if score > best_score:
            best_score = score; best = c
    return best if best_score >= threshold else None

def _escape_ident(name: str) -> str:
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
        return name
    return '"' + name.replace('"', '""') + '"'

# ---------- Extractors ----------

AGG_MAP = {
    "count": "COUNT",
    "number of": "COUNT",
    "how many": "COUNT",
    "sum": "SUM",
    "total": "SUM",
    "average": "AVG",
    "avg": "AVG",
    "mean": "AVG",
    "min": "MIN",
    "maximum": "MAX",
    "max": "MAX"
}

CMP_PHRASES = {
    "greater than": ">",
    "more than": ">",
    "less than": "<",
    "fewer than": "<",
    "at least": ">=",
    "at most": "<="
}

def _extract_limit(text: str) -> Optional[int]:
    m = re.search(r'\btop\s+(\d{1,5})\b', text, flags=re.I)
    if m: return int(m.group(1))
    m = re.search(r'\blimit\s+(\d{1,5})\b', text, flags=re.I)
    if m: return int(m.group(1))
    return None

def _extract_select_columns(text: str, columns: List[str]) -> List[str]:
    """
    Try to parse explicit SELECT targets from phrases: 'show X, Y', 'select A and B', or detect column tokens.
    """
    out: List[str] = []
    # 1) explicit patterns
    m = re.search(r'(?:select|show|display|list)\s+([A-Za-z0-9_,\s]+?)(?:\s+where|\s+group|\s+order|\s+limit|$)', text, flags=re.I)
    if m:
        raw = m.group(1)
        tokens = re.split(r',|\band\b|\bthen\b', raw)
        for t in tokens:
            t = t.strip()
            if not t: continue
            match = _best_column_match(t, columns)
            if match and match not in out:
                out.append(match)
    # 2) explicit comma-separated column names (with underscores)
    if not out:
        # find tokens that look like column names (have underscore or match column words)
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]+', text)
        for tok in tokens:
            mcol = _best_column_match(tok, columns)
            if mcol and mcol not in out:
                out.append(mcol)
    # 3) fallback to *
    if not out:
        return ["*"]
    return out

def _extract_aggregations(text: str, columns: List[str]) -> List[Tuple[str, Optional[str]]]:
    """
    Return list of (AGG_FUNC, column_or_None) detected. Supports multiple aggregations.
    """
    found = []
    lower = text.lower()
    for phrase, agg in AGG_MAP.items():
        if phrase in lower:
            # try to find a nearby column name after the phrase
            idx = lower.find(phrase)
            window = lower[idx: idx+80]
            tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]+', window)
            agg_col = None
            for t in tokens[1:]:
                c = _best_column_match(t, columns)
                if c:
                    agg_col = c
                    break
            found.append((agg, agg_col))
    return found

def _split_conditions(text: str) -> List[Tuple[str, str]]:
    """
    Split textual WHERE into list of (cond_text, glue) where glue is 'AND'/'OR' or ''.
    We simply split by ' and ' / ' or ' at top-level.
    """
    # naive splitting preserving conjunction
    parts = re.split(r'\s+(and|or)\s+', text, flags=re.I)
    out = []
    if len(parts) == 1:
        out.append((parts[0].strip(), ""))
    else:
        # parts will be like [cond1, glue1, cond2, glue2, cond3 ...]
        i = 0
        while i < len(parts):
            cond = parts[i].strip()
            glue = ""
            if i+1 < len(parts):
                glue = parts[i+1].strip().upper()
            out.append((cond, glue))
            i += 2
    return out

def _parse_single_condition(cond: str, columns: List[str]) -> Optional[str]:
    """
    Parse one condition snippet into SQL condition, using operators or phrases.
    """
    low = cond.lower()

    # BETWEEN
    m = re.search(r'([A-Za-z0-9_]+)\s+between\s+([0-9\.\-]+)\s+and\s+([0-9\.\-]+)', low)
    if m:
        col = _best_column_match(m.group(1), columns)
        if col:
            return f"{_escape_ident(col)} BETWEEN {m.group(2)} AND {m.group(3)}"

    # IS NULL / NOT NULL
    m = re.search(r'([A-Za-z0-9_]+)\s+is\s+not\s+null', low)
    if m:
        col = _best_column_match(m.group(1), columns)
        if col: return f"{_escape_ident(col)} IS NOT NULL"
    m = re.search(r'([A-Za-z0-9_]+)\s+is\s+null', low)
    if m:
        col = _best_column_match(m.group(1), columns)
        if col: return f"{_escape_ident(col)} IS NULL"

    # IN list (comma-separated or 'in A, B and C')
    m = re.search(r'([A-Za-z0-9_]+)\s+(?:in|is one of|from)\s+([A-Za-z0-9_\s,\'"-]+)', cond, flags=re.I)
    if m:
        col = _best_column_match(m.group(1), columns)
        if col:
            vals_raw = m.group(2)
            vals = re.split(r',|\band\b', vals_raw)
            vals = [v.strip(" '\"") for v in vals if v.strip()]
            if vals:
                vals_esc = ", ".join([f"'{v}'" for v in vals])
                return f"{_escape_ident(col)} IN ({vals_esc})"

    # comparison with symbol
    m = re.search(r'([A-Za-z0-9_]+)\s*(>=|<=|=|>|<)\s*([0-9\.\-]+)', cond)
    if m:
        col = _best_column_match(m.group(1), columns)
        if col:
            return f"{_escape_ident(col)} {m.group(2)} {m.group(3)}"

    # phrase comparisons: 'greater than X', 'less than X'
    for phrase, op in CMP_PHRASES.items():
        if phrase in low:
            m = re.search(r'([A-Za-z0-9_]+)\s+' + re.escape(phrase) + r'\s+([0-9\.\-]+)', low)
            if m:
                col = _best_column_match(m.group(1), columns)
                if col:
                    return f"{_escape_ident(col)} {op} {m.group(2)}"

    # textual equality: "status is approved" or "status = approved"
    m = re.search(r'([A-Za-z0-9_]+)\s+(?:is|=|equals)\s+([A-Za-z0-9_\- ]+)', cond, flags=re.I)
    if m:
        col = _best_column_match(m.group(1), columns)
        if col:
            val = m.group(2).strip().strip("'\"")
            if val and not re.search(r'\b(top|by|group|order|limit)\b', val):
                return f"{_escape_ident(col)} = '{val}'"

    return None

def _extract_where_clause(text: str, columns: List[str]) -> Optional[str]:
    """
    Locate likely WHERE clause fragments (words after 'where' or sentences containing comparisons).
    Return combined SQL WHERE with AND/OR or None.
    """
    # if explicit WHERE
    m = re.search(r'\bwhere\b\s+(.*)', text, flags=re.I)
    target = None
    if m:
        target = m.group(1)
    else:
        # try to find fragments with comparison verbs or numbers
        if re.search(r'\b(in|between|greater than|less than|=|>|<|is not null|is null)\b', text, flags=re.I):
            # heuristically take the clause after verbs like 'where' synonyms: 'for', 'with', 'where'
            m2 = re.search(r'(?:where|for|with)\s+(.*)', text, flags=re.I)
            target = m2.group(1) if m2 else text
    if not target:
        return None

    # cut off trailing phrases that are not conditions: before 'group by', 'order by', 'limit'
    target = re.split(r'\b(group\s+by|order\s+by|limit|top)\b', target, flags=re.I)[0].strip()

    # split by AND / OR
    parts = _split_conditions(target)
    cond_sqls = []
    for idx, (cond_text, glue) in enumerate(parts):
        parsed = _parse_single_condition(cond_text, columns)
        if parsed:
            cond_sqls.append((parsed, glue))
    if not cond_sqls:
        return None
    # reconstruct with glue tokens
    sql = cond_sqls[0][0]
    for i in range(1, len(cond_sqls)):
        glue = cond_sqls[i-1][1] or "AND"
        sql = f"({sql}) {glue} ({cond_sqls[i][0]})"
    return sql

def _extract_group_by(text: str, columns: List[str]) -> List[str]:
    m = re.search(r'group\s+by\s+([A-Za-z0-9_,\s]+)', text, flags=re.I)
    if m:
        parts = re.split(r',|\band\b|\b&\b', m.group(1))
        out = []
        for p in parts:
            p = p.strip()
            if not p: continue
            col = _best_column_match(p, columns)
            if col and col not in out: out.append(col)
        return out
    # fallback: "by X" after agg phrase
    m2 = re.search(r'\b(?:count|sum|avg|average|min|max|total)\s+by\s+([A-Za-z0-9_\s]+)', text, flags=re.I)
    if m2:
        p = m2.group(1).strip().split()[0:3]
        col = _best_column_match(" ".join(p), columns)
        return [col] if col else []
    return []

def _extract_order_by(text: str, columns: List[str]) -> Tuple[List[str], Optional[str]]:
    m = re.search(r'order\s+by\s+([A-Za-z0-9_,\s]+)(?:\s+(asc|desc))?', text, flags=re.I)
    dir = None
    cols = []
    if m:
        tokens = re.split(r',|\band\b', m.group(1))
        for t in tokens:
            t = t.strip()
            c = _best_column_match(t, columns)
            if c and c not in cols: cols.append(c)
        if m.group(2): dir = m.group(2).upper()
        return cols, dir

    # heuristic: "top N by amount" or "highest amount"
    if re.search(r'\b(top|highest|largest)\b', text, flags=re.I):
        m2 = re.search(r'by\s+([A-Za-z0-9_]+)', text, flags=re.I)
        if m2:
            c = _best_column_match(m2.group(1), columns)
            if c: return [c], "DESC"
        # fallback numeric candidate
        for cand in columns:
            if re.search(r'(amount|income|total|price|value|score|salary|amount)', cand, flags=re.I):
                return [cand], "DESC"
    return [], None

# ---------- Main translator ----------

def translate_prompt_to_sql(prompt: str, table_name: str, columns: List[str]) -> Tuple[str, str]:
    prompt = (prompt or "").strip()
    if not prompt:
        return "-- no input", "No prompt provided."

    explanation = []
    text = prompt

    # SELECT columns
    sel_cols = _extract_select_columns(text, columns)

    # Aggregations
    aggs = _extract_aggregations(text, columns)  # list of (AGG, col_opt)
    if aggs:
        explanation.append("Detected aggregations: " + ", ".join([f"{a}({c or '*'})" for a,c in aggs]))

    # WHERE
    where = _extract_where_clause(text, columns)
    if where:
        explanation.append(f"Filter: {where}")

    # GROUP BY
    group = _extract_group_by(text, columns)
    if group:
        explanation.append("Group by: " + ", ".join(group))

    # ORDER BY
    order_cols, order_dir = _extract_order_by(text, columns)
    if order_cols:
        explanation.append("Order by: " + ", ".join(order_cols) + (f" {order_dir}" if order_dir else ""))

    # LIMIT
    limit = _extract_limit(text)
    if limit:
        explanation.append(f"Limit: {limit}")

    # Build SQL
    parts = []
    if aggs:
        # support multiple aggregations: SELECT group_cols..., agg_expr AS alias, ...
        agg_exprs = []
        for agg_func, agg_col in aggs:
            col = agg_col or (columns[0] if columns else "*")
            alias = f"{agg_func.lower()}_{col if col else 'col'}"
            agg_exprs.append(f"{agg_func}({_escape_ident(col)}) AS {alias}")
        if group:
            selects = ", ".join([_escape_ident(c) for c in group] + agg_exprs)
        else:
            selects = ", ".join(agg_exprs)
        parts.append(f"SELECT {selects}")
        parts.append(f"FROM {_escape_ident(table_name)}")
        if where: parts.append(f"WHERE {where}")
        if group: parts.append("GROUP BY " + ", ".join([_escape_ident(c) for c in group]))
        if order_cols: parts.append("ORDER BY " + ", ".join([_escape_ident(c) for c in order_cols]) + (f" {order_dir}" if order_dir else ""))
        if limit: parts.append(f"LIMIT {limit}")
    else:
        # normal select
        sel_list = []
        if sel_cols == ["*"]:
            sel_list = ["*"]
        else:
            sel_list = [_escape_ident(c) for c in sel_cols]
        parts.append("SELECT " + ", ".join(sel_list))
        parts.append(f"FROM {_escape_ident(table_name)}")
        if where: parts.append(f"WHERE {where}")
        if group: parts.append("GROUP BY " + ", ".join([_escape_ident(c) for c in group]))
        if order_cols: parts.append("ORDER BY " + ", ".join([_escape_ident(c) for c in order_cols]) + (f" {order_dir}" if order_dir else ""))
        if limit: parts.append(f"LIMIT {limit}")

    sql = "\n".join(parts).rstrip() + ";"
    expl = " | ".join(explanation) if explanation else "Translated using heuristic rules."
    return sql, expl

# ---------- Sandbox executor (safe) ----------

def _run_sql_in_memory(df: pd.DataFrame, table_name: str, sql: str, max_rows: int = 1000):
    """
    Create an in-memory SQLite DB, write df to table_name, execute sql, return result DataFrame.
    """
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        cur = conn.cursor()
        # execute
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows)
        res_df = pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame()
        return res_df
    finally:
        conn.close()

# ---------- Streamlit UI ----------

def render_nl2sql():
    st.set_page_config(page_title="NL → SQL", layout="wide")
    st.title("Natural Language → SQL (improved)")

    st.markdown("Describe the query you want in plain English. This tool generates SQL heuristically (single-table). Use 'Run in sandbox' to execute it against the chosen DataFrame.")

    tables = _detect_tables_in_session()
    if not tables:
        st.warning("No DataFrame found in session. Upload a dataset first on Data Upload page.")
        if st.button("Show example SQL"):
            st.code("SELECT applicant_id, applicant_income, loan_amount\nFROM loans\nORDER BY applicant_income DESC\nLIMIT 5;", language="sql")
        return

    table_keys = list(tables.keys())
    table_items = [f"{k} — {v.shape[0]:,} rows × {v.shape[1]:,} cols" for k, v in tables.items()]
    selected_index = st.selectbox("Pick table to target", options=list(range(len(table_keys))), format_func=lambda i: table_items[i])
    selected_key = table_keys[selected_index]
    df = tables[selected_key]
    columns = list(df.columns)

    with st.expander("Show table columns", expanded=False):
        st.write(columns)

    examples = [
        "Show top 5 applicants by applicant_income",
        "Count loans by gender",
        "Average loan_amount by education",
        "List applicants where credit_history = 0 and property_area = 'Urban'",
        "Show loan_id, applicant_income, loan_amount where loan_amount > 200"
    ]

    col_main, col_examples = st.columns([3,1])
    with col_main:
        prompt = st.text_area("Instruction", value=examples[0], height=140)
    with col_examples:
        st.markdown("Examples")
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state["_nl_example_"] = ex
                st.rerun()

    if "_nl_example_" in st.session_state:
        prompt = st.session_state.pop("_nl_example_")

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        safe_mode = st.checkbox("Safe-by-default: add LIMIT 100 if none", value=True)
        run_sandbox = st.checkbox("Show Run in sandbox option", value=True)
    with opt_col2:
        show_expl = st.checkbox("Show explanation", value=True)
        show_download = st.checkbox("Show download button", value=True)

    if st.button("Generate SQL"):
        sql, explanation = translate_prompt_to_sql(prompt, selected_key, columns)

        # Safe-mode addition
        if safe_mode and not re.search(r'\blimit\b', sql, flags=re.I):
            # don't add LIMIT if aggregated and grouped (can still be fine but safer not to add)
            if not re.search(r'\b(group\s+by|count\(|sum\(|avg\(|min\(|max\()', sql, flags=re.I):
                sql = sql[:-1] + "\nLIMIT 100;"

        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        if show_expl:
            st.subheader("Explanation")
            st.write(explanation)

        if show_download:
            st.download_button("Download SQL (.sql)", data=sql.encode("utf-8"), file_name="generated_query.sql", mime="text/sql")

        # Run in sandbox
        if run_sandbox:
            if st.button("Run in sandbox (execute on selected DataFrame)"):
                try:
                    res = _run_sql_in_memory(df, "data_table", sql, max_rows=1000)
                    st.success(f"Returned {len(res)} rows (preview).")
                    st.dataframe(res)
                except Exception as e:
                    st.error("Execution failed (sandbox). See error below.")
                    st.exception(e)

    else:
        st.info("Type an instruction and click 'Generate SQL'.")

