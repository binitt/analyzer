from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph


CSV_PATH = "data/predictions.csv"
PROJECT_DIR = "TicketBooking"


class GraphState(TypedDict, total=False):
    row: Dict[str, Any]
    file: str
    metrics: Dict[str, Any]
    prediction: Dict[str, Any]
    raw_code: str
    functions: List[Dict[str, Any]]
    risk_level: str
    analysis_signals: List[Dict[str, Any]]
    function_results: List[Dict[str, Any]]
    report: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def metrics_loader(state: GraphState) -> GraphState:
    row = state["row"]
    file_name = str(row.get("file_name", ""))
    project_dir = _as_path(PROJECT_DIR)
    file_path = project_dir / file_name

    raw_code = ""
    if file_path.exists() and file_path.is_file():
        raw_code = file_path.read_text(encoding="utf-8", errors="ignore")
    else:
        print(f"[WARN] File not found: {file_path}")

    metrics = {k: v for k, v in row.items() if k not in {"file_name", "defect", "prob_no_defect", "prob_defect"}}
    prediction = {
        "defect": row.get("defect"),
        "prob_no_defect": _safe_float(row.get("prob_no_defect")),
        "prob_defect": _safe_float(row.get("prob_defect")),
    }

    return {
        **state,
        "file": file_name,
        "metrics": metrics,
        "prediction": prediction,
        "raw_code": raw_code,
    }


def ast_splitter(state: GraphState) -> GraphState:
    file_name = state.get("file", "")
    raw_code = state.get("raw_code", "")
    functions: List[Dict[str, Any]] = []

    if not raw_code:
        return {**state, "functions": functions}

    try:
        tree = ast.parse(raw_code)
    except SyntaxError as exc:
        print(f"[WARN] AST parse failed for {file_name}: {exc}")
        return {**state, "functions": functions}

    lines = raw_code.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            fn_lines = lines[start - 1 : end]

            if (end - start + 1) > 500:
                print(f"[INFO] Skipping function >500 lines: {file_name}:{node.name}")
                continue

            functions.append(
                {
                    "name": node.name,
                    "start_line": start,
                    "end_line": end,
                    "code": "\n".join(fn_lines),
                }
            )

    return {**state, "functions": functions}


def risk_interpreter(state: GraphState) -> GraphState:
    prob_defect = _safe_float(state.get("prediction", {}).get("prob_defect"))
    if prob_defect > 0.75:
        risk = "HIGH"
    elif prob_defect >= 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {**state, "risk_level": risk}


def route_by_risk(state: GraphState) -> str:
    return "report_builder" if state.get("risk_level") == "LOW" else "code_analyzer"


def code_analyzer(state: GraphState) -> GraphState:
    signals: List[Dict[str, Any]] = []
    for fn in state.get("functions", []):
        code = fn.get("code", "")
        lower = code.lower()
        signals.append(
            {
                "function": fn.get("name"),
                "length": fn.get("end_line", 0) - fn.get("start_line", 0) + 1,
                "has_bare_except": "except:" in lower,
                "has_todo": "todo" in lower,
                "has_print_debug": "print(" in lower,
                "uses_eval": "eval(" in lower,
            }
        )
    return {**state, "analysis_signals": signals}


def issue_detector(state: GraphState) -> GraphState:
    results: List[Dict[str, Any]] = []
    for signal in state.get("analysis_signals", []):
        issues: List[str] = []
        if signal.get("length", 0) > 80:
            issues.append("Function is long and likely hard to maintain/test.")
        if signal.get("has_bare_except"):
            issues.append("Bare except found; this can hide real exceptions.")
        if signal.get("uses_eval"):
            issues.append("Use of eval detected; this is a security risk.")
        if signal.get("has_print_debug"):
            issues.append("Debug print statements found in function body.")
        if signal.get("has_todo"):
            issues.append("TODO marker found, indicating unfinished logic.")

        if issues:
            results.append({"function": signal.get("function"), "issues": issues})
    return {**state, "function_results": results}


def fix_generator(state: GraphState) -> GraphState:
    enriched: List[Dict[str, Any]] = []
    for item in state.get("function_results", []):
        fixes: List[str] = []
        for issue in item.get("issues", []):
            if "Bare except" in issue:
                fixes.append("Catch specific exceptions and log context before handling.")
            elif "eval" in issue:
                fixes.append("Replace eval with safe parsing or explicit dispatch logic.")
            elif "Debug print" in issue:
                fixes.append("Use structured logging with proper log levels.")
            elif "TODO marker" in issue:
                fixes.append("Resolve TODO with implementation or track it in issue backlog.")
            elif "Function is long" in issue:
                fixes.append("Refactor into smaller helpers with single responsibilities.")
            else:
                fixes.append("Review and refactor this pattern.")

        enriched.append({**item, "fixes": fixes})
    return {**state, "function_results": enriched}


def aggregate_results(state: GraphState) -> GraphState:
    results = state.get("function_results", [])
    issue_count = sum(len(r.get("issues", [])) for r in results)
    summary = {
        "function_count": len(state.get("functions", [])),
        "flagged_functions": len(results),
        "total_issues": issue_count,
    }
    return {**state, "summary": summary}


def report_builder(state: GraphState) -> GraphState:
    file_name = state.get("file", "<unknown>")
    risk = state.get("risk_level", "UNKNOWN")
    prob = _safe_float(state.get("prediction", {}).get("prob_defect"))
    results = state.get("function_results", [])

    lines = [
        f"File: {file_name}",
        f"Risk Level: {risk}",
        f"Defect Probability: {prob:.4f}",
    ]

    if risk == "LOW":
        lines.append("Analysis skipped because file risk is LOW (<0.5).")
    elif not results:
        lines.append("No obvious function-level issues detected by simple analyzer.")
    else:
        lines.append("Function-level findings:")
        for result in results:
            lines.append(f"- Function: {result.get('function')}")
            for issue in result.get("issues", []):
                lines.append(f"  Issue: {issue}")
            for fix in result.get("fixes", []):
                lines.append(f"  Fix: {fix}")

    report = "\n".join(lines)
    return {**state, "report": report}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("metrics_loader", metrics_loader)
    graph.add_node("ast_splitter", ast_splitter)
    graph.add_node("risk_interpreter", risk_interpreter)
    graph.add_node("code_analyzer", code_analyzer)
    graph.add_node("issue_detector", issue_detector)
    graph.add_node("fix_generator", fix_generator)
    graph.add_node("aggregate_results", aggregate_results)
    graph.add_node("report_builder", report_builder)

    graph.set_entry_point("metrics_loader")
    graph.add_edge("metrics_loader", "ast_splitter")
    graph.add_edge("ast_splitter", "risk_interpreter")
    graph.add_conditional_edges(
        "risk_interpreter",
        route_by_risk,
        {"code_analyzer": "code_analyzer", "report_builder": "report_builder"},
    )
    graph.add_edge("code_analyzer", "issue_detector")
    graph.add_edge("issue_detector", "fix_generator")
    graph.add_edge("fix_generator", "aggregate_results")
    graph.add_edge("aggregate_results", "report_builder")
    graph.add_edge("report_builder", END)
    return graph.compile()


def run():
    csv_path = _as_path(CSV_PATH)
    project_dir = _as_path(PROJECT_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {csv_path}")
    if not project_dir.exists():
        raise FileNotFoundError(f"Missing project directory: {project_dir}")

    app = build_graph()
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_path}")
    for idx, row in df.iterrows():
        state = {"row": row.to_dict()}
        result = app.invoke(state)
        print("\n" + "=" * 80)
        print(f"ROW {idx + 1}/{len(df)}")
        print(result.get("report", "No report generated."))


if __name__ == "__main__":
    run()
