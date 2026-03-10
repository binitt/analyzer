import os
import re
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

from pydriller import Repository
from radon.complexity import cc_visit
from radon.raw import analyze


REPO_PATH = "./TicketBooking"
OUTPUT_FILE = "data/dataset.csv"


BUG_KEYWORDS = [
    "fix", "bug", "issue", "error", "defect",
    "patch", "resolve", "correct"
]


# temp comment
def analyze_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        raw = analyze(code)
        complexity_blocks = cc_visit(code)

        avg_complexity = (
            sum(c.complexity for c in complexity_blocks) / len(complexity_blocks)
            if complexity_blocks else 0
        )

        num_functions = len(complexity_blocks)

        num_classes = len(re.findall(r"\bclass\s+\w+", code))

        comment_density = raw.comments / raw.loc if raw.loc > 0 else 0

        avg_function_length = raw.loc / num_functions if num_functions > 0 else 0

        return raw.loc, avg_complexity, num_functions, num_classes, comment_density, avg_function_length

    except Exception:
        return 0, 0, 0, 0, 0, 0


def extract_git_features(repo_path):

    file_data = defaultdict(lambda: {
        "num_commits": 0,
        "authors": set(),
        "churn": 0,
        "lines_added": 0,
        "lines_deleted": 0,
        "bug_fix_commits": 0,
        "first_commit": None,
        "last_commit": None
    })

    for commit in Repository(repo_path).traverse_commits():

        commit_time = commit.committer_date.astimezone(timezone.utc)

        message = commit.msg.lower()

        is_bugfix = any(k in message for k in BUG_KEYWORDS)

        for m in commit.modified_files:

            path = m.new_path or m.old_path

            if path is None or not path.endswith(".py"):
                continue

            data = file_data[path]

            data["num_commits"] += 1
            data["authors"].add(commit.author.name)

            added = m.added_lines or 0
            deleted = m.deleted_lines or 0

            data["lines_added"] += added
            data["lines_deleted"] += deleted
            data["churn"] += added + deleted

            if is_bugfix:
                data["bug_fix_commits"] += 1

            if data["first_commit"] is None or commit_time < data["first_commit"]:
                data["first_commit"] = commit_time

            if data["last_commit"] is None or commit_time > data["last_commit"]:
                data["last_commit"] = commit_time

    return file_data


def build_dataset(repo_path):

    git_data = extract_git_features(repo_path)

    rows = []

    now = datetime.now(timezone.utc)

    for file_path, data in git_data.items():

        abs_path = os.path.join(repo_path, file_path)

        (
            loc,
            complexity,
            num_functions,
            num_classes,
            comment_density,
            avg_function_length
        ) = (0, 0, 0, 0, 0, 0)

        if os.path.exists(abs_path):
            (
                loc,
                complexity,
                num_functions,
                num_classes,
                comment_density,
                avg_function_length
            ) = analyze_file(abs_path)

        first_commit = data["first_commit"]

        file_age_days = (now - first_commit).days if first_commit else 0

        commit_frequency = (
            data["num_commits"] / file_age_days if file_age_days > 0 else 0
        )

        bug_fix_commits = data["bug_fix_commits"]

        rows.append({
            "file_name": file_path,
            "lines_of_code": loc,
            "cyclomatic_complexity": complexity,
            "num_functions": num_functions,
            "num_classes": num_classes,
            "comment_density": comment_density,
            "code_churn": data["churn"],
            "num_developers": len(data["authors"]),
            "commit_frequency": commit_frequency,
            "avg_function_length": avg_function_length,
            "bug_fix_commits": bug_fix_commits,

            # same value as requested
            "past_defects": bug_fix_commits
        })

    return pd.DataFrame(rows)


def main():

    print("Mining git repository...")

    df = build_dataset(REPO_PATH)

    print("Rows extracted:", len(df))

    df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
