import os
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

from pydriller import Repository
from radon.complexity import cc_visit
from radon.raw import analyze


REPO_PATH = "./TicketBooking"
OUTPUT_FILE = "dataset.csv"


def analyze_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        raw = analyze(code)
        complexity_blocks = cc_visit(code)

        if complexity_blocks:
            avg_complexity = sum(c.complexity for c in complexity_blocks) / len(complexity_blocks)
        else:
            avg_complexity = 0

        return raw.loc, avg_complexity

    except Exception:
        return 0, 0


def extract_git_features(repo_path):

    file_data = defaultdict(lambda: {
        "num_commits": 0,
        "authors": set(),
        "churn": 0,
        "lines_added": 0,
        "lines_deleted": 0,
        "first_commit": None,
        "last_commit": None
    })

    for commit in Repository(repo_path).traverse_commits():

        for m in commit.modified_files:

            path = m.new_path or m.old_path

            if path is None:
                continue

            if not path.endswith(".py"):
                continue

            data = file_data[path]

            data["num_commits"] += 1
            data["authors"].add(commit.author.name)

            added = m.added_lines or 0
            deleted = m.deleted_lines or 0

            data["lines_added"] += added
            data["lines_deleted"] += deleted
            data["churn"] += added + deleted

            commit_time = commit.committer_date

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

        loc = 0
        complexity = 0

        if os.path.exists(abs_path):
            loc, complexity = analyze_file(abs_path)

        first_commit = data["first_commit"]
        last_commit = data["last_commit"]
        
        if first_commit:
            first_commit = first_commit.astimezone(timezone.utc)
        
        if last_commit:
            last_commit = last_commit.astimezone(timezone.utc)
        
        file_age_days = (now - first_commit).days if first_commit else 0
        last_modified_days = (now - last_commit).days if last_commit else 0

        rows.append({
            "file_name": file_path,
            "loc": loc,
            "cyclomatic_complexity": complexity,
            "num_commits": data["num_commits"],
            "num_authors": len(data["authors"]),
            "lines_added": data["lines_added"],
            "lines_deleted": data["lines_deleted"],
            "churn": data["churn"],
            "file_age_days": file_age_days,
            "last_modified_days": last_modified_days
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
