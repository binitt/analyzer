from pydriller import Repository
from collections import defaultdict
import pandas as pd

repo_path = "./TicketBooking"

file_metrics = defaultdict(lambda: {
    "num_commits": 0,
    "authors": set(),
    "churn": 0
})

for commit in Repository(repo_path).traverse_commits():

    for m in commit.modified_files:

        if not m.filename.endswith(".py"):
            continue

        path = m.new_path or m.old_path

        file_metrics[path]["num_commits"] += 1
        file_metrics[path]["authors"].add(commit.author.name)

        added = m.added_lines or 0
        deleted = m.deleted_lines or 0

        file_metrics[path]["churn"] += added + deleted


rows = []

for f, data in file_metrics.items():

    rows.append({
        "file_name": f,
        "num_commits": data["num_commits"],
        "num_authors": len(data["authors"]),
        "churn": data["churn"]
    })

df = pd.DataFrame(rows)

print(df.head())
