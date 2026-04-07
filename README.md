# BugDetect Feature Extractor

This project extracts software engineering metrics from a Git repository to build a dataset for **bug prediction**, **code quality analysis**, and **software analytics**.

The dataset can be used for machine learning models that predict which files are more likely to contain defects.

---

## Run.

```bash
python analyzer/feature-extractor.py
```

The script analyzes the repository and generates a **CSV dataset** containing metrics for each source file.

---

## What the Script Does

The feature extractor performs the following steps:

1. **Mine Git History**

   * Uses **PyDriller** to analyze the commit history of the repository.
   * Tracks which files were modified, how often they changed, and who changed them.

2. **Compute Code Metrics**

   * Uses **Radon** to measure code complexity and size.

3. **Build Dataset**

   * Combines Git metrics and code metrics into a structured dataset.
   * Outputs the dataset as `dataset.csv`.

This dataset can then be used to train models that identify **high-risk files**, **maintenance hotspots**, or **bug-prone modules**.

---

## Components Used

| Component | Purpose                                                     |
| --------- | ----------------------------------------------------------- |
| PyDriller | Mining Git history and file change information              |
| Radon     | Computing code metrics such as complexity and lines of code |
| Pandas    | Creating and exporting the dataset                          |

---

## Extracted Metrics

Each row in the dataset corresponds to a **file** in the repository.

### file_name

Path of the file within the repository.

---

### loc (Lines of Code)

The number of lines of code contained in a file.

This is a basic size metric used in software engineering research.
Larger files often have a higher probability of defects simply due to increased complexity.

---

### cyclomatic_complexity

Cyclomatic complexity measures **how complex the control flow of a program is**.

It is calculated based on the number of decision points such as:

* `if`
* `for`
* `while`
* `case`
* logical branches

Higher values indicate:

* more execution paths
* harder testing
* higher likelihood of bugs

Example interpretation:

| Complexity | Risk         |
| ---------- | ------------ |
| 1–5        | Simple       |
| 6–10       | Moderate     |
| 11–20      | Complex      |
| 21+        | Very complex |

---

### num_commits

Number of commits that modified the file.

Files that change frequently are often **more unstable** and may contain more defects.

---

### num_authors

Number of unique developers who modified the file.

Higher numbers may indicate:

* lower ownership
* coordination challenges
* increased risk of defects

---

### churn

Total number of lines added and removed over time.

```
churn = lines_added + lines_deleted
```

High churn suggests:

* frequent changes
* unstable code areas
* potential bug hotspots

---

### lines_added / lines_deleted

Total lines added or removed across the file's history.

These help measure how actively a file evolves.

---

### file_age_days

Number of days since the file first appeared in the repository.

Older files may become more stable over time.

---

### last_modified_days

Number of days since the file was last modified.

Recently modified files may require closer attention during testing.

---

## Example Dataset

```
file_name,loc,cyclomatic_complexity,num_commits,num_authors,churn
main.py,320,7.4,15,4,720
auth.py,110,3.1,8,2,240
db/models.py,450,12.3,22,5,1100
```

---

## Use Cases

The generated dataset can be used for:

* Bug prediction models
* Code quality analysis
* Identifying risky or complex modules
* Prioritizing testing efforts
* Software engineering research

---

## Future Improvements

Possible extensions include:

* developer ownership metrics
* change entropy
* bug-fix labeling from commit messages
* hotspot detection
* additional complexity metrics

---

# LangGraph

## Start Ollama locally

Download and start ollama
```
mkdir llm
export OLLAMA_MODELS=./llm
export OLLAMA_HOST=0.0.0.0
ollama pull tinyllama
ollama serve
```

Expose to public with ngrok
```
ngrok http http://localhost:11434
```

