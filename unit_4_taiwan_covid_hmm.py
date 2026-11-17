# Unit 4: Taiwan COVID-19 HMM backward algorithm
# Extracted from Appendix A of Team_FloppyDisk_Extended_Final_Report.docx.
# For the original Colab notebook where available, see the matching .ipynb file.

!pip -q install pandas numpy matplotlib requests openpyxl

import os, re, zipfile, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

FIGSHARE_ARTICLE_ID = 24623964
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
DATA_DIR = "/content/unit4_taiwan_covid"
os.makedirs(DATA_DIR, exist_ok=True)

response = requests.get(FIGSHARE_API, timeout=60)
response.raise_for_status()
article = response.json()
print("Article:", article.get("title"), "| version:", article.get("version"))
for item in article.get("files", []):
    print("•", item["name"], f"({item.get('size', 0):,} bytes)")

files = article.get("files", [])
excel_files = [f for f in files if f["name"].lower().endswith((".xlsx", ".xls"))]
zip_files = [f for f in files if f["name"].lower().endswith(".zip")]

if excel_files:
    preferred = [f for f in excel_files if "figshare_taiwan_covid" in f["name"].lower()]
    selected_file = preferred[0] if preferred else excel_files[0]
elif zip_files:
    selected_file = zip_files[0]
else:
    raise RuntimeError("No Excel or ZIP data file found in the Figshare article.")

r = requests.get(selected_file["download_url"], timeout=120)
r.raise_for_status()
download_path = os.path.join(DATA_DIR, selected_file["name"])
with open(download_path, "wb") as f:
    f.write(r.content)

if download_path.lower().endswith(".zip"):
    with zipfile.ZipFile(download_path) as z:
        workbooks = [n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls"))]
        if not workbooks:
            raise RuntimeError("The downloaded ZIP contains no Excel workbook.")
        preferred = [n for n in workbooks if "figshare_taiwan_covid" in n.lower()]
        workbook_path = z.extract(preferred[0] if preferred else workbooks[0], DATA_DIR)
else:
    workbook_path = download_path
print("Workbook saved to:", workbook_path)

def normalize_column(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

excel = pd.ExcelFile(workbook_path)
print("Sheets:", excel.sheet_names)
frames = {}
for sheet in excel.sheet_names:
    df = pd.read_excel(workbook_path, sheet_name=sheet)
    df.columns = [normalize_column(c) for c in df.columns]
    frames[sheet] = df

markers = {
    "id", "onset_of_symptom", "confirmed_date", "icu",
    "recovery", "death_date", "symptom",
    "date_of_contact_with_infected_case"
}
scores = {name: len(markers.intersection(df.columns)) for name, df in frames.items()}
print("Case-field matches:", scores)
case_sheet = max(scores, key=scores.get)
cases = frames[case_sheet].copy()
if "id" not in cases.columns:
    raise KeyError(f"Case id not found. Review columns: {cases.columns.tolist()}")
print("Selected:", case_sheet, "| shape:", cases.shape)
print(cases.columns.tolist())
display(cases.head())

date_fields = [
    "date_of_contact_with_infected_case", "onset_of_symptom",
    "report_to_cdc", "confirmed_date", "icu", "recovery", "death_date"
]
date_fields = [c for c in date_fields if c in cases.columns]

def parse_date(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value, errors="coerce")
    text = str(value).strip()
    m = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text)
    if m:
        text = m.group(0)
    return pd.to_datetime(text, errors="coerce")

for col in date_fields:
    cases[col] = cases[col].map(parse_date)

coverage = pd.DataFrame({
    "field": date_fields,
    "rows_with_date": [int(cases[c].notna().sum()) for c in date_fields],
    "missing_rows": [int(cases[c].isna().sum()) for c in date_fields],
})
coverage["coverage_percent"] = (100 * coverage["rows_with_date"] / len(cases)).round(1)
display(coverage)
ax = coverage.set_index("field")["coverage_percent"].sort_values().plot(
    kind="barh", figsize=(9, 4), color="#2563eb"
)
ax.set(xlabel="Rows with parseable date (%)", ylabel="", xlim=(0, 100),
       title="Available date coverage in the case-course sheet")
plt.tight_layout()
plt.show()

event_fields = {
    "date_of_contact_with_infected_case": "contact",
    "onset_of_symptom": "symptom",
    "confirmed_date": "confirmation",
    "icu": "critical",
    "recovery": "recovery",
    "death_date": "death",
}
event_fields = {c: label for c, label in event_fields.items() if c in cases.columns}
if not event_fields:
    raise RuntimeError("No recognized case-event date columns found; check the codebook.")

event_rows = []
for _, row in cases.iterrows():
    by_date = {}
    for col, label in event_fields.items():
        value = row.get(col, pd.NaT)
        if pd.notna(value):
            day = pd.Timestamp(value).normalize()
            by_date.setdefault(day, []).append(label)
    for day, labels in sorted(by_date.items()):
        event_rows.append({
            "case_id": row["id"],
            "event_date": day,
            "observation": labels[0] if len(labels) == 1 else "multiple",
            "reported_event_types": ", ".join(sorted(set(labels))),
        })

events = pd.DataFrame(event_rows).sort_values(["case_id", "event_date"]).reset_index(drop=True)
sequences = {cid: g["observation"].tolist() for cid, g in events.groupby("case_id", sort=False)}
print(f"Cases with dated events: {len(sequences):,} / {len(cases):,}")
print("Event counts:")
display(events.groupby("observation").size().rename("count").to_frame())
display(events.head(12))

states = ["asymptomatic", "symptomatic", "confirmed", "critical", "recovered", "deceased"]
observations = ["contact", "symptom", "confirmation", "critical", "recovery", "death", "multiple"]
obs_index = {name: i for i, name in enumerate(observations)}

pi = np.array([0.60, 0.24, 0.10, 0.03, 0.02, 0.01], dtype=float)

# Rows: current hidden state; columns: next state.
A = pd.DataFrame([
    [0.74, 0.20, 0.04, 0.01, 0.01, 0.00],
    [0.00, 0.42, 0.50, 0.04, 0.03, 0.01],
    [0.00, 0.00, 0.57, 0.14, 0.26, 0.03],
    [0.00, 0.00, 0.00, 0.52, 0.33, 0.15],
    [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
], index=states, columns=states)

# Rows: hidden state; columns: observed report type.
B = pd.DataFrame([
    [0.28, 0.04, 0.15, 0.02, 0.05, 0.01, 0.45],
    [0.12, 0.45, 0.25, 0.03, 0.03, 0.01, 0.11],
    [0.05, 0.08, 0.55, 0.11, 0.17, 0.02, 0.02],
    [0.03, 0.04, 0.20, 0.48, 0.17, 0.06, 0.02],
    [0.02, 0.03, 0.08, 0.02, 0.80, 0.03, 0.02],
    [0.02, 0.02, 0.08, 0.03, 0.05, 0.78, 0.02],
], index=states, columns=observations)

assert np.isclose(pi.sum(), 1)
assert np.allclose(A.sum(axis=1), 1)
assert np.allclose(B.sum(axis=1), 1)
display(A.style.format("{:.2f}").set_caption("Illustrative transition probabilities"))
display(B.style.format("{:.2f}").set_caption("Illustrative emission probabilities"))

A_array = A.to_numpy(dtype=float)
B_array = B.to_numpy(dtype=float)

def forward_backward(observation_sequence):
    obs_idx = [obs_index[o] for o in observation_sequence]
    T = len(obs_idx)
    if T == 0:
        raise ValueError("Observation sequence is empty.")
    n_states = len(states)
    alpha = np.zeros((T, n_states), dtype=float)
    beta = np.ones((T, n_states), dtype=float)

    alpha[0] = pi * B_array[:, obs_idx[0]]
    alpha[0] /= alpha[0].sum()
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A_array) * B_array[:, obs_idx[t]]
        total = alpha[t].sum()
        alpha[t] = alpha[t] / total if total else np.full(n_states, 1 / n_states)

    # Backward pass; beta at final event is 1 for all states.
    for t in range(T - 2, -1, -1):
        beta[t] = A_array @ (B_array[:, obs_idx[t + 1]] * beta[t + 1])
        total = beta[t].sum()
        if total:
            beta[t] /= total

    posterior = alpha * beta
    totals = posterior.sum(axis=1, keepdims=True)
    posterior = np.divide(posterior, totals, out=np.full_like(posterior, 1 / n_states), where=totals > 0)
    return (pd.DataFrame(alpha, columns=states),
            pd.DataFrame(beta, columns=states),
            pd.DataFrame(posterior, columns=states))

if not sequences:
    raise RuntimeError("No dated case sequences were built.")

sample_case_id = max(sequences, key=lambda cid: len(sequences[cid]))
sample_events = events[events["case_id"] == sample_case_id].sort_values("event_date").reset_index(drop=True)
sample_observations = sample_events["observation"].tolist()
alpha, beta, posterior = forward_backward(sample_observations)

result = sample_events[["case_id", "event_date", "observation", "reported_event_types"]].copy()
result["filtered_most_likely_stage"] = alpha.idxmax(axis=1).values
result["smoothed_most_likely_stage"] = posterior.idxmax(axis=1).values
for stage in states:
    result[f"P({stage})"] = posterior[stage].round(3).values

print("Illustration case ID:", sample_case_id)
display(result)

plot_data = posterior.copy()
plot_data.index = sample_events["event_date"].dt.strftime("%Y-%m-%d")
ax = plot_data.plot(marker="o", figsize=(12, 5), colormap="tab10")
ax.set(title=f"Smoothed state probabilities — case {sample_case_id}",
       xlabel="Recorded event date", ylabel="Probability", ylim=(0, 1))
ax.legend(title="Candidate hidden stage", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
