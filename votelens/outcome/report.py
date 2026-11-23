"""Render Task B result summaries."""

import json


def write_report(output, results, metadata):
    model_order = [
        "majority_class",
        "vote_statistics",
        "text_only",
        "graph_only",
        "text_and_graph",
    ]
    window_order = {
        "percentage": ["10%", "25%", "50%", "75%"],
        "fixed": ["5 votes", "10 votes", "20 votes"],
    }
    tables = []
    for protocol, columns in window_order.items():
        frame = results.loc[results.protocol.eq(protocol)]
        if frame.empty:
            continue
        pivot = (
            frame.pivot(index="model", columns="window", values="macro_f1")
            .reindex(model_order)
            .reindex(columns=columns)
            .mul(100)
            .round(2)
        )
        cohort = frame[
            [
                "window",
                "train_elections",
                "validation_elections",
                "test_elections",
            ]
        ].drop_duplicates()
        rank = {value: index for index, value in enumerate(columns)}
        cohort = cohort.sort_values("window", key=lambda values: values.map(rank))
        title = (
            "Retrospective percentage windows"
            if protocol == "percentage"
            else "Deployment-realistic fixed windows"
        )
        tables.append(
            f"<h2>{title}: macro-F1 (%)</h2>{pivot.to_html()}"
            f"<h3>Cohort sizes</h3>{cohort.to_html(index=False)}"
        )
    tables_html = "".join(tables)
    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task B baseline results</title>
<style>
body {{ margin:0; color:#21334c; background:#f2f5f8; font-family:system-ui,sans-serif; }}
main {{ max-width:1100px; margin:24px auto; padding:28px 36px; background:white; }}
table {{ border-collapse:collapse; width:100%; margin:20px 0; font-size:.93rem; }}
th,td {{ border:1px solid #dce4ec; padding:7px 9px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ background:#f3f6f9; }}
code {{ background:#eef3f6; padding:2px 4px; }}
.note {{ border-left:4px solid #39779f; background:#f2f7fa; padding:12px 16px; }}
</style>
<main>
<h1>Task B: early election-outcome baselines</h1>
<p class="note">Primary score is election-level macro-F1. Percentage windows are a
retrospective benchmark because they use final election size only to select the
prefix; fixed vote windows are deployment-realistic.</p>
{tables_html}
<h2>Protocol</h2>
<pre>{json.dumps(metadata, indent=2)}</pre>
</main>
</html>
"""
    (output / "report.html").write_text(html, encoding="utf-8")
