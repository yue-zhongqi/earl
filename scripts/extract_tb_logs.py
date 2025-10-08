import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def extract_scalars(logdir, tag, smooth=0.0):
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return pd.DataFrame()
    events = ea.Scalars(tag)
    values = [e.value for e in events]
    steps = [e.step for e in events]

    if smooth > 0:
        smoothed = []
        last = values[0]
        for v in values:
            last = smooth * last + (1 - smooth) * v
            smoothed.append(last)
        values = smoothed

    return pd.DataFrame({"step": steps, "value": values})

# Example: read several runs
logdirs = [
    "tensorboard_log/earl/calc_bench/qwen2.5-3b/earl-cpo",
    "tensorboard_log/earl/calc_bench/qwen2.5-3b/earl-cpo-no-init",
    "tensorboard_log/earl/calc_bench/qwen2.5-3b/earl-cpo-init-step",
]

# tag = "critic/score/mean"
tag = "val-core/tool_uses"
dfs = []
for ld in logdirs:
    run_name = os.path.basename(ld.rstrip("/"))
    df = extract_scalars(ld, tag, 0.5)
    if not df.empty:
        df = df.rename(columns={"value": f"value({run_name})"})
        dfs.append(df)

# Merge on "step"
if dfs:
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(result, df, on="step", how="outer")
    result = result.sort_values("step")
    result.to_csv("scripts/critic_score_mean_wide_base.csv", index=False)
    print("Saved to scripts/critic_score_mean_wide_base.csv")
else:
    print("No data found for tag:", tag)