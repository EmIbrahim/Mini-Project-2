"""
generate_report.py — Reads sft_trial_results.csv and dpo_trial_results.csv,
outputs pre-formatted Markdown tables for the final PDF report.
"""
import pandas as pd
import sys, os

def load_csv(path, label):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Using placeholder data for {label}.")
        return None
    return pd.read_csv(path)

def main():
    sft_path = sys.argv[1] if len(sys.argv) > 1 else "sft_trial_results.csv"
    dpo_path = sys.argv[2] if len(sys.argv) > 2 else "dpo_trial_results.csv"

    report = []
    report.append("# Mini-Project 2 — Experiment Results Summary")
    report.append("## Track 1, Option A: SFT → DPO Pipeline\n")

    # --- SFT Table ---
    sft_df = load_csv(sft_path, "SFT")
    if sft_df is not None:
        report.append("## Table 1: SFT Trial Results\n")
        report.append("| Trial | LoRA r | LoRA α | Target Modules | LR | Batch Size | Epochs | Train Loss | Val Loss | BLEU | BERTScore F1 |")
        report.append("|-------|--------|--------|----------------|-----|-----------|--------|------------|----------|------|--------------|")
        for _, row in sft_df.iterrows():
            trial = row.get("Trial", "")
            label = "**Baseline**" if str(trial) == "0" else str(trial)
            modules = str(row.get("Target_Modules", "-")).replace("['", "").replace("']", "").replace("', '", ", ")
            report.append(
                f"| {label} | {row.get('LoRA_r', '-')} | {row.get('LoRA_alpha', '-')} | {modules} "
                f"| {row.get('LR', '-')} | {row.get('Batch_Size', '-')} | {row.get('Epochs', '-')} "
                f"| {row.get('Train_Loss', '-')} | {row.get('Val_Loss', '-')} "
                f"| {row.get('BLEU', '-')} | {row.get('BERTScore_F1', '-')} |"
            )

        # Identify best
        trials_only = sft_df[sft_df["Trial"].astype(str) != "0"].copy()
        if not trials_only.empty:
            trials_only["BLEU"] = pd.to_numeric(trials_only["BLEU"], errors="coerce")
            trials_only["BERTScore_F1"] = pd.to_numeric(trials_only["BERTScore_F1"], errors="coerce")
            trials_only["composite"] = trials_only["BLEU"] + trials_only["BERTScore_F1"]
            best = trials_only.loc[trials_only["composite"].idxmax()]
            report.append(f"\n**Best SFT Trial: Trial {int(best['Trial'])}** — "
                          f"BLEU={best['BLEU']}, BERTScore F1={best['BERTScore_F1']}\n")

    # --- DPO Table ---
    dpo_df = load_csv(dpo_path, "DPO")
    if dpo_df is not None:
        report.append("\n## Table 2: DPO Trial Results\n")
        report.append("| Trial | Beta | LR | Batch Size | Epochs | Train Loss | Val Loss | BLEU | BERTScore F1 |")
        report.append("|-------|------|-----|-----------|--------|------------|----------|------|--------------|")
        for _, row in dpo_df.iterrows():
            report.append(
                f"| {row['Trial']} | {row['Beta']} | {row['LR']} | {row['Batch_Size']} "
                f"| {row['Epochs']} | {row['Train_Loss']} | {row['Val_Loss']} "
                f"| {row['BLEU']} | {row['BERTScore_F1']} |"
            )

        dpo_df["composite"] = pd.to_numeric(dpo_df["BLEU"], errors="coerce") + pd.to_numeric(dpo_df["BERTScore_F1"], errors="coerce")
        best = dpo_df.loc[dpo_df["composite"].idxmax()]
        report.append(f"\n**Best DPO Trial: Trial {int(best['Trial'])}** — "
                      f"BLEU={best['BLEU']}, BERTScore F1={best['BERTScore_F1']}\n")

    # Output
    output = "\n".join(report)
    print(output)
    with open("report_tables.md", "w", encoding="utf-8") as f:
        f.write(output)
    print("\n--- Report tables saved to report_tables.md ---")

if __name__ == "__main__":
    main()
