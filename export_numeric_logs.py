import os
import re
import csv
import math

def main():
    log_dirs = [
        "src_continuous_control/runs/logs",
        "src_continuous_control/src_continuous_control/runs/logs"
    ]
    out_csv_dir = "src_continuous_control/numeric_logs_csv"
    os.makedirs(out_csv_dir, exist_ok=True)

    pattern = re.compile(
        r"\[train\] step (\d+)/\d+\s+return=([-+]?\d*\.\d+|\d+)\s+sps=(\d+)\s+actor_loss=([-+]?\d*\.\d+|\d+)\s+critic_loss=([-+]?\d*\.\d+|\d+)"
    )

    agents = ["pt", "vanilla", "ewc"]
    seeds = [0, 1, 2, 3, 4]
    prefixes = ["singletask_", ""]

    exported_count = 0
    for prefix in prefixes:
        for ag in agents:
            for s in seeds:
                log_name = f"{prefix}{ag}_seed_{s}.log"
                found_log = None
                for ldir in log_dirs:
                    cand = os.path.join(ldir, log_name)
                    if os.path.exists(cand):
                        found_log = cand
                        break
                if not found_log:
                    continue

                csv_name = f"{prefix}{ag}_seed_{s}_numeric_metrics.csv"
                out_csv = os.path.join(out_csv_dir, csv_name)

                rows = []
                with open(found_log, "r") as f:
                    for line in f:
                        match = pattern.search(line)
                        if match:
                            closs = float(match.group(5))
                            rows.append({
                                "step": int(match.group(1)),
                                "return": float(match.group(2)),
                                "sps": int(match.group(3)),
                                "actor_loss": float(match.group(4)),
                                "critic_loss": closs,
                                "td_error_mse": closs,
                                "td_error_rmse": round(math.sqrt(2.0 * max(0.0, closs)), 4)
                            })

                if rows:
                    with open(out_csv, "w", newline="") as cf:
                        writer = csv.DictWriter(cf, fieldnames=["step", "return", "sps", "actor_loss", "critic_loss", "td_error_mse", "td_error_rmse"])
                        writer.writeheader()
                        writer.writerows(rows)
                    exported_count += 1

    print(f"Successfully exported {exported_count} numeric CSV files to {out_csv_dir}")

if __name__ == "__main__":
    main()
