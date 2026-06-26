"""
compute_fp_rate.py

Reads back the review_checklist.csv produced by sample_for_review.py, once
a human has filled in the 'review_result' column, and reports the lane-line
false-positive rate against the agreed 5-10% go/no-go threshold.

Usage:
    python compute_fp_rate.py --checklist ~/Pavement_Distress_Detection/review_sample_cycle1/review_checklist.csv
"""

import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checklist", required=True)
    ap.add_argument("--threshold-pct", type=float, default=7.5, help="Midpoint of the agreed 5-10%% go/no-go range")
    args = ap.parse_args()

    rows = []
    with open(args.checklist, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    reviewed = [r for r in rows if r["review_result"].strip()]
    unreviewed = len(rows) - len(reviewed)

    if not reviewed:
        raise SystemExit("No rows have been filled in yet. Fill in 'review_result' for each image first.")

    lane_fp = sum(1 for r in reviewed if r["review_result"].strip().lower() == "lane_line_fp")
    other_fp = sum(1 for r in reviewed if r["review_result"].strip().lower() == "other_fp")
    keep = sum(1 for r in reviewed if r["review_result"].strip().lower() == "keep")
    unsure = sum(1 for r in reviewed if r["review_result"].strip().lower() == "unsure")

    n = len(reviewed)
    lane_fp_pct = lane_fp / n * 100

    print("=" * 60)
    print(f"Reviewed: {n} / {len(rows)} images ({unreviewed} not yet reviewed)")
    print(f"  keep:         {keep} ({keep/n*100:.1f}%)")
    print(f"  lane_line_fp: {lane_fp} ({lane_fp_pct:.1f}%)")
    print(f"  other_fp:     {other_fp} ({other_fp/n*100:.1f}%)")
    print(f"  unsure:       {unsure} ({unsure/n*100:.1f}%)")
    print("-" * 60)
    if lane_fp_pct > args.threshold_pct:
        print(f"RESULT: {lane_fp_pct:.1f}% exceeds the {args.threshold_pct}% threshold.")
        print("Do NOT merge this cycle yet -- address the lane-line issue first")
        print("(negative class examples, or a geometric/color post-filter).")
    else:
        print(f"RESULT: {lane_fp_pct:.1f}% is within the {args.threshold_pct}% threshold.")
        print("Lane-line false-positive rate looks acceptable for merging,")
        print("pending the class-distribution check passing too.")
    print("=" * 60)


if __name__ == "__main__":
    main()
