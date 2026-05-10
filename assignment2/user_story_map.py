"""
Assignment 2 — User Story Map
==============================
Generates a User Story Map (USM) visualization for the three stakeholders:
  1. Head of Data Science
  2. Company Director
  3. Loan Applicant

Each map shows: user goal -> activities -> tasks -> information/explanations encountered.

Usage:
    python assignment2/user_story_map.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
C_BLUE = "#3A86FF"
C_ORANGE = "#FF6B35"
C_GREEN = "#06D6A0"
C_RED = "#EF476F"
C_PURPLE = "#845EC2"
C_GRAY = "#8D99AE"
C_DARK = "#1A1A2E"
BG = "#FAFAFA"

# --------------------------------------------------------------------------
# USM data for each stakeholder
# --------------------------------------------------------------------------

USM_DATA = {
    "Head of Data Science": {
        "color": C_PURPLE,
        "goal": "Validate & debug the model before deployment",
        "activities": [
            {
                "name": "Review model\nperformance",
                "tasks": [
                    "Compare accuracy,\nAUC across models",
                    "Inspect confusion\nmatrices",
                    "Check per-class\nperformance",
                ],
                "info": [
                    "Accuracy, ROC-AUC\nmetrics table",
                    "Confusion matrix\nheatmaps",
                    "Classification\nreport",
                ],
            },
            {
                "name": "Audit feature\nimportance",
                "tasks": [
                    "Examine global\nSHAP rankings",
                    "Compare feature use\nacross models",
                    "Flag sensitive\nattribute reliance",
                ],
                "info": [
                    "SHAP bar charts\n(both models)",
                    "Side-by-side\nfeature comparison",
                    "Sensitive features\nhighlighted in red",
                ],
            },
            {
                "name": "Assess fairness\n& bias",
                "tasks": [
                    "Check approval rates\nby demographic group",
                    "Evaluate disparate\nimpact ratio",
                    "Inspect latent space\nfor concept encoding",
                ],
                "info": [
                    "Approval rate\nbar charts",
                    "80% rule\nthreshold line",
                    "t-SNE plots colored\nby sensitive attrs",
                ],
            },
            {
                "name": "Investigate\nedge cases",
                "tasks": [
                    "Study local SHAP\nfor misclassifications",
                    "Check if proxy features\nencode removed attrs",
                    "Review concept\nprobe results",
                ],
                "info": [
                    "Local SHAP\nwaterfall plots",
                    "Proxy correlation\nanalysis",
                    "Probe accuracy\nvs baseline table",
                ],
            },
        ],
    },
    "Company Director": {
        "color": C_ORANGE,
        "goal": "Decide whether to approve deployment",
        "activities": [
            {
                "name": "Understand\nmodel accuracy",
                "tasks": [
                    "See high-level\nperformance summary",
                    "Compare the\ntwo models",
                ],
                "info": [
                    "Simple accuracy\nbar chart (%)",
                    "Side-by-side\nmodel comparison",
                ],
            },
            {
                "name": "Understand what\nthe AI uses",
                "tasks": [
                    "See top decision\nfactors",
                    "Check if sensitive\ndata is used",
                ],
                "info": [
                    "Top-5 features\nbutterfly chart",
                    "Sensitive features\nflagged in red",
                ],
            },
            {
                "name": "Assess fairness\n& risk",
                "tasks": [
                    "Check for gender/\nrace bias",
                    "Review deployment\nrisk level",
                ],
                "info": [
                    "Approval rates by\ndemographic group",
                    "Risk verdict card\n(LOW/MEDIUM/HIGH)",
                ],
            },
            {
                "name": "Make deployment\ndecision",
                "tasks": [
                    "Read recommendation\nsummary",
                    "Identify conditions\nfor deployment",
                ],
                "info": [
                    "Deployment verdict\nwith justification",
                    "Required actions\nbefore go-live",
                ],
            },
        ],
    },
    "Loan Applicant": {
        "color": C_BLUE,
        "goal": "Understand decision & find path to approval",
        "activities": [
            {
                "name": "See the\ndecision",
                "tasks": [
                    "View approval/\nrejection status",
                    "See approval\nprobability",
                ],
                "info": [
                    "Clear APPROVED/\nREJECTED label",
                    "Probability gauge\n(0-100%)",
                ],
            },
            {
                "name": "Understand\nwhy",
                "tasks": [
                    "See which factors\nhelped or hurt",
                    "Understand factor\nimportance ranking",
                ],
                "info": [
                    "LIME waterfall\n(green/red bars)",
                    "Factors sorted\nby impact",
                ],
            },
            {
                "name": "Explore what\nto change",
                "tasks": [
                    "Identify actionable\nfactors",
                    "Distinguish what you\ncan vs cannot change",
                ],
                "info": [
                    "Actionable tips\npanel",
                    "Controllable factors\nhighlighted",
                ],
            },
            {
                "name": "Assess\nfairness",
                "tasks": [
                    "Check if protected\nattributes are used",
                    "Understand model\nlimitations",
                ],
                "info": [
                    "Statement on which\nfeatures are excluded",
                    "Disclaimer on\nmodel uncertainty",
                ],
            },
        ],
    },
}


def draw_usm(stakeholder_name, data, filename):
    """Draw a single User Story Map for one stakeholder."""
    activities = data["activities"]
    n_activities = len(activities)
    max_tasks = max(len(a["tasks"]) for a in activities)

    fig_width = 4.0 * n_activities + 1
    fig_height = 2.8 * (max_tasks + 2) + 1.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    color = data["color"]

    # Title
    ax.text(0.5, 0.97, f"User Story Map: {stakeholder_name}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=16, fontweight="bold", color=C_DARK)

    # Goal bar
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.03, 0.90), 0.94, 0.05, boxstyle="round,pad=0.01",
        facecolor=color, edgecolor=color, alpha=0.9,
        transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.925, f"Goal: {data['goal']}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    # Row labels
    row_labels = ["Activity", "Tasks", "Information\nNeeded"]
    row_y_starts = [0.82, 0.42, 0.02]  # approximate y positions

    col_width = 0.94 / n_activities
    col_start = 0.03

    for col_i, activity in enumerate(activities):
        x_center = col_start + col_i * col_width + col_width / 2
        x_left = col_start + col_i * col_width + 0.01
        box_w = col_width - 0.02

        # Activity box
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_left, 0.82), box_w, 0.06, boxstyle="round,pad=0.008",
            facecolor=color + "33", edgecolor=color, linewidth=1.5,
            transform=ax.transAxes, clip_on=False))
        ax.text(x_center, 0.85, activity["name"],
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, fontweight="bold", color=C_DARK)

        # Task boxes
        n_tasks = len(activity["tasks"])
        task_height = 0.06
        task_gap = 0.015
        task_start_y = 0.75

        for t_i, task in enumerate(activity["tasks"]):
            ty = task_start_y - t_i * (task_height + task_gap)
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_left, ty), box_w, task_height, boxstyle="round,pad=0.006",
                facecolor="white", edgecolor=C_GRAY, linewidth=0.8,
                transform=ax.transAxes, clip_on=False))
            ax.text(x_center, ty + task_height / 2, task,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7.5, color=C_DARK)

        # Info boxes
        info_start_y = 0.75 - max_tasks * (task_height + task_gap) - 0.04

        for i_i, info in enumerate(activity["info"]):
            iy = info_start_y - i_i * (task_height + task_gap)
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_left, iy), box_w, task_height, boxstyle="round,pad=0.006",
                facecolor=color + "15", edgecolor=color + "66", linewidth=0.8,
                transform=ax.transAxes, clip_on=False))
            ax.text(x_center, iy + task_height / 2, info,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color=C_DARK, style="italic")

    # Row labels on the left
    ax.text(0.01, 0.85, "Activities", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=C_GRAY, rotation=90)
    ax.text(0.01, 0.60, "Tasks", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=C_GRAY, rotation=90)

    info_label_y = 0.75 - max_tasks * 0.075 - 0.04 - 0.08
    ax.text(0.01, info_label_y, "Info\nNeeded", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=C_GRAY, rotation=90)

    # Horizontal separators
    sep_y1 = 0.77
    sep_y2 = 0.75 - max_tasks * 0.075 - 0.02
    ax.plot([0.03, 0.97], [sep_y1, sep_y1], color=C_GRAY, lw=0.5,
            ls="--", transform=ax.transAxes, clip_on=False)
    ax.plot([0.03, 0.97], [sep_y2, sep_y2], color=C_GRAY, lw=0.5,
            ls="--", transform=ax.transAxes, clip_on=False)

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def draw_combined_usm():
    """Draw all three USMs stacked vertically on one figure for the report."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 30), facecolor=BG)
    fig.suptitle("User Story Maps for All Stakeholders",
                 fontsize=20, fontweight="bold", color=C_DARK, y=0.995)

    for ax_i, (stakeholder, data) in enumerate(USM_DATA.items()):
        ax = axes[ax_i]
        ax.set_facecolor(BG)
        ax.axis("off")

        color = data["color"]
        activities = data["activities"]
        n_activities = len(activities)
        max_tasks = max(len(a["tasks"]) for a in activities)

        # Title
        ax.text(0.5, 0.97, f"{stakeholder}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=14, fontweight="bold", color=color)

        # Goal bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.03, 0.88), 0.94, 0.06, boxstyle="round,pad=0.01",
            facecolor=color, edgecolor=color, alpha=0.9,
            transform=ax.transAxes, clip_on=False))
        ax.text(0.5, 0.91, f"Goal: {data['goal']}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

        col_width = 0.94 / n_activities
        col_start = 0.03

        for col_i, activity in enumerate(activities):
            x_center = col_start + col_i * col_width + col_width / 2
            x_left = col_start + col_i * col_width + 0.005
            box_w = col_width - 0.01

            # Activity box
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_left, 0.80), box_w, 0.06, boxstyle="round,pad=0.008",
                facecolor=color + "33", edgecolor=color, linewidth=1.5,
                transform=ax.transAxes, clip_on=False))
            ax.text(x_center, 0.83, activity["name"],
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=C_DARK)

            # Task boxes
            task_h = 0.055
            task_gap = 0.012
            task_y = 0.73

            for t_i, task in enumerate(activity["tasks"]):
                ty = task_y - t_i * (task_h + task_gap)
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x_left, ty), box_w, task_h, boxstyle="round,pad=0.005",
                    facecolor="white", edgecolor=C_GRAY, linewidth=0.8,
                    transform=ax.transAxes, clip_on=False))
                ax.text(x_center, ty + task_h / 2, task,
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color=C_DARK)

            # Info boxes
            info_y = task_y - max_tasks * (task_h + task_gap) - 0.03

            for i_i, info in enumerate(activity["info"]):
                iy = info_y - i_i * (task_h + task_gap)
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x_left, iy), box_w, task_h, boxstyle="round,pad=0.005",
                    facecolor=color + "15", edgecolor=color + "66", linewidth=0.8,
                    transform=ax.transAxes, clip_on=False))
                ax.text(x_center, iy + task_h / 2, info,
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=6.5, color=C_DARK, style="italic")

        # Separators
        sep1 = 0.76
        sep2 = task_y - max_tasks * (task_h + task_gap) - 0.01
        ax.plot([0.03, 0.97], [sep1, sep1], color=C_GRAY, lw=0.5,
                ls="--", transform=ax.transAxes, clip_on=False)
        ax.plot([0.03, 0.97], [sep2, sep2], color=C_GRAY, lw=0.5,
                ls="--", transform=ax.transAxes, clip_on=False)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    path = OUTPUT_DIR / "user_story_maps_combined.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("Generating User Story Maps...")

    for stakeholder, data in USM_DATA.items():
        safe_name = stakeholder.lower().replace(" ", "_").replace("of_", "")
        draw_usm(stakeholder, data, f"usm_{safe_name}.png")

    draw_combined_usm()

    print("\nDone!")


if __name__ == "__main__":
    main()
