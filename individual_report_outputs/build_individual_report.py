"""Build Yankun Zhu's individual report with the complete relevant code appendix."""

from pathlib import Path
from math import ceil
from html import escape
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pymupdf
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/individual_predictions_overview_Yankun_Zhu.pdf"
APPENDIX_OUT = ROOT / "output/pdf/individual_predictions_code_appendix_Yankun_Zhu.pdf"
COMPLETE_OUT = ROOT / "output/pdf/individual_predictions_overview_Yankun_Zhu_with_appendix.pdf"
TEMP = ROOT / "tmp/pdfs/individual_report"
TEMP.mkdir(parents=True, exist_ok=True)
NAVY = colors.HexColor("#15354f")
BLUE = colors.HexColor("#225b7c")
PALE = colors.HexColor("#edf4f8")
GRAY = colors.HexColor("#566575")
FONT_ROOT = Path("/System/Library/Fonts/Supplemental")
for name, filename in (("Report", "Arial.ttf"), ("ReportBold", "Arial Bold.ttf"),
                       ("ReportItalic", "Arial Italic.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT / filename)))
pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold",
                            italic="ReportItalic", boldItalic="ReportBold")
styles = {
    "body": ParagraphStyle("body", fontName="Report", fontSize=10,
                           leading=13.5, spaceAfter=7, textColor=colors.HexColor("#1f2933")),
    "title": ParagraphStyle("title", fontName="ReportBold", fontSize=18,
                            leading=21, spaceAfter=9, textColor=NAVY),
    "heading": ParagraphStyle("heading", fontName="ReportBold", fontSize=13,
                              leading=16, spaceBefore=8, spaceAfter=7, textColor=NAVY),
    "sub": ParagraphStyle("sub", fontName="ReportBold", fontSize=10.4,
                          leading=14, spaceBefore=6, spaceAfter=5, textColor=BLUE),
    "note": ParagraphStyle("note", fontName="Report", fontSize=8.5,
                           leading=11.3, spaceAfter=6, textColor=GRAY),
    "table": ParagraphStyle("table", fontName="Report", fontSize=8.7,
                            leading=11.1, textColor=colors.HexColor("#1f2933")),
    "th": ParagraphStyle("th", fontName="ReportBold", fontSize=8.7,
                         leading=11.2, textColor=colors.white),
}
STORY = []


def p(text, style="body"):
    STORY.append(Paragraph(text, styles[style]))


def heading(text):
    p(text, "heading")


def table(headers, rows, widths, last_bold=False):
    data = [[Paragraph(str(v), styles["th"]) for v in headers]]
    for row in rows:
        data.append([Paragraph(str(v), styles["table"]) for v in row])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    rules = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#bacbd7")),
    ]
    if last_bold:
        rules.append(("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#dcecf4")))
    t.setStyle(TableStyle(rules))
    STORY.append(t)
    STORY.append(Spacer(1, 7))


def new_page():
    STORY.append(PageBreak())


def report_chrome(c, doc):
    c.saveState()
    c.setFont("Report", 8)
    c.setFillColor(GRAY)
    c.drawString(44, 766, "46-937 | Individual Predictions Overview")
    c.drawRightString(568, 766, "Yankun Zhu")
    c.setStrokeColor(colors.HexColor("#bacbd7"))
    c.line(44, 756, 568, 756)
    c.line(44, 39, 568, 39)
    c.drawString(44, 25, "Individual section | Main report")
    c.drawRightString(568, 25, str(doc.page))
    c.restoreState()


SOURCE_FILES = [
    ("evaluation_framework.py", "Data audit, train/validation split, mean baseline, exact sMAPE"),
    ("benchmark_models.py", "Common-split non-DL models and feedforward neural network"),
    ("oof_models.py", "Five-fold tree-family OOF predictions"),
    ("hard_target_oof.py", "Fixed lag features and target-transform experiments"),
    ("analyze_oof.py", "Lag baselines, target-specific selection, accounting identities"),
    ("locked_recipe_cv.py", "Frozen recipe on new CV seeds and shared reconciliation"),
    ("arcsinh_ablation_cv.py", "All-target raw/arcsinh ablation"),
    ("hard_representations_cv.py", "Margin, sign/magnitude and EBITDA residual experiments"),
    ("lightgbm_small_search.py", "Five controlled LightGBM configurations"),
    ("candidate_recipe_cv.py", "Candidate selection and reserved-seed comparison"),
    ("submission_code.py", "Self-contained final full-data prediction pipeline"),
    ("individual_report_outputs/report_uncertainty.py", "Paired bootstrap of stored holdout errors"),
]


def make_code_appendix():
    appendix = TEMP / "code_appendix.pdf"
    width, height = landscape(letter)
    code_x, top_y, leading = 74, 526, 9.6
    maximum_rows = 50
    c = canvas.Canvas(str(appendix), pagesize=(width, height), pageCompression=1)
    c.setTitle("Relevant code appendix - Yankun Zhu")
    c.setAuthor("Yankun Zhu")
    entries, page_number = [], 1
    for filename, purpose in SOURCE_FILES:
        lines = (ROOT / filename).read_text().splitlines()
        first = page_number
        rows_per_page = ceil(len(lines) / ceil(len(lines) / maximum_rows))
        for chunk_start in range(0, len(lines), rows_per_page):
            c.setFillColor(NAVY)
            c.setFont("ReportBold", 11)
            c.drawString(42, 575, "Code appendix | " + filename)
            c.setFont("Report", 8)
            c.setFillColor(GRAY)
            c.drawString(42, 560, purpose)
            c.setStrokeColor(colors.HexColor("#bacbd7"))
            c.line(42, 549, width - 42, 549)
            for j, line in enumerate(lines[chunk_start:chunk_start + rows_per_page]):
                y = top_y - j * leading
                c.setFont("Courier", 8.3)
                c.setFillColor(colors.HexColor("#7a8998"))
                c.drawRightString(61, y, f"{chunk_start + j + 1:03d}")
                c.setFillColor(BLUE if line.lstrip().startswith("#") else colors.HexColor("#1f2933"))
                c.drawString(code_x, y, line)
                if pdfmetrics.stringWidth(line, "Courier", 8.3) > width - 42 - code_x:
                    raise ValueError(f"Source line would be clipped: {filename}:{chunk_start+j+1}")
            c.setStrokeColor(colors.HexColor("#bacbd7"))
            c.line(42, 38, width - 42, 38)
            c.setFont("Report", 7.5)
            c.setFillColor(GRAY)
            c.drawString(42, 25, "Yankun Zhu | Complete source listing")
            c.drawRightString(width - 42, 25, f"Appendix page A{page_number}")
            c.showPage()
            page_number += 1
        entries.append((filename, purpose, first, page_number - 1, len(lines)))
    c.save()
    return appendix, entries


def make_report(entries):
    comparison = pd.read_csv(ROOT / "benchmark_outputs/comparison.csv")
    stats = json.loads((ROOT / "individual_report_outputs/uncertainty_summary.json").read_text())
    held = pd.read_csv(ROOT / "candidate_cv_314159/held_target_scores.csv")
    fold = pd.read_csv(ROOT / "candidate_cv_314159/held_fold_scores.csv")
    short = ["Total assets", "Total liabilities", "Stockholders' equity", "Gross profit",
             "Cost of revenues", "Revenues", "Operating income", "Operating expenses", "EBITDA"]
    targets = held.target.tolist()

    # Main report page 1: required Data Prep subsection.
    p("1.2 Group Member Predictions Overview", "title")
    p("<b>Yankun Zhu</b> | Individual prediction approach", "body")
    p("I selected a target-specific LightGBM pipeline with signed target transformations, "
      "lag features and accounting-derived blends. The fixed candidate achieved mean out-of-fold "
      "(OOF) sMAPE of <b>16.5719%</b> on five shuffled folds with seed 314159. In an earlier "
      "common 80/20 benchmark, LightGBM achieved 29.3330%, the feedforward neural network "
      "34.4984%, and the required mean baseline 136.1519%. These are two separate evaluation "
      "stages; none of these numbers is an official test-set score.")
    heading("1.2.1 Data Prep")
    p("<b>Data and targets.</b> The training file contains 100,000 rows and 212 columns; the "
      "individual test file contains 50,000 rows and 203 columns. After excluding the identifier "
      "and all nine Q0 response columns, there are 202 predictors: 198 numeric and four categorical "
      "(industry, sector, financial currency and recommendation category). The responses are total "
      "assets, total liabilities, stockholders' equity, gross profit, cost of revenues, revenues, "
      "operating income, operating expenses and EBITDA. Q1-Q10 describe the ten preceding quarters. "
      "The two files have identical predictor names and order. <b>Id is never a predictor</b>; it is "
      "retained only for alignment and submission.")
    p("<b>Missing values and encoding.</b> The audit found no parsed missing cells, non-finite "
      "numeric values or missing targets. Literal category placeholders were retained as category "
      "levels: 1,344 training records in each of industry and sector, and 1,232 in each of financial "
      "currency and recommendation category. The pipeline nevertheless fits median numeric "
      "imputation and most-frequent categorical imputation on training rows only, then dense "
      "one-hot encoding with unseen validation/test categories ignored. This produced 332 encoded "
      "features in the original holdout benchmark. No validation/test statistics determine "
      "imputation, category vocabularies or scaling.")
    p("<b>Scale and signs.</b> Financial targets are highly skewed: total assets have a training "
      "median of about 0.823 billion versus a mean of 5.285 billion. Operating income and EBITDA "
      "are negative in 79.542% and 78.018% of training rows. Even assets include synthetic negative "
      "values. I retained these observations and the fractional fiscal-year-end indicators instead "
      "of dropping, winsorizing or binarizing them. Tree inputs remain on their original numeric "
      "scale; Ridge and the neural network additionally use arcsinh-transformed numeric predictors "
      "followed by training-only standardization.")
    p("<b>Target transformation.</b> For transformed regression, the scale is the training-only "
      "median absolute target, replaced by 1 only if it is nonpositive. I fit to "
      "z = arcsinh(y / s) and invert with y-hat = s sinh(z-hat). This compresses long tails while "
      "preserving negative values and values near zero. Each CV fold estimates its own scale; "
      "the final fit estimates it from all training labels.")

    # Page 2: remaining preparation and evaluation design.
    new_page()
    heading("Data Prep: lag features and validation assumptions")
    p("<b>Target-specific lag features.</b> For gross profit, operating income and EBITDA, I "
      "append the same eleven row-wise features: Q1 value; Q1 minus Q2; Q1 sign; Q2 sign; the "
      "product of those signs; Q1-Q4 mean, median and population standard deviation; a Q1-Q4 "
      "linear slope; Q1-Q10 mean; and a Q1-Q10 slope. For a K-quarter slope I center the recency "
      "coordinates (K-1, ..., 0), then divide the weighted sum of values by the sum of squared "
      "centered coordinates. A positive slope means growth toward Q1. These features use only "
      "past-quarter values from the same row and require no target-based fitting.")
    p("<b>Leakage and structure.</b> The dictionary places Q0 at the latest reported quarter; "
      "the file contains neither a usable company grouping identifier nor varying row timestamps. "
      "There are no exact duplicate predictor rows or historical-feature rows. A shuffled row "
      "split is therefore a practical comparison for this supplied dataset, but cannot establish "
      "future-quarter or unseen-company generalization. Synthetic rows could still share an "
      "underlying company. Same-quarter metadata such as totalRevenue and ebitda were retained "
      "as provided features; their availability at a real forecasting date would need verification. "
      "The audit found correlation 0.9655 between totalRevenue and Q0 revenues. Excluding Id "
      "and Q0 labels prevents direct label inclusion, but does not resolve that deployment concern.")
    heading("1.2.2 Method Selection")
    p("<b>Common comparison.</b> I used a fixed random 80/20 split with seed 42: 80,000 training "
      "rows and the same 20,000 validation rows for every initial model. I saved validation "
      "predictions with row index and Id to verify alignment. All preprocessing was fit on the "
      "80,000 training rows; every method predicted all nine targets and was evaluated on the "
      "original response scale.")
    p("<b>Metric.</b> For target j, the assignment score is:")
    p("sMAPE<sub>j</sub> = (100 / n) &Sigma;<sub>i</sub> "
      "|y<sub>ij</sub> - y-hat<sub>ij</sub>| / "
      "[0.5 (|y<sub>ij</sub>| + |y-hat<sub>ij</sub>|)]", "sub")
    p("The overall score is the unweighted average of the nine target scores; lower is better. "
      "I use zero error only when both actual and predicted values are zero. The denominator "
      "makes relative scale important: opposite signs give the maximum 200% error, and observations "
      "near zero can be difficult. The exact same scoring implementation was used throughout. "
      "Improvement over baseline is 100 times (baseline score minus model score) divided by "
      "baseline score.")
    p("<b>Separation from test data.</b> Test data were initially inspected for schema and "
      "missing values, then used for prediction only after the recipe was fixed. Test predictions "
      "and their distribution diagnostics did not select models, transformations or weights.")

    # Page 3: baseline, non-DL, and justified DL architecture.
    new_page()
    heading("Method Selection: approaches considered")
    p("<b>Naive baseline.</b> Each response is predicted by its mean from the training portion, "
      "ignoring every predictor. This is the rubric's required reference rather than a fitted "
      "relationship between features and targets. Long tails pull the means away from a typical "
      "firm, and a constant sign cannot adapt to firm-level profit variation.")
    p("<b>Non-deep-learning methods.</b> Ridge provides a regularized linear reference. Random "
      "Forest averages bootstrap-grown trees; ExtraTrees randomizes tree thresholds more strongly. "
      "Both used multi-output regressors. HistGradientBoosting and LightGBM fit one boosted-tree "
      "regressor per target. These models are appropriate for tabular mixed-scale financial data "
      "because trees can capture nonlinear persistence and interactions without treating monetary "
      "scale as a distance. Ridge was retained as a diagnostic, despite its poor initial result.")
    table(["Method", "Fixed initial configuration"], [
        ["Ridge", "Penalty 100; iterative least-squares solver; transformed/scaled numeric inputs"],
        ["Random Forest / ExtraTrees", "48 trees; maximum depth 14; minimum leaf size 8; feature fraction 0.7"],
        ["HistGradientBoosting", "80 iterations; 31 leaves; 128 bins; L2 penalty 1; early stopping disabled"],
        ["LightGBM", "120 trees; learning rate 0.08; 31 leaves; minimum leaf size 40; feature fraction 0.8"],
    ], [140, 384])
    p("<b>Deep learning: feedforward multi-output network.</b> The network is a multilayer "
      "perceptron with <b>332 inputs, hidden layers of 128 and 64 ReLU units, and nine unrestricted "
      "linear outputs</b> (about 51,465 trainable parameters). Two moderate hidden layers give "
      "nonlinear capacity without making the initial experiment large. Shared hidden layers let "
      "related financial targets use common representations. A feedforward network fits the "
      "supplied tabular rows; the historical quarters are features, while an unrestricted output "
      "allows losses as well as profits. It was randomly initialized; no pretrained model or "
      "transfer learning was used.")
    p("The network uses the same imputation and category encoding as the other models. Numeric "
      "inputs are arcsinh-transformed and standardized. Each target is arcsinh-transformed using "
      "its training median absolute value, then centered and standardized using training-only "
      "moments. Predictions undo standardization and then apply sinh. Training uses AdamW "
      "(learning rate 0.001, weight decay 0.0001), SmoothL1 loss, batches of 2,048, and 12 epochs "
      "with seed 42. Target standardization balances the nine outputs and SmoothL1 reduces "
      "sensitivity to large errors in transformed space; neither directly minimizes sMAPE.")
    p("<b>Initial tuning policy.</b> These were fixed, modest configurations, not a large "
      "hyperparameter search. The network had one architecture and a fixed epoch budget, with "
      "no validation-based early stopping, convergence analysis or architecture search. The "
      "neural-network experiment therefore provides a limited initial benchmark. Its result "
      "evaluates this particular network pipeline and does not establish that all deep learning "
      "methods are inferior. The final submission contains the selected tree-based method, "
      "while this report documents the neural network that was actually considered.")

    # Page 4: direct rubric comparison, per-target evidence, uncertainty.
    new_page()
    heading("Method Selection: common-split comparison results")
    names = {"mean": "Naive mean", "lightgbm": "LightGBM (raw targets)",
             "neural_network": "Feedforward network", "random_forest": "Random Forest",
             "extra_trees": "ExtraTrees", "ridge": "Ridge",
             "hist_gradient_boosting": "HistGradientBoosting"}
    rows = []
    for _, row in comparison.iterrows():
        ci = stats["models"].get(row.model, {}).get("conditional_bootstrap_95_interval_percent")
        rows.append([names[row.model], f"{row.mean_smape_percent:.4f}",
                     f"{row.improvement_vs_mean_percent:.2f}%",
                     f"{row.training_seconds:.2f}",
                     f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "--"])
    table(["Model", "Mean sMAPE (%)", "Improvement vs mean", "Train time (s)", "95% interval (%)"],
          rows, [161, 83, 95, 80, 105])
    p("Training times include preprocessing and fitting in the recorded local environment; "
      "they are illustrative rather than hardware-neutral comparisons. Intervals shown are "
      "conditional percentile intervals from 2,000 shared row-bootstrap resamples of the stored "
      "20,000 validation predictions (seed 20261006); no model was retrained.", "note")
    by_model = comparison.set_index("model")
    rows = [[name, f"{by_model.loc['mean', target]:.2f}",
             f"{by_model.loc['lightgbm', target]:.2f}",
             f"{by_model.loc['neural_network', target]:.2f}"]
            for name, target in zip(short, targets)]
    table(["Target", "Naive mean (%)", "LightGBM (%)", "Network (%)"],
          rows, [188, 112, 112, 112])
    p("<b>Decision from the common benchmark.</b> LightGBM had the lowest overall score "
      "(29.3330%) and was lower than the network for every target. The network still improved "
      "on the naive mean by 74.66%. The paired network-minus-LightGBM gap was <b>5.1654 "
      "percentage points</b>, with a 95% row-bootstrap interval of <b>4.9631-5.3693</b>. "
      "Random Forest was especially strong on assets, liabilities and equity, despite a worse "
      "overall mean, motivating target-specific rather than uniform model selection.")
    p("These intervals condition on the fitted models and exchangeable validation rows. They "
      "do not account for training randomness, model selection or hidden company clusters. "
      "The benchmark compares complete pipelines: input/target transforms differ by model, "
      "so it does not isolate architecture alone. The later optimized OOF score is reported "
      "separately, not as a new common-split neural-network comparison.")

    # Page 5: OOF evidence and small search, no claim of independent selection performance.
    new_page()
    heading("Method Selection: OOF refinement and controlled tuning")
    p("I next generated aligned five-fold OOF predictions (shuffle, seed 42) for Random Forest, "
      "LightGBM and HistGradientBoosting; each row was predicted without using that row to fit "
      "its model. Their base mean scores were 32.9352%, 29.3010% and 31.2856%, respectively. "
      "Q1 persistence, Q1-Q2/Q1-Q4 means, and a 0.60/0.25/0.10/0.05 recency average were "
      "also checked. Q1 persistence scored 47.3769% overall and 18.0507% for cost of revenues, "
      "showing useful persistence but poor robustness for profit signs.")
    p("RF/LightGBM and lag weights were tried target by target on these OOF predictions, then "
      "accounting-derived alternatives were compared. The resulting original recipe scored "
      "21.2509% on the same selection OOF labels, an optimistic post-selection estimate. I "
      "froze it and reran it without changing choices or weights on seed 2026, obtaining "
      "21.1944%. Both sets still contain the same training rows.")
    p("<b>Transformation ablation.</b> On the seed-2026 folds, raw and arcsinh LightGBM used "
      "identical base features, preprocessing and parameters. All nine targets improved in "
      "all five folds. Mean sMAPE fell from 29.1420% to 19.1005% for these single-model pipelines.")
    arc = pd.read_csv(ROOT / "ablation_cv_2026/target_scores.csv")
    table(["Target", "Raw (%)", "Arcsinh (%)"],
          [[name, f"{row.raw_smape_percent:.2f}", f"{row.arcsinh_smape_percent:.2f}"]
           for name, (_, row) in zip(short, arc.iterrows())], [288, 118, 118])
    p("<b>Hard-target representations.</b> With the same target-specific lag features and "
      "seed-2026 folds, operating-income sMAPE was 51.7784% for direct arcsinh, 51.0429% "
      "for a modeled revenue margin, and <b>43.7901%</b> for sign plus magnitude. EBITDA "
      "was 50.3554%, 50.1255% and <b>42.9105%</b>, respectively; modeling EBITDA minus "
      "operating income gave 48.6407%. Margin and residual reconstruction used other targets' "
      "OOF predictions, not validation-row labels. Sign plus magnitude won in all five folds; "
      "its sign accuracy was 92.967% for operating income and 93.317% for EBITDA.")
    p("<b>Limited search.</b> Five configurations were compared only for arcsinh-beneficial "
      "targets: the 31-leaf/40-minimum baseline; 15/80; 63/20; 31/40 with 180 trees at "
      "learning rate 0.05; and the baseline with L1 loss. All others used 120 trees at rate "
      "0.08 and feature fraction 0.8. A configuration needed at least 0.15 percentage-point "
      "gain and three of five fold wins. Candidate direct substitutions then needed at least "
      "0.05 point improvement in the complete recipe and three fold wins, applied in a fixed "
      "one-pass target order. Selected changes actually won on all five development folds. "
      "Accounting/lag blend weights stayed frozen; seed 314159 was reserved before this selection.")

    # Page 6: exact final method and nonrecursive accounting equations.
    new_page()
    heading("Method Selection: fixed individual prediction recipe")
    p("The final approach uses six direct LightGBM regressions and two classifier/regressor "
      "pairs, producing eight direct financial outputs. Revenue is derived from gross profit "
      "and cost, so no independent revenue model is needed in the final output. All models use "
      "120 trees, learning rate 0.08 and feature fraction 0.8; the final full-data random seed "
      "is 42. The table specifies the target-dependent differences.")
    table(["Direct output", "Model / loss", "Leaves / min. leaf", "Extra lags"], [
        ["Assets, liabilities, equity", "Arcsinh regression; squared-error loss", "63 / 20", "No"],
        ["Gross profit", "Arcsinh regression; L1 loss", "31 / 40", "Yes"],
        ["Cost of revenues", "Arcsinh regression; L1 loss", "31 / 40", "No"],
        ["Operating expenses", "Arcsinh regression; squared-error loss", "63 / 20", "No"],
        ["Operating income, EBITDA", "Binary sign classifier + arcsinh magnitude regression", "31 / 40 (both)", "Yes"],
    ], [153, 206, 104, 61])
    p("For each sign/magnitude pair, the classifier fits the indicator that y is positive. "
      "The regression fits arcsinh(|y| / s), using the median absolute training target for s. "
      "The inverse-transformed magnitude is positive when the predicted positive probability "
      "is at least <b>0.5</b>, and negative otherwise. The threshold is fixed, not optimized "
      "on the test predictions.")
    p("<b>Accounting-derived final outputs.</b> Let D denote the unmodified direct predictions "
      "and let A, L, E, GP, C, OI, OE and B denote assets, liabilities, equity, gross profit, "
      "cost, operating income, operating expenses and EBITDA. Every expression below uses "
      "D values; already adjusted final outputs are never fed into another expression.")
    table(["Final target", "Frozen expression"], [
        ["Assets", "D<sub>A</sub>"],
        ["Liabilities", "0.50 D<sub>L</sub> + 0.50 (D<sub>A</sub> - D<sub>E</sub>)"],
        ["Equity", "D<sub>A</sub> - D<sub>L</sub>"],
        ["Gross profit / cost", "D<sub>GP</sub> / D<sub>C</sub>, respectively"],
        ["Revenues", "max(0, D<sub>GP</sub> + D<sub>C</sub>)"],
        ["Operating income", "0.50 D<sub>OI</sub> + 0.50 (D<sub>GP</sub> - D<sub>OE</sub>)"],
        ["Operating expenses", "max[0, 0.25 D<sub>OE</sub> + 0.75 (D<sub>GP</sub> - D<sub>OI</sub>)]"],
        ["EBITDA", "D<sub>B</sub>"],
    ], [149, 375])
    p("The training identities assets = liabilities + equity and operating income = gross "
      "profit - expenses hold to floating-point precision. Revenue = gross profit + cost is "
      "approximate, with identity sMAPE 0.7972%. This supports accounting-derived candidates, "
      "but noisy model predictions still require OOF evaluation. These are target-wise blends, "
      "not a joint constrained projection that guarantees every identity among final outputs. "
      "Nonnegative guards apply only to revenue and operating expenses, which are nonnegative "
      "in the training targets; other signed outputs are retained.")

    # Page 7: final fixed-seed results, uncertainty and limits.
    new_page()
    heading("Method Selection: fixed-recipe validation and decision")
    p("The candidate scored 16.5413% on seed-2026 development folds. After the candidate "
      "choices were fixed, I retrained both it and the original recipe on the same new "
      "five-fold partition with seed 314159, with no reselection on that partition. The "
      "candidate's mean was <b>16.5719%</b>, versus <b>21.2064%</b> for the original recipe: "
      "a <b>4.6345 percentage-point</b> reduction (21.85% relative reduction). All nine targets "
      "improved in the pooled OOF comparison.")
    table(["Target", "Original recipe (%)", "Fixed candidate (%)"],
          [[name, f"{row.locked_smape_percent:.3f}", f"{row.candidate_smape_percent:.3f}"]
           for name, (_, row) in zip(short, held.iterrows())], [238, 143, 143])
    table(["Fold", "Original (%)", "Candidate (%)", "Gain (points)"],
          [[str(int(row.fold)), f"{row.locked_mean_smape_percent:.4f}",
            f"{row.candidate_mean_smape_percent:.4f}", f"{row.improvement_pp:.4f}"]
           for _, row in fold.iterrows()], [64, 152, 152, 156])
    p("The candidate improves all five folds; its sample fold standard deviation is "
      "<b>0.0798 points</b> and its range is 16.4828%-16.6768%. This is descriptive fold "
      "variability, not a confidence interval based on independent experiments. A new seed "
      "still uses the same 100,000 labeled rows previously used in model development; "
      "overlapping training folds and selection can bias performance estimates. It establishes "
      "stability to this repartition, not fully independent generalization.")
    p("<b>Why I chose this approach.</b> The common-split evidence favored boosted trees over "
      "the tested neural network; arcsinh, sign/magnitude decomposition and target-specific "
      "accounting blends then reduced the remaining financial-scale and sign errors consistently. "
      "Operating income (42.666%) and EBITDA (43.085%) remain the hardest targets. The neural "
      "network was not rerun on these final folds, so I do not interpret 16.5719% as a "
      "head-to-head final-CV comparison against it.")
    p("I refit the frozen method on all 100,000 training rows to produce 50,000 individual "
      "test predictions. The self-contained submission script reproduced the existing final "
      "CSV byte for byte, preserved Id order, and passed finite-value and output-schema checks. "
      "This report does not supply a group-method comparison or a leaderboard/test score.")

    # Page 8: evidence and appendix guide, no source code in the main report.
    new_page()
    heading("Evidence, reproducibility and code appendix")
    p("Results above are drawn from the saved project artifacts, rather than new model fits "
      "performed for this report. The only added statistical calculation resamples stored "
      "holdout errors to describe uncertainty; it does not alter any model, threshold or weight.")
    table(["Evidence", "Saved artifact(s)"], [
        ["Data audit and target distributions", "audit_outputs/summary.json; target_distributions.csv; same_quarter_metadata.csv"],
        ["Common-split three-approach comparison", "benchmark_outputs/comparison.csv; validation_truth.csv; validation_predictions/"],
        ["Conditional paired uncertainty", "individual_report_outputs/uncertainty_summary.json"],
        ["OOF selection and accounting checks", "oof_outputs/base_model_scores.csv; lag_scores.csv; identity_checks.csv; target_recipes.csv"],
        ["Arcsinh / hard-target development", "ablation_cv_2026/target_scores.csv; representation_cv_2026/"],
        ["Frozen candidate and new-fold results", "candidate_cv_314159/selected_substitutions.json; held_target_scores.csv; held_fold_scores.csv"],
        ["Final prediction artifact", "final_submission.csv; submission_code_output.csv (identical)"],
    ], [207, 317])
    p("Repository: <link href='https://github.com/Bumblebee0427/Deep-Learning-Financial-Performance-Prediction' "
      "color='#225b7c'>Bumblebee0427/Deep-Learning-Financial-Performance-Prediction</link>. "
      "The separately supplied code appendix contains the complete relevant source listings, including the "
      "evaluation metric, non-DL benchmark, neural network, OOF selection and final pipeline. "
      "The narrative report above contains no source-code listing.", "note")
    table(["Complete source listing", "Appendix pages"],
          [[escape(filename), f"A{first}-A{last}"] for filename, _, first, last, _ in entries],
          [425, 99])
    report_path = TEMP / "main_report.pdf"
    doc = SimpleDocTemplate(str(report_path), pagesize=letter,
                            leftMargin=44, rightMargin=44,
                            topMargin=51, bottomMargin=53,
                            title="Group Member Predictions Overview - Yankun Zhu",
                            author="Yankun Zhu")
    doc.build(STORY, onFirstPage=report_chrome, onLaterPages=report_chrome)
    return report_path


def main():
    appendix, entries = make_code_appendix()
    report = make_report(entries)
    writer = PdfWriter()
    writer.append(str(report))
    writer.append(str(appendix))
    main_pages = len(PdfReader(str(report)).pages)
    writer.add_outline_item("Individual Predictions Overview - Yankun Zhu", 0)
    writer.add_outline_item("Data Prep", 0)
    writer.add_outline_item("Method Selection", 1)
    parent = writer.add_outline_item("Relevant code appendix", main_pages)
    for filename, _, first, _, _ in entries:
        writer.add_outline_item(filename, main_pages + first - 1, parent=parent)
    writer.add_metadata({"/Title": "Group Member Predictions Overview - Yankun Zhu",
                         "/Author": "Yankun Zhu", "/Subject": "Individual report and relevant code appendix"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report, OUT)
    shutil.copyfile(appendix, APPENDIX_OUT)
    with COMPLETE_OUT.open("wb") as f:
        writer.write(f)
    manifest = {filename: {"sha256": hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(),
                           "lines": lines, "appendix_pages": [first, last]}
                for filename, _, first, last, lines in entries}
    (ROOT / "individual_report_outputs/source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("main report pages:", len(PdfReader(str(report)).pages))
    print("appendix pages:", len(PdfReader(str(appendix)).pages))
    print("main report:", OUT)
    print("separate appendix:", APPENDIX_OUT)
    print("combined submission:", COMPLETE_OUT)


if __name__ == "__main__":
    main()
