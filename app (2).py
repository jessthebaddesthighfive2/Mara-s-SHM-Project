import io
import math
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Hispanic Heritage Historical Data Explorer
# Student display name can be changed in the sidebar.
st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", layout="wide")

COUNTRIES = {
    "Mexico": "MEX",
    "Brazil (Latin American comparison; not Spanish-speaking Hispanic)": "BRA",
    "Argentina": "ARG",
    "Colombia": "COL",
    "Chile": "CHL",
    "Peru": "PER",
    "Venezuela": "VEN",
    "Ecuador": "ECU",
    "Uruguay": "URY",
}
INDICATORS = {
    "Life expectancy at birth": {
        "code": "SP.DYN.LE00.IN", "units": "years",
        "definition": "Life expectancy at birth, total population."
    },
    "Unemployment rate (national estimate)": {
        "code": "SL.UEM.TOTL.NE.ZS", "units": "% of total labor force",
        "definition": "National-estimate unemployment rate."
    },
    "GDP per capita": {
        "code": "NY.GDP.PCAP.CD", "units": "current US$ per person",
        "definition": "Economic output per person; not average wealth or household income."
    },
    "Crude birth rate": {
        "code": "SP.DYN.CBRT.IN", "units": "births per 1,000 people",
        "definition": "Annual crude birth rate, not a count of births."
    },
    "Net migration": {
        "code": "SM.POP.NETM", "units": "people",
        "definition": "Immigration minus emigration for the source period."
    },
    "Intentional homicide rate": {
        "code": "VC.IHR.PSRC.P5", "units": "intentional homicides per 100,000 people",
        "definition": "Intentional homicide rate per 100,000 people."
    },
    "Consumer price inflation": {
        "code": "FP.CPI.TOTL.ZG", "units": "% annual change",
        "definition": "Annual percentage change in consumer prices."
    },
    "Central government debt": {
        "code": "GC.DOD.TOTL.GD.ZS", "units": "% of GDP",
        "definition": "Central government debt as a share of GDP, not all public-sector debt."
    },
    "Total reserves, including gold": {
        "code": "FI.RES.TOTL.CD", "units": "current US$",
        "definition": "Total reserves including gold."
    },
    "External balance: goods and services": {
        "code": "NE.RSB.GNFS.ZS", "units": "% of GDP",
        "definition": "External balance on goods and services; not the current account."
    },
}

def fetch_world_bank(country_code, indicator_code):
    """Retrieve genuine annual data from the World Bank API for the target window."""
    current_year = date.today().year
    start_year, end_year = current_year - 70, current_year - 1
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/"
        f"{indicator_code}?date={start_year}:{end_year}&format=json&per_page=200"
    )
    try:
        response = requests.get(url, timeout=18)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["Year", "Value"]), f"No valid data response returned by the World Bank API."
        rows = []
        for item in payload[1]:
            if item.get("value") is not None and item.get("date") is not None:
                try:
                    rows.append({"Year": int(item["date"]), "Value": float(item["value"])})
                except (ValueError, TypeError):
                    continue
        df = pd.DataFrame(rows, columns=["Year", "Value"]).drop_duplicates("Year").sort_values("Year")
        if df.empty:
            return df, "The official API returned no non-missing observations for this series and period."
        return df.reset_index(drop=True), None
    except requests.Timeout:
        return pd.DataFrame(columns=["Year", "Value"]), "The World Bank API request timed out. Please retry."
    except (requests.RequestException, ValueError) as exc:
        return pd.DataFrame(columns=["Year", "Value"]), f"Could not retrieve official data: {exc}"

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(country_code, indicator_code):
    return fetch_world_bank(country_code, indicator_code)

def validate_working_data(df, indicator_name):
    clean = df.copy()
    clean["Year"] = pd.to_numeric(clean["Year"], errors="coerce")
    clean["Value"] = pd.to_numeric(clean["Value"], errors="coerce")
    errors = []
    if clean["Year"].isna().any():
        errors.append("Rows with missing or non-numeric years were removed.")
    if clean["Value"].isna().any():
        errors.append("Rows with missing or non-numeric values were removed.")
    clean = clean.dropna(subset=["Year", "Value"])
    integer_year = np.isclose(clean["Year"], np.round(clean["Year"]))
    if not integer_year.all():
        errors.append("Rows with non-integer years were removed.")
        clean = clean.loc[integer_year]
    clean["Year"] = clean["Year"].astype(int)
    if clean["Year"].duplicated().any():
        errors.append("Duplicate years were rejected; only the first occurrence was kept.")
        clean = clean.drop_duplicates("Year", keep="first")
    if ((clean["Year"] < 1900) | (clean["Year"] > date.today().year)).any():
        errors.append("Years outside 1900 through the current year were removed.")
        clean = clean[(clean["Year"] >= 1900) & (clean["Year"] <= date.today().year)]
    if indicator_name == "Life expectancy at birth" or indicator_name == "Crude birth rate":
        invalid = clean["Value"] < 0
        if invalid.any():
            errors.append("Negative values are not valid for this indicator and were removed.")
            clean = clean.loc[~invalid]
    if indicator_name == "Unemployment rate (national estimate)" and ((clean["Value"] < 0) | (clean["Value"] > 100)).any():
        errors.append("Unemployment percentages outside 0–100 were removed.")
        clean = clean[(clean["Value"] >= 0) & (clean["Value"] <= 100)]
    return clean.sort_values("Year").reset_index(drop=True), errors

def fit_polynomial(df, degree):
    if df["Year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct years.")
    base_year = int(df["Year"].min())
    t = (df["Year"].to_numpy(dtype=float) - base_year)
    y = df["Value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    mae = float(np.mean(np.abs(y - pred)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return base_year, coeff, pred, mae, r2

def equation_text(coeff, base_year):
    terms = []
    degree = len(coeff) - 1
    for i, c in enumerate(coeff):
        power = degree - i
        if abs(c) < 1e-14:
            continue
        if power == 0:
            terms.append(f"{c:.8g}")
        elif power == 1:
            terms.append(f"{c:.8g}t")
        else:
            terms.append(f"{c:.8g}t^{power}")
    return "f(t) = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0") + f", where t = year − {base_year}"

def create_excel(summary, data, results):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working Data", index=False)
        if results is not None and not results.empty:
            results.to_excel(writer, sheet_name="Calculated Years", index=False)
    buffer.seek(0)
    return buffer.getvalue()

def create_pdf(title, summary, data, chart_bytes, results):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=0.45*inch,
                            leftMargin=0.45*inch, topMargin=0.45*inch, bottomMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for key, value in summary.items():
        story.append(Paragraph(f"<b>{key}:</b> {str(value)}", styles["BodyText"]))
    story.append(Spacer(1, 10))
    if chart_bytes:
        story.append(Image(io.BytesIO(chart_bytes), width=8.8*inch, height=4.1*inch))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Working data preview", styles["Heading2"]))
    preview = data.head(15).round(6)
    table_data = [list(preview.columns)] + preview.astype(str).values.tolist()
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(tbl)
    if results is not None and not results.empty:
        story += [Spacer(1, 8), Paragraph("Calculated-year results", styles["Heading2"])]
        rt = Table([list(results.columns)] + results.astype(str).values.tolist(), repeatRows=1)
        rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(rt)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Persistent student display name and calculated results.
if "display_name" not in st.session_state:
    st.session_state.display_name = "Amarachi Onwo"
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "single_result" not in st.session_state:
    st.session_state.single_result = None

st.sidebar.title("Project controls")
st.sidebar.text_input("Student name (shown on the app)", key="display_name")
st.sidebar.caption("Your name appears in the app header and can appear in the shared link preview only if your hosting platform supports custom metadata. The URL itself is assigned by Streamlit.")
country_label = st.sidebar.selectbox("Country", list(COUNTRIES.keys()), index=0)
country_code = COUNTRIES[country_label]
indicator_name = st.sidebar.selectbox("Indicator", list(INDICATORS.keys()), index=0)
indicator = INDICATORS[indicator_name]
degree = st.sidebar.slider("Polynomial degree", 3, 8, 3)
sample_increment = st.sidebar.slider("Model marker sample increment (years)", 1, 10, 1)
horizon = st.sidebar.slider("Forward display horizon (years)", 0, 50, 5)
comparison_labels = st.sidebar.multiselect(
    "Compare with same indicator", [c for c in COUNTRIES if c != country_label],
    default=["Colombia"] if country_label == "Mexico" else []
)
printer_mode = st.sidebar.toggle("Printer-friendly view", value=False)

st.title("Hispanic Heritage Historical Data Explorer")
st.subheader(f"Created by {st.session_state.display_name.strip() or 'Student'}")
st.write("Explore official historical indicators and fit polynomial models. Data describe measured trends; they do not, by themselves, prove why a change happened.")
st.info("Brazil is included as a Latin American comparison; it is not a Spanish-speaking Hispanic country.")

st.markdown(f"**Featured question:** How has **{indicator_name.lower()}** changed over time in **{country_label}**, and how does it compare with selected countries?")
st.markdown(f"**Indicator definition:** {indicator['definition']}  \n**Units:** {indicator['units']}  \n**Official source:** [World Bank indicator page](https://data.worldbank.org/indicator/{indicator['code']})  \n**Indicator code:** `{indicator['code']}`  \n**Retrieved on:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with st.spinner("Retrieving official World Bank historical data..."):
    source_df, fetch_error = cached_fetch(country_code, indicator["code"])
if fetch_error:
    st.error(fetch_error)
    st.caption("No invented or random replacement data are used. Try again later or choose another indicator.")
    st.stop()

original_df = source_df.copy()
st.caption(f"Available coverage: {int(original_df.Year.min())}–{int(original_df.Year.max())}; {len(original_df)} observations. The target API window is {date.today().year-70}–{date.today().year-1}, but actual coverage varies by series.")

if "selection_signature" not in st.session_state:
    st.session_state.selection_signature = None
signature = (country_code, indicator["code"], degree)
if st.session_state.selection_signature != signature:
    st.session_state.batch_results = None
    st.session_state.single_result = None
    st.session_state.selection_signature = signature

st.subheader("Working data table")
st.write("Edit values below to explore a user-modified scenario. Reset to source restores the official values.")
edited_df = st.data_editor(original_df, num_rows="dynamic", use_container_width=True, key=f"editor_{country_code}_{indicator['code']}")
col_reset, col_status = st.columns([1, 3])
with col_reset:
    if st.button("Reset to source"):
        st.rerun()
with col_status:
    st.caption("If you change a value, year, or row, all analysis and downloads use the validated working table.")
working_df, validation_messages = validate_working_data(edited_df, indicator_name)
is_edited = not working_df.equals(original_df.reset_index(drop=True))
if is_edited:
    st.warning("User-modified scenario: working data differ from the official source table.")
for msg in validation_messages:
    st.warning(msg)
if len(working_df) < degree + 1 or working_df.Year.nunique() < degree + 1:
    st.error(f"Not enough distinct valid years for degree {degree}. Need at least {degree + 1}.")
    st.stop()

try:
    base_year, coeff, fitted_values, mae, r2 = fit_polynomial(working_df, degree)
except (ValueError, np.linalg.LinAlgError) as exc:
    st.error(f"Could not fit the selected model: {exc}")
    st.stop()

working_df = working_df.copy()
working_df["Fitted value"] = fitted_values
working_df["Residual"] = working_df["Value"] - fitted_values
equation = equation_text(coeff, base_year)
min_year, max_year = int(working_df.Year.min()), int(working_df.Year.max())
mean_value = float(working_df.Value.mean())

st.subheader("Model summary")
metric_cols = st.columns(4)
metric_cols[0].metric("Observations", len(working_df))
metric_cols[1].metric("R² (in-sample)", f"{r2:.5f}" if math.isfinite(r2) else "undefined")
metric_cols[2].metric("MAE", f"{mae:.5g} {indicator['units']}")
metric_cols[3].metric("Mean", f"{mean_value:.6g} {indicator['units']}")
st.write(f"**Base year:** {base_year}  \n**Degree:** {degree}  \n**Fitted domain:** {min_year}–{max_year}  \n**Equation:** `{equation}`")
st.caption("R² is an in-sample fit measure, not proof of forecast accuracy. Residual/MAE summaries do not provide validated prediction intervals.")

# Chart
display_end = max_year + horizon
smooth_years = np.linspace(min_year, display_end, max(300, (display_end-min_year+1)*2))
smooth_t = smooth_years - base_year
smooth_values = np.polyval(coeff, smooth_t)
sample_years = np.arange(min_year, display_end + 1, sample_increment)
sample_values = np.polyval(coeff, sample_years - base_year)
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.scatter(working_df.Year, working_df.Value, label="Observed / working data", zorder=3)
historical_mask = smooth_years <= max_year
future_mask = smooth_years > max_year
ax.plot(smooth_years[historical_mask], smooth_values[historical_mask], label="Fitted curve (historical range)")
if future_mask.any():
    ax.plot(smooth_years[future_mask], smooth_values[future_mask], linestyle="--", label="Extrapolated segment")
ax.scatter(sample_years, sample_values, marker="x", s=22, label=f"Sampled model markers (every {sample_increment} year(s))")
ax.axvline(max_year, linestyle=":", label=f"Historical boundary: {max_year}")
ax.set_title(f"{indicator_name} — {country_label}")
ax.set_xlabel("Year")
ax.set_ylabel(indicator["units"])
ax.grid(True, alpha=0.25)
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
chart_buffer = io.BytesIO()
fig.savefig(chart_buffer, format="png", dpi=160, bbox_inches="tight")
chart_bytes = chart_buffer.getvalue()
if printer_mode:
    st.markdown("---")
    st.markdown(f"## Printer-friendly report: {country_label}")
    st.write(f"{indicator_name} | {indicator['units']} | Source: World Bank | Coverage: {min_year}–{max_year}")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Comparison overlays use same indicator, same units, and each country's actual coverage.
if comparison_labels:
    st.subheader("Same-indicator country comparison")
    compare_fig, compare_ax = plt.subplots(figsize=(10, 4.5))
    compare_ax.scatter(working_df.Year, working_df.Value, label=f"{country_label} observed", s=20)
    compare_ax.plot(smooth_years[historical_mask], smooth_values[historical_mask], label=f"{country_label} model")
    compare_notes = []
    for label in comparison_labels:
        comp_code = COUNTRIES[label]
        comp_df, comp_error = cached_fetch(comp_code, indicator["code"])
        if comp_error or comp_df.empty:
            compare_notes.append(f"{label}: no usable official series ({comp_error or 'empty series'}).")
            continue
        compare_ax.plot(comp_df.Year, comp_df.Value, marker=".", markersize=3, linewidth=1, label=f"{label} observed")
        compare_notes.append(f"{label}: coverage {int(comp_df.Year.min())}–{int(comp_df.Year.max())}, {len(comp_df)} observations.")
    compare_ax.set_title(f"Same indicator: {indicator_name}")
    compare_ax.set_xlabel("Year")
    compare_ax.set_ylabel(indicator["units"])
    compare_ax.grid(True, alpha=0.25)
    compare_ax.legend(fontsize=8)
    compare_fig.tight_layout()
    st.pyplot(compare_fig, use_container_width=True)
    plt.close(compare_fig)
    for note in compare_notes:
        st.caption(note)
    st.caption("Country series can have different coverage or methods; comparisons should be interpreted carefully.")

# Single-year estimate
st.subheader("Single-year estimate")
single_year = st.number_input("Year to estimate (1950–2100)", min_value=1950, max_value=2100, value=min(max_year + 1, 2100), step=1)
if st.button("Calculate value"):
    year = int(single_year)
    estimate = float(np.polyval(coeff, year - base_year))
    observed_match = working_df.loc[working_df.Year == year, "Value"]
    if year < min_year or year > max_year:
        kind = "Extrapolation (outside fitted historical range)"
    elif not observed_match.empty:
        kind = "Fitted value at an observed year"
    else:
        kind = "Within-range model estimate"
    st.session_state.single_result = {
        "Year": year, "Estimate": estimate, "Units": indicator["units"],
        "Calculation type": kind,
        "Observed value (if available)": float(observed_match.iloc[0]) if not observed_match.empty else None,
    }
if st.session_state.single_result:
    st.dataframe(pd.DataFrame([st.session_state.single_result]), use_container_width=True)
    val = st.session_state.single_result["Estimate"]
    if not math.isfinite(val) or (indicator_name in ["Life expectancy at birth", "Crude birth rate"] and val < 0):
        st.warning("This model output may be implausible for the indicator. It has not been clipped or silently changed.")
    elif st.session_state.single_result["Calculation type"].startswith("Extrapolation"):
        st.warning("Extrapolation is uncertain; polynomial behavior can become implausible farther beyond the data.")

# Batch estimates
st.subheader("Batch estimates")
batch_mode = st.radio("Batch input type", ["Year range", "Custom years"], horizontal=True)
batch_years = []
if batch_mode == "Year range":
    a, b, step_col = st.columns(3)
    start = int(a.number_input("Start year", 1950, 2100, min_year, step=1))
    end = int(b.number_input("End year", 1950, 2100, min(max_year + 5, 2100), step=1))
    step = int(step_col.number_input("Step", 1, 10, 1, step=1))
    if end <= start:
        st.warning("End year must be greater than start year.")
    else:
        batch_years = list(range(start, end + 1, step))
        st.caption(f"Valid years to calculate: {batch_years}")
else:
    custom = st.text_input("Enter comma-separated years", f"{min_year},{max_year},{min(max_year+1, 2100)}")
    parsed, invalid = [], []
    for part in custom.split(","):
        part = part.strip()
        try:
            y = int(part)
            if 1950 <= y <= 2100:
                parsed.append(y)
            else:
                invalid.append(part)
        except ValueError:
            if part:
                invalid.append(part)
    batch_years = list(dict.fromkeys(parsed))
    if invalid:
        st.warning(f"Invalid/out-of-range entries ignored: {', '.join(invalid)}")
    st.caption(f"Valid unique years to calculate: {batch_years}")
if st.button("Calculate batch", disabled=not batch_years):
    rows = []
    for year in batch_years:
        estimate = float(np.polyval(coeff, year - base_year))
        if year < min_year or year > max_year:
            kind = "Extrapolation"
        elif year in set(working_df.Year):
            kind = "Fitted value at observed year"
        else:
            kind = "Within-range estimate"
        rows.append({"Year": year, "Estimate": estimate, "Units": indicator["units"], "Calculation type": kind})
    st.session_state.batch_results = pd.DataFrame(rows)
if st.session_state.batch_results is not None:
    st.dataframe(st.session_state.batch_results, use_container_width=True)
    if any(st.session_state.batch_results["Calculation type"] == "Extrapolation"):
        st.caption("Batch contains extrapolations, which are uncertain and may be implausible.")

# Findings and limitations starter prompts, clearly not causal claims.
st.subheader("Interpretation prompts")
st.markdown(
    "- **Finding 1:** Describe the overall direction and any notable changes visible in the observed data.\n"
    "- **Finding 2:** Compare the model curve with observed points and mention where the fit is weaker.\n"
    "- **Finding 3:** Compare coverage and trends with the selected country series, if available.\n"
    "- **Limitation 1:** Missing years and differences in national data methods can affect comparisons.\n"
    "- **Limitation 2:** A polynomial is a descriptive fit; extrapolation does not establish a dependable forecast.\n"
    "- **Follow-up question:** What historical, policy, or community context might help explain a trend? Verify the explanation with credible sources."
)

# Export data and model output.
st.subheader("Download reports")
summary = {
    "Student": st.session_state.display_name.strip() or "Student",
    "Country": country_label,
    "Country code": country_code,
    "Indicator": indicator_name,
    "Indicator code": indicator["code"],
    "Units": indicator["units"],
    "Definition": indicator["definition"],
    "Source": f"https://data.worldbank.org/indicator/{indicator['code']}",
    "Retrieved at": datetime.now().isoformat(timespec="minutes"),
    "Coverage": f"{min_year}–{max_year}",
    "Observation count": len(working_df),
    "Edited user scenario": is_edited,
    "Degree": degree,
    "Base year": base_year,
    "Equation": equation,
    "R-squared (in-sample)": r2,
    "MAE": mae,
    "Mean": mean_value,
    "Limitations": "MAE/residuals are not validated prediction intervals; polynomial extrapolation is uncertain."
}
export_data = working_df[["Year", "Value"]].copy()
csv_bytes = export_data.to_csv(index=False).encode("utf-8")
st.download_button("Download working data (CSV)", data=csv_bytes,
                   file_name=f"{country_code}_{indicator['code'].replace('.', '-')}_working_data.csv",
                   mime="text/csv")
results_for_export = st.session_state.batch_results
xlsx_bytes = create_excel(summary, export_data, results_for_export)
st.download_button("Download Excel workbook (.xlsx)", data=xlsx_bytes,
                   file_name=f"{country_code}_{indicator['code'].replace('.', '-')}_report.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
try:
    pdf_bytes = create_pdf("Hispanic Heritage Historical Data Explorer", summary, export_data, chart_bytes, results_for_export)
    st.download_button("Download PDF report", data=pdf_bytes,
                       file_name=f"{country_code}_{indicator['code'].replace('.', '-')}_report.pdf",
                       mime="application/pdf")
except Exception as exc:
    st.error(f"PDF export could not be generated: {exc}")

st.markdown("---")
st.caption(f"Student project by {st.session_state.display_name.strip() or 'Student'} · Official historical data: World Bank API · Last app refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.caption("To make your name part of the actual web address, set a custom app subdomain when deploying if the platform offers one, or name the GitHub repository accordingly. Streamlit assigns the final URL; the app can display your name but cannot force the URL to contain it.")
