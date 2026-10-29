"""
Hispanic Heritage Historical Data Explorer
A Streamlit app using official World Bank indicator data.
Run locally with: streamlit run app.py
"""

from datetime import date, datetime
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Hispanic Heritage Historical Data Explorer",
    page_icon="📊",
    layout="wide",
)

COUNTRIES = {
    "Mexico": "MEX",
    "Brazil (Latin American comparison)": "BRA",
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
        "code": "SP.DYN.LE00.IN",
        "units": "years",
        "definition": "Average number of years a newborn would live if current mortality patterns continued.",
    },
    "Unemployment rate (national estimate)": {
        "code": "SL.UEM.TOTL.NE.ZS",
        "units": "% of labor force",
        "definition": "Unemployment as a percentage of the total labor force, using national estimates.",
    },
    "GDP per capita": {
        "code": "NY.GDP.PCAP.CD",
        "units": "current US dollars per person",
        "definition": "Gross domestic product divided by population; it is not the same as average salary or household wealth.",
    },
    "Crude birth rate": {
        "code": "SP.DYN.CBRT.IN",
        "units": "births per 1,000 people",
        "definition": "Annual number of births per 1,000 people.",
    },
    "Net migration": {
        "code": "SM.POP.NETM",
        "units": "people",
        "definition": "Net number of migrants over the reporting period.",
    },
    "Intentional homicide rate": {
        "code": "VC.IHR.PSRC.P5",
        "units": "homicides per 100,000 people",
        "definition": "Intentional homicides per 100,000 people.",
    },
    "Consumer price inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "units": "% annual change",
        "definition": "Annual percentage change in consumer prices.",
    },
    "Central government debt": {
        "code": "GC.DOD.TOTL.GD.ZS",
        "units": "% of GDP",
        "definition": "Central government debt as a share of GDP.",
    },
    "Total reserves, including gold": {
        "code": "FI.RES.TOTL.CD",
        "units": "current US dollars",
        "definition": "Total reserves, including gold, reported in current US dollars.",
    },
    "External balance: goods and services": {
        "code": "NE.RSB.GNFS.ZS",
        "units": "% of GDP",
        "definition": "External balance on goods and services as a share of GDP.",
    },
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_data(country_code: str, indicator_code: str):
    """Fetch all pages of the World Bank API for the most recent 70-year window."""
    end_year = date.today().year - 1
    start_year = max(1960, end_year - 69)
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/"
        f"{indicator_code}"
    )
    params = {
        "date": f"{start_year}:{end_year}",
        "format": "json",
        "per_page": 200,
    }
    rows = []
    try:
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2:
            return pd.DataFrame(columns=["Year", "Value"]), (
                "The World Bank did not return a usable data response. Please try again."
            )
        metadata = payload[0] or {}
        pages = int(metadata.get("pages", 1))
        for page in range(1, pages + 1):
            if page == 1:
                page_payload = payload
            else:
                page_params = dict(params)
                page_params["page"] = page
                page_response = requests.get(url, params=page_params, timeout=25)
                page_response.raise_for_status()
                page_payload = page_response.json()
            if not isinstance(page_payload, list) or len(page_payload) < 2:
                continue
            for item in page_payload[1] or []:
                year = item.get("date")
                value = item.get("value")
                if year is None or value is None:
                    continue
                try:
                    rows.append({"Year": int(year), "Value": float(value)})
                except (TypeError, ValueError):
                    pass
        frame = pd.DataFrame(rows, columns=["Year", "Value"])
        if frame.empty:
            return frame, "No non-missing observations were found for this indicator and country."
        frame = frame.drop_duplicates(subset=["Year"]).sort_values("Year")
        return frame.reset_index(drop=True), None
    except requests.RequestException:
        return pd.DataFrame(columns=["Year", "Value"]), (
            "Could not connect to the World Bank API. Check your internet connection "
            "or try again later."
        )
    except (ValueError, TypeError, KeyError):
        return pd.DataFrame(columns=["Year", "Value"]), (
            "The World Bank response could not be read. Please try again later."
        )


def clean_data(frame: pd.DataFrame):
    """Validate edited rows without inventing replacement values."""
    data = frame.copy()
    data["Year"] = pd.to_numeric(data.get("Year"), errors="coerce")
    data["Value"] = pd.to_numeric(data.get("Value"), errors="coerce")
    before = len(data)
    data = data.dropna(subset=["Year", "Value"])
    data = data[np.isclose(data["Year"], np.round(data["Year"]))]
    data["Year"] = data["Year"].astype(int)
    data = data[(data["Year"] >= 1900) & (data["Year"] <= date.today().year)]
    data = data.drop_duplicates(subset=["Year"], keep="first")
    data = data.sort_values("Year").reset_index(drop=True)
    removed = before - len(data)
    return data, removed


def fit_model(data: pd.DataFrame, degree: int):
    """Fit in scaled year coordinates to reduce numerical conditioning problems."""
    years = data["Year"].to_numpy(dtype=float)
    values = data["Value"].to_numpy(dtype=float)
    center = float(years.mean())
    scale = float(max(np.ptp(years) / 2, 1.0))
    x = (years - center) / scale
    coefficients = np.polynomial.polynomial.polyfit(x, values, degree)
    fitted = np.polynomial.polynomial.polyval(x, coefficients)
    residuals = values - fitted
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_total = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = float(1 - ss_res / ss_total) if ss_total > 0 else float("nan")
    return center, scale, coefficients, fitted, mae, r_squared


def predict(years, center, scale, coefficients):
    x = (np.asarray(years, dtype=float) - center) / scale
    return np.polynomial.polynomial.polyval(x, coefficients)


def equation_label(coefficients, center, scale):
    # Coefficients use z=(year-center)/scale, which is more stable than raw years.
    parts = [f"{c:.5g}z^{i}" if i > 1 else (f"{c:.5g}z" if i == 1 else f"{c:.5g}")
             for i, c in enumerate(coefficients) if abs(c) > 1e-12]
    return "f(z) = " + (" + ".join(parts).replace("+ -", "- ") if parts else "0") + (
        f", where z = (year - {center:.2f}) / {scale:.2f}"
    )


# Sidebar controls
st.sidebar.title("Project settings")
student_name = st.sidebar.text_input("Name to display on the app", value="Your Name").strip()
if not student_name:
    student_name = "Student"
country_name = st.sidebar.selectbox("Country", list(COUNTRIES.keys()))
indicator_name = st.sidebar.selectbox("Indicator", list(INDICATORS.keys()))
indicator = INDICATORS[indicator_name]
comparison_names = st.sidebar.multiselect(
    "Compare with other countries",
    [name for name in COUNTRIES if name != country_name],
)
st.sidebar.caption(
    "Your name is shown inside the app. Streamlit controls the actual web address."
)

st.title("Hispanic Heritage Historical Data Explorer")
st.subheader(f"Created by {student_name}")
st.write(
    "Explore official historical data, compare countries, and use polynomial models "
    "to describe trends and estimate values."
)
st.info(
    "Brazil is included as a Latin American comparison; it is not a Spanish-speaking "
    "Hispanic country."
)

st.markdown(f"**Indicator:** {indicator_name}  \n"
            f"**Definition:** {indicator['definition']}  \n"
            f"**Units:** {indicator['units']}  \n"
            f"**Indicator code:** `{indicator['code']}`  \n"
            f"**Source:** [World Bank indicator page](https://data.worldbank.org/indicator/{indicator['code']})")

with st.spinner("Loading official World Bank data..."):
    source_data, error = get_world_bank_data(COUNTRIES[country_name], indicator["code"])

if error:
    st.error(error)
    st.markdown(
        "You can try refreshing the page or selecting another indicator. "
        "This app does not create fake replacement data."
    )
    st.stop()

st.caption(
    f"Retrieved {len(source_data)} observations, covering "
    f"{int(source_data['Year'].min())}–{int(source_data['Year'].max())}."
)

st.subheader("Data table")
st.write("You may edit the values or add/remove rows. Analysis uses the edited table.")
edited_data = st.data_editor(
    source_data,
    num_rows="dynamic",
    use_container_width=True,
    key=f"data_{COUNTRIES[country_name]}_{indicator['code']}",
)
data, removed_rows = clean_data(edited_data)
if removed_rows:
    st.warning(f"{removed_rows} invalid or duplicate row(s) were removed from the working data.")
if len(data) < 2:
    st.error("At least two valid data rows are needed. Reset the table or select another indicator.")
    st.stop()

minimum_degree = 1
maximum_degree = min(8, len(data) - 1)
degree = st.slider(
    "Polynomial degree",
    min_value=minimum_degree,
    max_value=maximum_degree,
    value=min(3, maximum_degree),
    help="Higher-degree curves can match historical data closely but may behave unrealistically outside the data range.",
)

try:
    center, scale, coefficients, fitted, mae, r_squared = fit_model(data, degree)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("The model produced non-finite coefficients.")
except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
    st.error(f"The model could not be fitted: {exc}")
    st.stop()

model_data = data.copy()
model_data["Fitted value"] = fitted
model_data["Residual"] = model_data["Value"] - model_data["Fitted value"]

st.subheader("Model summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Observations", f"{len(data)}")
m2.metric("R² (in-sample)", f"{r_squared:.4f}" if math.isfinite(r_squared) else "Undefined")
m3.metric("Mean absolute error", f"{mae:.5g}")
m4.metric("Average observed value", f"{data['Value'].mean():.5g}")
st.write(f"**Model equation:** `{equation_label(coefficients, center, scale)}`")
st.caption(
    "R² and mean absolute error describe how the model fits the available data. "
    "They do not prove a cause or guarantee that future estimates will be accurate."
)

# Plot
st.subheader("Graph")
future_years = st.slider("Years beyond the latest observation to display", 0, 30, 5)
min_year = int(data["Year"].min())
max_year = int(data["Year"].max())
plot_end = max_year + future_years
line_years = np.linspace(min_year, plot_end, max(200, (plot_end - min_year + 1) * 3))
line_values = predict(line_years, center, scale, coefficients)
history = line_years <= max_year
future = line_years > max_year

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(data["Year"], data["Value"], label="Observed / working data", s=24)
ax.plot(line_years[history], line_values[history], label="Model in historical range")
if np.any(future):
    ax.plot(line_years[future], line_values[future], "--", label="Extrapolation")
ax.axvline(max_year, linestyle=":", linewidth=1, label=f"Last observed year: {max_year}")
ax.set_title(f"{indicator_name} — {country_name}")
ax.set_xlabel("Year")
ax.set_ylabel(indicator["units"])
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

image_buffer = io.BytesIO()
fig.savefig(image_buffer, format="png", dpi=150, bbox_inches="tight")
image_bytes = image_buffer.getvalue()
plt.close(fig)

if future_years > 0:
    st.warning(
        "Values beyond the last observed year are polynomial extrapolations, not official "
        "World Bank observations or guaranteed forecasts."
    )

# Country comparison
if comparison_names:
    st.subheader("Country comparison")
    compare_fig, compare_ax = plt.subplots(figsize=(10, 4.8))
    compare_ax.plot(data["Year"], data["Value"], marker=".", label=f"{country_name} (working data)")
    comparison_notes = []
    for name in comparison_names:
        other_data, other_error = get_world_bank_data(COUNTRIES[name], indicator["code"])
        if other_error or other_data.empty:
            comparison_notes.append(f"{name}: data unavailable for this indicator.")
        else:
            compare_ax.plot(
                other_data["Year"], other_data["Value"],
                marker=".", markersize=3, linewidth=1, label=name
            )
            comparison_notes.append(
                f"{name}: {int(other_data['Year'].min())}–{int(other_data['Year'].max())}"
            )
    compare_ax.set_title(f"{indicator_name}: country comparison")
    compare_ax.set_xlabel("Year")
    compare_ax.set_ylabel(indicator["units"])
    compare_ax.grid(True, alpha=0.25)
    compare_ax.legend(fontsize=8)
    compare_fig.tight_layout()
    st.pyplot(compare_fig, use_container_width=True)
    plt.close(compare_fig)
    for note in comparison_notes:
        st.caption(note)
    st.caption("Country data coverage and reporting methods may differ.")

# Calculators
st.subheader("Estimate a value for one year")
single_year = st.number_input(
    "Year",
    min_value=1900,
    max_value=2100,
    value=min(max_year + 1, 2100),
    step=1,
    key="single_year_input",
)
if st.button("Calculate single-year estimate"):
    estimate = float(predict([int(single_year)], center, scale, coefficients)[0])
    st.session_state["single_estimate"] = {
        "Year": int(single_year),
        "Estimated value": estimate,
        "Units": indicator["units"],
        "Type": "Extrapolation" if single_year < min_year or single_year > max_year else "Within-range model estimate",
    }
if st.session_state.get("single_estimate"):
    st.dataframe(pd.DataFrame([st.session_state["single_estimate"]]), use_container_width=True)

st.subheader("Calculate estimates for multiple years")
batch_mode = st.radio("Choose year input", ["Year range", "Enter specific years"], horizontal=True)
batch_years = []
if batch_mode == "Year range":
    c1, c2, c3 = st.columns(3)
    start_year = int(c1.number_input("Start year", min_value=1900, max_value=2100, value=min_year, step=1))
    end_year = int(c2.number_input("End year", min_value=1900, max_value=2100, value=min(max_year + 5, 2100), step=1))
    year_step = int(c3.number_input("Interval", min_value=1, max_value=25, value=1, step=1))
    if end_year >= start_year:
        batch_years = list(range(start_year, end_year + 1, year_step))
    else:
        st.warning("The end year must be equal to or later than the start year.")
else:
    years_text = st.text_input("Years separated by commas", f"{min_year}, {max_year}, {min(max_year + 1, 2100)}")
    valid_years = []
    invalid_years = []
    for item in years_text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            year = int(item)
            if 1900 <= year <= 2100:
                valid_years.append(year)
            else:
                invalid_years.append(item)
        except ValueError:
            invalid_years.append(item)
    batch_years = list(dict.fromkeys(valid_years))
    if invalid_years:
        st.warning("Ignored invalid years: " + ", ".join(invalid_years))

if st.button("Calculate batch estimates", disabled=not batch_years):
    estimates = predict(batch_years, center, scale, coefficients)
    rows = []
    observed_years = set(data["Year"].tolist())
    for year, value in zip(batch_years, estimates):
        if year < min_year or year > max_year:
            kind = "Extrapolation"
        elif year in observed_years:
            kind = "Model fit at an observed year"
        else:
            kind = "Within-range estimate"
        rows.append({
            "Year": int(year),
            "Estimated value": float(value),
            "Units": indicator["units"],
            "Calculation type": kind,
        })
    st.session_state["batch_estimates"] = pd.DataFrame(rows)

batch_results = st.session_state.get("batch_estimates")
if isinstance(batch_results, pd.DataFrame) and not batch_results.empty:
    st.dataframe(batch_results, use_container_width=True)
    if (batch_results["Calculation type"] == "Extrapolation").any():
        st.caption("The batch includes extrapolations, which may be unreliable.")

# Interpretation guidance
st.subheader("Interpretation prompts")
st.markdown(
    "- Describe the overall direction of the observed data.\n"
    "- Identify any years with noticeable changes.\n"
    "- Compare the fitted curve with the observed points.\n"
    "- Explain one limitation, such as missing observations or uncertain extrapolation.\n"
    "- Research credible historical sources before claiming why a trend occurred."
)

# Downloads
st.subheader("Download your results")
download_data = model_data[["Year", "Value", "Fitted value", "Residual"]].copy()
csv_bytes = download_data.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download working data as CSV",
    data=csv_bytes,
    file_name=f"{COUNTRIES[country_name]}_{indicator['code'].replace('.', '-')}_data.csv",
    mime="text/csv",
)
if isinstance(batch_results, pd.DataFrame) and not batch_results.empty:
    batch_csv = batch_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download batch estimates as CSV",
        data=batch_csv,
        file_name="batch_estimates.csv",
        mime="text/csv",
    )
st.download_button(
    "Download graph as PNG",
    data=image_bytes,
    file_name="historical_data_graph.png",
    mime="image/png",
)

st.caption(
    f"Student project by {student_name} · Data source: World Bank API · "
    f"App refreshed {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
