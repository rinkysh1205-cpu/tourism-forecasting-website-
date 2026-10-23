
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="International Tourism Trends & Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "international-tourist-trips.csv"
OUTPUT_DIR = BASE_DIR / "tourism_outputs"

# ============================================================
# PROFESSIONAL UI
# ============================================================
st.markdown("""
<style>
/* Final professional research-dashboard finish */
.stApp {
    background: linear-gradient(180deg, #FCFDFE 0%, #F7FAFC 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EEF4FA 0%, #F6F3FA 100%);
    border-right: 1px solid #D8E2EC;
}
h1, h2, h3 {
    color: #243447;
    letter-spacing: -0.02em;
}
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DDE6EE;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(36, 52, 71, 0.06);
}
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid rgba(90, 110, 130, 0.12);
}
div[data-baseweb="select"] > div {
    border-radius: 10px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
}
</style>

<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .main-title {
        font-size: 2.75rem; font-weight: 800; text-align: center;
        margin-bottom: .25rem;
    }
    .subtitle {
        text-align:center; font-size:1.15rem; opacity:.75;
        margin-bottom:1.5rem;
    }
    .hero {
        padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.22);
        border-radius: 18px; margin-bottom: 1.2rem;
    }
    .section-card {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 16px; padding: 1rem 1.15rem; margin: .5rem 0;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.20);
        padding: 14px; border-radius: 14px;
    }
    .small-note {font-size:.9rem; opacity:.72;}

    /* Soft professional academic palette */
    .stApp {
        background: linear-gradient(180deg, #fbfdff 0%, #ffffff 35%, #f8fbff 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef6ff 0%, #f7f3ff 100%);
        border-right: 1px solid #dbe7f3;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f2f8ff 0%, #faf7ff 100%);
        border: 1px solid #d9e6f2;
        box-shadow: 0 3px 10px rgba(44, 76, 110, 0.05);
    }
    div[data-testid="stAlert"] {
        border-radius: 14px;
    }
    .hero {
        background: linear-gradient(135deg, #eef8ff 0%, #f8f3ff 52%, #f1fbf7 100%);
        border: 1px solid #dce9f3;
        box-shadow: 0 4px 14px rgba(44, 76, 110, 0.06);
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #dfe8f0;
        box-shadow: 0 3px 10px rgba(44, 76, 110, 0.04);
    }
    footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def fmt_num(x):
    if pd.isna(x):
        return "N/A"
    x = float(x)
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:,.0f}"

def style_chart(fig):
    """Apply a clean, light professional style to every Plotly chart."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fbfdff",
        font=dict(color="#243447"),
        title_font=dict(size=18),
        margin=dict(l=45, r=30, t=65, b=45),
        legend=dict(
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#e3ebf2",
            borderwidth=1
        )
    )
    fig.update_xaxes(showgrid=False, linecolor="#dfe7ef")
    fig.update_yaxes(gridcolor="#e8eef4", zerolinecolor="#cfd9e3")
    return fig

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return None
    df = pd.read_csv(DATA_FILE)
    df = df.rename(columns={
        "Entity": "Country",
        "Code": "Country_Code",
        "Arrivals of tourists from abroad": "Tourist_Arrivals",
        "World region according to OWID": "Region"
    })
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Tourist_Arrivals"] = pd.to_numeric(df["Tourist_Arrivals"], errors="coerce")
    df = df.dropna(subset=["Country", "Year", "Tourist_Arrivals"]).copy()
    df["Year"] = df["Year"].astype(int)
    return df.sort_values(["Country", "Year"]).reset_index(drop=True)

@st.cache_data
def load_output(filename):
    path = OUTPUT_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

df = load_data()
best_models = load_output("best_model_by_country.csv")
model_comparison = load_output("model_comparison_all_countries.csv")
forecast = load_output("tourist_arrivals_forecast_2025_2029.csv")
forecast_growth = load_output("forecast_growth_2025_2029.csv")

overall_models = load_output("overall_model_summary.csv")

# Country-level forecasting eligibility used in the final project
coverage_info = (
    df.groupby("Country")
      .agg(
          Observations=("Year", "count"),
          First_Year=("Year", "min"),
          Last_Year=("Year", "max")
      )
      .reset_index()
)
coverage_info["Year_Span"] = coverage_info["Last_Year"] - coverage_info["First_Year"] + 1
coverage_info["Coverage_%"] = (
    coverage_info["Observations"] / coverage_info["Year_Span"] * 100
).round(2)
coverage_info["Eligible"] = (
    (coverage_info["Observations"] >= 15) &
    (coverage_info["Last_Year"] == 2024) &
    (coverage_info["Coverage_%"] >= 80)
)


if df is None:
    st.error("Dataset file not found. Keep international-tourist-trips.csv in the same folder as app.py.")
    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title(" Tourism Research")
st.sidebar.markdown("---")

pages = [
    " Home",
    " Tourism Overview",
    " Country Analysis",
    " Regional Analysis",
    " COVID & Recovery",
    " Forecasting",
    " Model Comparison",
    " Final Results",
]
page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("###  Project Snapshot")
st.sidebar.metric("Records", f"{len(df):,}")
st.sidebar.metric("Countries / Entities", f"{df['Country'].nunique():,}")
st.sidebar.metric("Historical Period", f"{df['Year'].min()}–{df['Year'].max()}")
if not forecast.empty:
    st.sidebar.metric("Forecasted Countries", forecast["country"].nunique())

# ============================================================
# HOME
# ============================================================
if page == " Home":
    st.markdown('<div class="main-title"> International Tourism Trends & Tourist Arrivals Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Country-wise historical analysis and validated time-series forecasts for 2025–2029</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
    <b>Research Platform</b><br>
    This interactive application transforms the finalized Jupyter analysis into an easy-to-explore tourism research dashboard.
    It studies historical tourist-arrival trends, checks forecasting eligibility, compares forecasting models, and produces country-wise forecasts.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows", f"{len(df):,}")
    c2.metric("Countries / Entities", df["Country"].nunique())
    c3.metric("Observed Years", f"{df['Year'].min()}–{df['Year'].max()}")
    c4.metric("Forecast Horizon", "2025–2029")

    st.markdown("###  Main Objective")
    st.info("To forecast international tourist arrivals for the next five years (2025–2029) using historical tourism data and time-series forecasting techniques.")

    st.markdown("###  Research Framework")
    a, b, c = st.columns(3)
    with a:
        st.markdown("####  Historical Trends")
        st.write("Understand how tourist arrivals changed over time before forecasting.")
    with b:
        st.markdown("####  Eligibility & Validation")
        st.write("Forecast only countries with sufficient, recent and well-covered historical data.")
    with c:
        st.markdown("####  Forecasting")
        st.write("Compare Naive, Holt and ARIMA and select the lowest-RMSE model for each eligible country.")

# ============================================================
# TOURISM OVERVIEW
# ============================================================
elif page == " Tourism Overview":
    st.title(" International Tourism Overview")

    years = sorted(df["Year"].unique())
    selected_range = st.slider(
        "Select historical period",
        int(min(years)), int(max(years)),
        (int(min(years)), int(max(years)))
    )
    filtered = df[df["Year"].between(*selected_range)].copy()

    annual = filtered.groupby("Year", as_index=False)["Tourist_Arrivals"].sum()
    annual["YoY_Growth_%"] = annual["Tourist_Arrivals"].pct_change() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Years Selected", annual["Year"].nunique())
    c2.metric("Latest Total*", fmt_num(annual.iloc[-1]["Tourist_Arrivals"]))
    c3.metric("Entities in Selection", filtered["Country"].nunique())

    fig = px.line(
        annual, x="Year", y="Tourist_Arrivals", markers=True,
        title="International Tourist Arrivals Over Time",
        color_discrete_sequence=["#5B9BD5"]
    )
    fig.update_layout(yaxis_title="Tourist Arrivals", hovermode="x unified")
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**What this graph shows:** This line shows how total recorded international tourist arrivals changed over the historical period. "
        "An upward line indicates growth in recorded arrivals, while a fall indicates a decline. "
        "This historical movement provides the time-series pattern used before forecasting."
    )
    st.caption("*Annual total is the sum of entities available in the dataset for that year; historical country coverage can vary.")

# ============================================================
# COUNTRY ANALYSIS
# ============================================================

    latest_year = int(df["Year"].max())
    top_n = st.slider("Number of top destinations", 5, 20, 10)
    top = (
        df[df["Year"] == latest_year]
        .nlargest(top_n, "Tourist_Arrivals")
        .sort_values("Tourist_Arrivals")
    )
    pastel_bar_colors = [
        "#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#7262FD",
        "#78D3F8", "#9661BC", "#F6903D", "#008685", "#F08BB4"
    ]
    fig3 = go.Figure(go.Bar(
        x=top["Tourist_Arrivals"],
        y=top["Country"],
        orientation="h",
        marker=dict(color=pastel_bar_colors[:len(top)]),
        hovertemplate="<b>%{y}</b><br>Tourist Arrivals: %{x:,.0f}<extra></extra>"
    ))
    fig3.update_layout(
        title=f"Top {top_n} Destinations by Tourist Arrivals ({latest_year})",
        xaxis_title="Tourist Arrivals",
        yaxis_title="Country"
    )
    style_chart(fig3)
    st.plotly_chart(fig3, use_container_width=True)
    st.info(
        f"**What this graph shows:** This chart compares the destinations with the highest recorded tourist arrivals in {latest_year}. "
        "Longer bars represent a larger number of international tourist arrivals."
    )

elif page == " Country Analysis":
    st.title(" Country-wise Tourism Analysis")

    countries = sorted(df["Country"].unique())
    default_country = "India" if "India" in countries else countries[0]
    country = st.selectbox("Select Country / Entity", countries, index=countries.index(default_country))

    cdf = df[df["Country"] == country].sort_values("Year").copy()
    cdf["YoY_Growth_%"] = cdf["Tourist_Arrivals"].pct_change() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("First Year", int(cdf["Year"].min()))
    c2.metric("Latest Year", int(cdf["Year"].max()))
    c3.metric("Latest Arrivals", fmt_num(cdf.iloc[-1]["Tourist_Arrivals"]))
    c4.metric("Observations", len(cdf))

    fig = px.line(cdf, x="Year", y="Tourist_Arrivals", markers=True,
                  title=f"Historical Tourist Arrivals — {country}")
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**What this graph shows:** This graph displays how tourist arrivals for the selected country changed over time. It makes the country's historical tourism pattern easy to understand.")

# ============================================================
# REGIONAL ANALYSIS
# ============================================================
elif page == " Regional Analysis":
    st.title(" Regional Tourism Analysis")

    rdf = df.dropna(subset=["Region"]).copy()
    latest_year = int(rdf["Year"].max())

    latest_region = (
        rdf[rdf["Year"] == latest_year]
        .groupby("Region", as_index=False)["Tourist_Arrivals"].sum()
        .sort_values("Tourist_Arrivals", ascending=False)
    )

    fig = px.bar(
        latest_region, x="Region", y="Tourist_Arrivals",
        title=f"Tourist Arrivals by Region ({latest_year})"
    )
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**What this graph shows:** This graph compares tourist-arrival patterns across regions. It helps identify which regions record higher or lower tourism activity.")

# ============================================================
# COVID & RECOVERY
# ============================================================
elif page == " COVID & Recovery":
    st.title(" COVID-19 Impact & Tourism Recovery")

    pivot = df.pivot_table(index="Country", columns="Year", values="Tourist_Arrivals", aggfunc="sum")

    if 2019 in pivot.columns and 2020 in pivot.columns:
        covid = pivot[[2019, 2020]].dropna().copy()
        covid["Change_%"] = ((covid[2020] - covid[2019]) / covid[2019].replace(0, np.nan)) * 100
        worst = covid.nsmallest(15, "Change_%").reset_index()

        fig = px.bar(
            worst.sort_values("Change_%"), x="Change_%", y="Country", orientation="h",
            title="Largest Observed Declines in Tourist Arrivals: 2019–2020",
        color_discrete_sequence=["#4E79A7", "#E15759", "#59A14F", "#B07AA1", "#EDC948"]
        )
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**What this graph shows:** This graph compares tourist arrivals before and during the 2020 disruption. A larger fall indicates a stronger decline in recorded arrivals.")

    if 2019 in pivot.columns and 2024 in pivot.columns:
        recovery = pivot[[2019, 2024]].dropna().copy()
        recovery["Recovery_Rate_%"] = recovery[2024] / recovery[2019].replace(0, np.nan) * 100
        recovery = recovery.replace([np.inf, -np.inf], np.nan).dropna()
        top_rec = recovery.nlargest(15, "Recovery_Rate_%").reset_index()

        fig2 = px.bar(
            top_rec.sort_values("Recovery_Rate_%"),
            x="Recovery_Rate_%", y="Country", orientation="h",
            title="Top Recovery Rates: 2024 Compared with 2019",
        color_discrete_sequence=["#4E79A7", "#E15759", "#59A14F", "#B07AA1", "#EDC948"]
        )
        fig2.add_vline(x=100, line_dash="dash")
        style_chart(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("**What this graph shows:** This graph compares tourist arrivals before and during the 2020 disruption. A larger fall indicates a stronger decline in recorded arrivals.")

        st.info("**Easy meaning:** Recovery Rate compares 2024 tourist arrivals with 2019. **100% means tourism returned to the 2019 level; above 100% means it performed better than 2019; below 100% means it has not fully returned yet.**")

# ============================================================
# FORECASTING
# ============================================================
elif page == " Forecasting":
    st.title(" Country-wise Tourist Arrivals Forecasting")
    st.caption("Only countries that satisfy the forecasting eligibility rules are shown below.")

    if forecast.empty or best_models.empty or model_comparison.empty:
        st.error("Forecast/model output files are missing. Keep the finalized tourism_outputs folder beside app.py.")
        st.stop()

    countries = sorted(forecast["country"].dropna().unique())
    default_country = "Albania" if "Albania" in countries else countries[0]
    country = st.selectbox(
        "Select Eligible Country",
        countries,
        index=countries.index(default_country)
    )

    f = forecast[forecast["country"] == country].sort_values("forecast_year").copy()
    b = best_models[best_models["country"] == country].iloc[0]
    hist = df[df["Country"] == country].sort_values("Year").copy()

    # ------------------------------------------------------------
    # 1. WHY THIS COUNTRY WAS SELECTED
    # ------------------------------------------------------------
    info = coverage_info[coverage_info["Country"] == country]
    if not info.empty:
        info = info.iloc[0]
        st.markdown("###  Why is this country eligible?")
        e1, e2, e3 = st.columns(3)
        e1.metric("Historical Observations", int(info["Observations"]), "Required ≥ 15")
        e2.metric("Latest Data Year", int(info["Last_Year"]), "Required = 2024")
        e3.metric("Historical Coverage", f"{info['Coverage_%']:.1f}%", "Required ≥ 80%")
        st.caption(
            f"{country} satisfies all three forecasting rules, so it was included in the eligible-country list."
        )

    # ------------------------------------------------------------
    # 2. HOW THE MODEL WAS SELECTED
    # ------------------------------------------------------------
    st.markdown("###  Why were these three models used?")
    st.write(
        "Countries can show different tourism patterns, so three different forecasting approaches were compared:"
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("**Naive — Simple / Stable**")
        st.caption("Useful as a basic benchmark when recent arrivals are a strong reference. Example: 5.0L → 5.1L → 5.0L → 5.1L.")
    with m2:
        st.markdown("**Holt — Trend**")
        st.caption("Useful when arrivals show a clear rising or falling trend. Example: 2L → 3L → 4L → 5L.")
    with m3:
        st.markdown("**ARIMA — Time Pattern**")
        st.caption("Useful when arrivals have more complex ups and downs over time. Example: 5L → 6L → 5.4L → 6.5L.")

    st.caption(
        "These examples explain why a model may suit a pattern. The final model is not chosen manually from the graph."
    )

    st.markdown("###  How was the best model selected?")
    st.write(
        "For every eligible country, **Naive, Holt and ARIMA** were tested on the same chronological validation period. "
        "Their validation RMSE values were compared, and the model with the **lowest RMSE** was automatically selected."
    )

    comp = model_comparison[model_comparison["country"] == country].copy()
    if not comp.empty:
        comp["Result"] = comp["model"].apply(
            lambda x: "✓ Selected" if x == b["best_model"] else ""
        )
        comp = comp.sort_values(["rmse", "mae"])

        compact = comp[["model", "rmse", "Result"]].copy()
        compact.columns = ["Model", "Validation RMSE", "Result"]

        st.dataframe(
            compact.style.format({"Validation RMSE": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True
        )

        best_rmse = comp.iloc[0]["rmse"]
        st.success(
            f"**{b['best_model']} was selected for {country}** because it produced "
            f"the lowest validation RMSE ({fmt_num(best_rmse)}) among the three models."
        )

    # ------------------------------------------------------------
    # 3. SELECTED MODEL PERFORMANCE
    # ------------------------------------------------------------
    st.markdown("###  Selected Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Model", str(b["best_model"]))
    c2.metric("Validation RMSE", fmt_num(b["rmse"]))
    c3.metric("Validation MAE", fmt_num(b["mae"]))
    c4.metric("Validation MAPE", f"{b['mape']:.2f}%")

    st.caption(
        "**RMSE:** main selection metric; lower is better.  •  "
        "**MAE:** average absolute difference between actual and predicted arrivals.  •  "
        "**MAPE:** average forecasting error expressed as a percentage."
    )

    # ------------------------------------------------------------
    # 4. HISTORICAL + FINAL FORECAST
    # ------------------------------------------------------------
    st.markdown("###  Historical & Forecast Tourist Arrivals")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Year"], y=hist["Tourist_Arrivals"],
        mode="lines+markers", name="Historical",
        line=dict(color="#4F81BD", width=3),
        marker=dict(color="#4F81BD", size=7),
        hovertemplate="<b>Historical Observation</b><br>Year: %{x}<br>Tourist Arrivals: %{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=f["forecast_year"], y=f["predicted_arrivals"],
        mode="lines+markers+text", name="Forecast",
        line=dict(color="#63B7AF", width=3),
        marker=dict(color="#63B7AF", size=8),
        text=[fmt_num(v) for v in f["predicted_arrivals"]],
        textposition="top center",
        hovertemplate="<b>Forecast Point</b><br>Year: %{x}<br>Predicted Tourist Arrivals: %{y:,.0f}<extra></extra>"
    ))

    if (
        "lower_95" in f.columns and "upper_95" in f.columns and
        f["lower_95"].notna().any() and f["upper_95"].notna().any()
    ):
        fig.add_trace(go.Scatter(
            x=list(f["forecast_year"]) + list(f["forecast_year"][::-1]),
            y=list(f["upper_95"]) + list(f["lower_95"][::-1]),
            fill="toself",
            line=dict(width=0),
            name="95% Prediction Interval",
            opacity=.2
        ))

    fig.add_vline(
        x=2024.5, line_dash="dash", line_width=2,
        annotation_text="Forecast starts", annotation_position="top"
    )
    fig.update_layout(
        title=f"Historical and Forecast Tourist Arrivals — {country}",
        xaxis_title="Year",
        yaxis_title="Tourist Arrivals",
        hovermode="x unified"
    )
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"**How to read this graph:** Each dark-blue point is the actual recorded tourist arrivals for that year. "
        f"Each light point is the predicted tourist arrivals for 2025–2029. Hover over any point for the exact year and value. "
        f"The dashed line shows where historical data ends and forecasting begins."
    )

    # ------------------------------------------------------------
    # 5. FORECAST TABLE
    # ------------------------------------------------------------
    st.markdown("###  Forecast 2025–2029")
    wanted = [
        c for c in [
            "forecast_year", "predicted_arrivals", "lower_95", "upper_95",
            "selected_model", "model_detail"
        ] if c in f.columns
    ]
    display_f = f[wanted].copy()

    rename_map = {
        "forecast_year": "Year",
        "predicted_arrivals": "Forecasted Arrivals",
        "lower_95": "Lower 95%",
        "upper_95": "Upper 95%",
        "selected_model": "Selected Model",
        "model_detail": "Model Detail"
    }
    display_f = display_f.rename(columns=rename_map)

    format_map = {}
    for col in ["Forecasted Arrivals", "Lower 95%", "Upper 95%"]:
        if col in display_f.columns:
            format_map[col] = "{:,.0f}"

    st.dataframe(
        display_f.style.format(format_map, na_rep="—"),
        use_container_width=True,
        hide_index=True
    )

    growth_row = forecast_growth[forecast_growth["country"] == country]
    if not growth_row.empty:
        g = growth_row.iloc[0]
        st.info(
            f"**2025–2029 projected change:** {g['forecast_growth_percent']:.2f}%  |  "
            f"**Forecast direction:** {str(g['forecast_direction']).replace('_', ' ').title()}"
        )

    st.download_button(
        " Download Country Forecast",
        f.to_csv(index=False).encode("utf-8"),
        f"{country.replace(' ','_')}_forecast_2025_2029.csv",
        "text/csv",
        use_container_width=True
    )

# ============================================================
# MODEL COMPARISON
# ============================================================
elif page == " Model Comparison":
    st.title(" Forecasting Model Comparison")

    if model_comparison.empty or best_models.empty:
        st.error("Model comparison output is missing.")
        st.stop()

    counts = best_models["best_model"].value_counts().rename_axis("Model").reset_index(name="Countries")
    c1, c2, c3 = st.columns(3)
    for col, model in zip([c1,c2,c3], ["Holt","Naive","ARIMA"]):
        value = int(counts.loc[counts["Model"] == model, "Countries"].sum())
        col.metric(f"{model} Selected", value)

    fig = px.bar(
        counts, x="Model", y="Countries", text="Countries",
        title="Best Forecasting Model by Number of Countries",
        color="Model",
        color_discrete_sequence=["#8FBCE6", "#A9D6C3", "#C7B6E8"]
    )
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**What this graph shows:** This graph compares forecasting-model performance. A lower validation RMSE indicates better prediction accuracy on the held-out data.")

    countries = sorted(model_comparison["country"].unique())
    country = st.selectbox("Compare models for a country", countries)
    comp = model_comparison[model_comparison["country"] == country].copy()

    fig2 = px.bar(
        comp, x="model", y="rmse", text_auto=".3s",
        title=f"Validation RMSE Comparison — {country}",
        color="model",
        color_discrete_sequence=["#8FBCE6", "#A9D6C3", "#C7B6E8"]
    )
    style_chart(fig2)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("**What this graph shows:** This graph compares forecasting-model performance. A lower validation RMSE indicates better prediction accuracy on the held-out data.")

    st.dataframe(
        comp.rename(columns={
            "country":"Country","model":"Model","arima_order":"ARIMA Order",
            "mae":"MAE","rmse":"RMSE","mape":"MAPE"
        }).style.format({"MAE":"{:,.2f}","RMSE":"{:,.2f}","MAPE":"{:.2f}%"}),
        use_container_width=True,
        hide_index=True
    )
    st.caption("The final country-wise model is selected primarily by the lowest validation RMSE, with MAE used as a secondary criterion.")

# ============================================================
# FINAL RESULTS
# ============================================================
elif page == " Final Results":
    st.title(" Final Forecasting Results")

    if forecast.empty:
        st.error("Forecast output file is missing.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecasted Countries", forecast["country"].nunique())
    c2.metric("Forecast Period", f"{int(forecast['forecast_year'].min())}–{int(forecast['forecast_year'].max())}")
    c3.metric("Models Compared", "3")
    c4.metric("Selection Metric", "RMSE")

    if not forecast_growth.empty:
        direction = forecast_growth["forecast_direction"].value_counts().rename_axis("Direction").reset_index(name="Countries")
        fig = px.pie(direction, names="Direction", values="Countries",
                     title="Forecast Direction Summary")
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**What this graph shows:** This graph compares forecasting-model performance. A lower validation RMSE indicates better prediction accuracy on the held-out data.")

        top_growth = forecast_growth.nlargest(10, "forecast_growth_percent").sort_values("forecast_growth_percent")
        fig2 = px.bar(
            top_growth, x="forecast_growth_percent", y="country", orientation="h",
            title="Top 10 Countries by Projected Growth, 2025–2029",
            color="forecast_growth_percent",
            color_continuous_scale=["#DCEEFF", "#9FC9EA", "#6AAED6"]
        )
        fig2.update_layout(coloraxis_showscale=False)
        style_chart(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        st.info(
            "**What this graph shows:** This chart highlights the countries with the highest projected percentage growth "
            "between 2025 and 2029 based on the final selected forecasting model for each country."
        )

    st.subheader("Complete Forecast Table")
    st.dataframe(
        forecast.style.format({
            "predicted_arrivals":"{:,.0f}",
            "lower_95":"{:,.0f}",
            "upper_95":"{:,.0f}",
            "validation_mae":"{:,.2f}",
            "validation_rmse":"{:,.2f}",
            "validation_mape":"{:.2f}"
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        " Download Complete Forecast Results",
        forecast.to_csv(index=False).encode("utf-8"),
        "tourist_arrivals_forecast_2025_2029.csv",
        "text/csv",
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; opacity:.7; font-size:.9rem;">
    <b>International Tourism Trends & Tourist Arrivals Forecasting</b><br>
    </div>
    """,
    unsafe_allow_html=True
)
