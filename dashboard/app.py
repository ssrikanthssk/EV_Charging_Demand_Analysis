"""EV Charging Demand Dashboard (Streamlit)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"

CAT_COLS = ["ChargerCompany", "ChargerType", "Location", "TimeOfDay"]
NUM_COLS = ["Hour", "Day", "Month", "DayOfWeek", "IsWeekend",
            "User_PrevSessions", "User_AvgPastDemand",
            "Charger_PrevSessions", "Charger_AvgPastDemand", "IsNewUser"]
FEATURES = CAT_COLS + NUM_COLS

st.set_page_config(page_title="EV Charging Demand", layout="wide")


# ====================== Loaders ======================
@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data/processed/featured_data.csv",
                     parse_dates=["StartDatetime"])
    return df.sort_values("StartDatetime").reset_index(drop=True)


@st.cache_resource
def load_models():
    reg = joblib.load(BASE / "models/best_regression_model.pkl")
    clf = joblib.load(BASE / "models/best_classification_model.pkl")
    return reg, clf


def time_of_day(h):
    if 5 <= h < 12:
        return "Morning"
    if 12 <= h < 17:
        return "Afternoon"
    if 17 <= h < 21:
        return "Evening"
    return "Night"


df = load_data()
reg_model, clf_model = load_models()
median_demand = df["Demand"].median()
dmin, dmax = df["StartDatetime"].min().date(), df["StartDatetime"].max().date()


def session_inputs(prefix):
    """Input widgets for one charging session. Returns a one-row DataFrame."""
    c1, c2, c3 = st.columns(3)
    company = c1.selectbox("Charger company", sorted(df["ChargerCompany"].unique()),
                           key=f"{prefix}_company")
    ctype = c2.selectbox("Charger type", sorted(df["ChargerType"].unique()),
                         key=f"{prefix}_type")
    location = c3.selectbox("Location", sorted(df["Location"].unique()),
                            key=f"{prefix}_loc")
    d1, d2 = st.columns(2)
    date = d1.date_input("Date", value=dmax, key=f"{prefix}_date")
    hour = d2.slider("Hour of day", 0, 23, 18, key=f"{prefix}_hour")
    h1, h2, h3, h4 = st.columns(4)
    user_prev = h1.number_input("User previous sessions", 0, value=5, key=f"{prefix}_up")
    user_avg = h2.number_input("User avg past demand", 0.0, value=float(median_demand),
                               key=f"{prefix}_ua")
    chg_prev = h3.number_input("Charger previous sessions", 0, value=100, key=f"{prefix}_cp")
    chg_avg = h4.number_input("Charger avg past demand", 0.0, value=float(median_demand),
                              key=f"{prefix}_ca")
    ts = pd.Timestamp(date)
    return pd.DataFrame([{
        "ChargerCompany": int(company), "ChargerType": int(ctype),
        "Location": location, "TimeOfDay": time_of_day(hour),
        "Hour": hour, "Day": ts.day, "Month": ts.month,
        "DayOfWeek": ts.dayofweek, "IsWeekend": int(ts.dayofweek >= 5),
        "User_PrevSessions": user_prev, "User_AvgPastDemand": user_avg,
        "Charger_PrevSessions": chg_prev, "Charger_AvgPastDemand": chg_avg,
        "IsNewUser": int(user_prev == 0),
    }])


def model_name(pipe):
    return type(pipe.named_steps["model"]).__name__


def gauge(value, title, vmax, split, number_fmt, suffix=""):
    """Gauge chart. Blue zone = below `split`, orange zone = at or above it."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": number_fmt, "suffix": suffix},
        title={"text": title},
        gauge={
            "axis": {"range": [0, vmax]},
            "bar": {"color": "#1f77b4"},
            "steps": [{"range": [0, split], "color": "#d6eaf8"},
                      {"range": [split, vmax], "color": "#fdebd0"}],
            "threshold": {"line": {"color": "red", "width": 4},
                          "thickness": 0.75, "value": split},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def show_importance(fname, title, key):
    """Horizontal bar chart of the top 15 feature importances from a results CSV."""
    p = RES / fname
    if not p.exists():
        st.info(f"{fname} not found in results/.")
        return
    imp = pd.read_csv(p, index_col=0).head(15).sort_values("importance")
    st.plotly_chart(px.bar(imp, x="importance", y=imp.index, orientation="h",
                           title=title, labels={"y": "Feature"}),
                    use_container_width=True, key=key)


# ====================== Sidebar filters ======================
st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to the Overview tab only.")
locs = st.sidebar.multiselect("Location", sorted(df["Location"].unique()),
                              default=sorted(df["Location"].unique()))
types = st.sidebar.multiselect("Charger type", sorted(df["ChargerType"].unique()),
                               default=sorted(df["ChargerType"].unique()))
comps = st.sidebar.multiselect("Charger company", sorted(df["ChargerCompany"].unique()),
                               default=sorted(df["ChargerCompany"].unique()))
date_range = st.sidebar.date_input("Date range", (dmin, dmax),
                                   min_value=dmin, max_value=dmax)

fdf = df[df["Location"].isin(locs) & df["ChargerType"].isin(types)
         & df["ChargerCompany"].isin(comps)]
if len(date_range) == 2:
    fdf = fdf[(fdf["StartDatetime"].dt.date >= date_range[0])
              & (fdf["StartDatetime"].dt.date <= date_range[1])]

st.title("EV Charging Demand Prediction and Usage Pattern Analysis")
if fdf.empty:
    st.warning("No data for the selected filters.")
    st.stop()

tab_o, tab_d, tab_r, tab_c, tab_p = st.tabs(
    ["Overview", "Data Explore", "1. Regression", "2. Classification",
     "Demand Prediction"])

# ====================== OVERVIEW ======================
with tab_o:
    st.markdown(
        "A quick look at EV charging activity for the filters selected in the "
        "sidebar. **Demand** is the energy requested in a charging session."
    )

    a, b = st.columns(2)

    # ---- Demand Distribution ----
    fig = px.histogram(fdf, x="Demand", nbins=50, title="Demand Distribution")
    fig.add_vline(x=median_demand, line_dash="dash", line_color="red",
                  annotation_text="median")
    a.plotly_chart(fig, use_container_width=True, key="ov_dist")
    a.caption(f"Most sessions are small and a few are very large, so the mean "
              f"({fdf['Demand'].mean():.2f}) is above the median "
              f"({fdf['Demand'].median():.2f}).")

    # ---- Sessions by Time Period ----
    tod = (fdf.groupby("TimeOfDay").size()
              .reindex(["Morning", "Afternoon", "Evening", "Night"])
              .reset_index(name="Sessions"))
    b.plotly_chart(px.bar(tod, x="TimeOfDay", y="Sessions",
                          labels={"TimeOfDay": "Time period"},
                          title="Sessions by Time Period"),
                   use_container_width=True, key="ov_tod")
    b.caption(f"{tod.loc[tod['Sessions'].idxmax(), 'TimeOfDay']} has the most sessions. "
              f"(Morning 5-12, Afternoon 12-17, Evening 17-21, Night 21-5.)")

    st.divider()
    a, b = st.columns(2)

    # ---- Average Demand by Location Type ----
    loc_avg = (fdf.groupby("Location")["Demand"].mean()
                  .sort_values(ascending=False).reset_index())
    loc_avg.columns = ["Location", "Avg demand"]
    a.plotly_chart(px.bar(loc_avg, x="Location", y="Avg demand",
                          title="Average Demand by Location Type"),
                   use_container_width=True, key="ov_loc")
    a.caption(f"{loc_avg.iloc[0]['Location']} has the highest average demand "
              f"({loc_avg.iloc[0]['Avg demand']:.2f}).")

    # ---- Hourly Demand Pattern ----
    hourly = fdf.groupby("Hour")["Demand"].mean().reset_index()
    hourly.columns = ["Hour", "Avg demand"]
    b.plotly_chart(px.line(hourly, x="Hour", y="Avg demand", markers=True,
                           title="Hourly Demand Pattern"),
                   use_container_width=True, key="ov_hour")
    b.caption(f"Average demand per session peaks around "
              f"{int(hourly.loc[hourly['Avg demand'].idxmax(), 'Hour'])}:00.")


# ====================== 1. REGRESSION ======================
with tab_r:
    st.header("1. Regression: Charging Demand Prediction")
    st.caption("Predicts the demand (a number) of a charging session. "
               "Models are tested on the last 20% of sessions (time-based split).")

    reg_res_path = RES / "regression_results.csv"
    if reg_res_path.exists():
        rr = pd.read_csv(reg_res_path)

        # ---- 1. performance comparison ----
        st.subheader("Performance Comparison")
        best_row = rr.sort_values("RMSE").iloc[0]
        st.success(f"Best model: **{best_row['Model']}** "
                   f"(RMSE {best_row['RMSE']:.2f}, R² {best_row['R2']:.3f})")
        a, b = st.columns(2)
        a.plotly_chart(px.bar(rr.sort_values("RMSE"), x="Model", y="RMSE",
                              title="RMSE by model (lower is better)"),
                       use_container_width=True, key="reg_rmse")
        b.plotly_chart(px.bar(rr.sort_values("R2", ascending=False), x="Model", y="R2",
                              title="R² by model (higher is better)"),
                       use_container_width=True, key="reg_r2")
        st.divider()

        # ---- 2. feature importance ----
        st.subheader("Feature Importance")
        show_importance("reg_feature_importance.csv",
                        "Feature importance (regression model)", "reg_imp")
        st.divider()

        # ---- 3. detailed metrics table ----
        st.subheader("Detailed Metrics Table")
        st.dataframe(rr.sort_values("RMSE"), use_container_width=True, hide_index=True)
    else:
        st.info("regression_results.csv not found in results/.")


# ====================== 2. CLASSIFICATION ======================
with tab_c:
    st.header("2. Classification: High-Demand Session Prediction")
    st.caption(f"Predicts whether a session is High Demand (Demand ≥ median "
               f"{median_demand:.2f}) or Normal Demand. "
               f"Models are tested on the last 20% of sessions (time-based split).")

    clf_res_path = RES / "classification_results.csv"
    if clf_res_path.exists():
        cr = pd.read_csv(clf_res_path)

        # ---- 1. performance comparison ----
        st.subheader("Performance Comparison")
        best_c = cr.sort_values("ROC_AUC", ascending=False).iloc[0]
        st.success(f"Best model by ROC-AUC: **{best_c['Model']}** "
                   f"(ROC-AUC {best_c['ROC_AUC']:.3f}, F1 {best_c['F1']:.3f})")
        long = cr.melt(id_vars="Model", var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                               title="All metrics by model"),
                        use_container_width=True, key="clf_perf")
        st.divider()

        # ---- 2. feature importance ----
        st.subheader("Feature Importance")
        show_importance("clf_feature_importance.csv",
                        "Feature importance (classification model)", "clf_imp")
        st.divider()

        # ---- 3. detailed metrics table ----
        st.subheader("Detailed Metrics Table")
        st.dataframe(cr.sort_values("ROC_AUC", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info("classification_results.csv not found in results/.")


# ====================== DEMAND PREDICTION (Regression + Classification) ======================
with tab_p:
    st.header("Demand Prediction")
    st.markdown(
        "Describe one charging session below. One click gives you **both** answers:\n"
        "- **Regression:** how much energy the session will need (a number).\n"
        "- **Classification:** whether it is a **High Demand** or **Normal Demand** "
        f"session (High = at or above the typical session of {median_demand:.2f})."
    )
    row_p = session_inputs("pred")

    if st.button("Predict", type="primary", key="btn_pred"):
        pred = float(reg_model.predict(row_p)[0])
        proba = float(clf_model.predict_proba(row_p)[0, 1])

        # same session at every hour of the day, to show the daily pattern
        day_rows = pd.concat([row_p] * 24, ignore_index=True)
        day_rows["Hour"] = list(range(24))
        day_rows["TimeOfDay"] = day_rows["Hour"].apply(time_of_day)
        curve = pd.DataFrame({
            "Hour": list(range(24)),
            "Predicted demand": reg_model.predict(day_rows),
            "P(High demand) %": clf_model.predict_proba(day_rows)[:, 1] * 100,
        })
        st.session_state["pred_result"] = {
            "pred": pred, "proba": proba, "row": row_p.iloc[0].to_dict(), "curve": curve,
        }

    res = st.session_state.get("pred_result")
    if res:
        pred, proba, curve = res["pred"], res["proba"], res["curve"]
        used = res["row"]
        hour_used = int(used["Hour"])
        is_high = proba >= 0.5
        diff = pred - median_demand

        st.divider()
        st.subheader("Result")
        st.caption("Results are for the inputs you had when you clicked Predict. "
                   "Change the inputs and click Predict again to refresh.")

        # ---- plain-language summary ----
        msg = (f"This session is predicted to need **{pred:.2f}**, which is "
               f"**{abs(diff):.2f} {'above' if diff >= 0 else 'below'}** the typical "
               f"session ({median_demand:.2f}). The classifier gives a "
               f"**{proba * 100:.0f}%** chance of High Demand, so it is labelled "
               f"**{'High Demand' if is_high else 'Normal Demand'}**.")
        (st.warning if is_high else st.success)(msg)

        # ---- key numbers ----
        k = st.columns(4)
        k[0].metric("Predicted demand", f"{pred:.2f}", f"{diff:+.2f} vs typical")
        k[1].metric("Typical session (median)", f"{median_demand:.2f}")
        k[2].metric("P(High demand)", f"{proba * 100:.1f}%")
        k[3].metric("Class", "High Demand" if is_high else "Normal Demand")

        # ---- gauges ----
        gmax = float(max(np.ceil(df["Demand"].max() / 10) * 10, pred * 1.1))
        g1, g2 = st.columns(2)
        g1.plotly_chart(gauge(pred, "Regression: predicted demand", gmax,
                              median_demand, ".2f"), use_container_width=True)
        g1.caption("Blue zone = below the typical session, orange zone = above it. "
                   "The red line is the median.")
        g2.plotly_chart(gauge(proba * 100, "Classification: P(High demand)", 100,
                              50, ".1f", "%"), use_container_width=True)
        g2.caption("Past the red line (50%) the session is labelled High Demand.")

        # ---- comparison chart ----
        cmp_df = pd.DataFrame({
            "Item": ["This session (predicted)", "Typical session (median)",
                     "User avg past demand", "Charger avg past demand"],
            "Demand": [pred, median_demand, float(used["User_AvgPastDemand"]),
                       float(used["Charger_AvgPastDemand"])],
        })
        fig_cmp = px.bar(cmp_df, x="Demand", y="Item", orientation="h", color="Item",
                         title="How this session compares")
        fig_cmp.update_layout(showlegend=False, yaxis_title="", height=300)
        fig_cmp.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.caption("Past behaviour is the strongest signal for the model, so the "
                   "prediction usually sits close to the user and charger averages.")

        # ---- 24-hour curves ----
        st.markdown(f"**Same session, different start hour** (you chose {hour_used}:00)")
        c1, c2 = st.columns(2)
        f1_ = px.line(curve, x="Hour", y="Predicted demand", markers=True,
                      title="Predicted demand by start hour")
        f1_.add_hline(y=median_demand, line_dash="dash", line_color="red",
                      annotation_text="typical")
        f1_.add_trace(go.Scatter(x=[hour_used], y=[pred], mode="markers",
                                 name="Your hour",
                                 marker=dict(size=14, color="green", symbol="star")))
        c1.plotly_chart(f1_, use_container_width=True)

        f2_ = px.line(curve, x="Hour", y="P(High demand) %", markers=True,
                      title="Chance of High Demand by start hour")
        f2_.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50%")
        f2_.add_trace(go.Scatter(x=[hour_used], y=[proba * 100], mode="markers",
                                 name="Your hour",
                                 marker=dict(size=14, color="green", symbol="star")))
        f2_.update_yaxes(range=[0, 100])
        c2.plotly_chart(f2_, use_container_width=True)

        best_h = int(curve.loc[curve["Predicted demand"].idxmax(), "Hour"])
        low_h = int(curve.loc[curve["Predicted demand"].idxmin(), "Hour"])
        st.caption(f"The green star is the hour you chose. With everything else the "
                   f"same, demand is predicted highest around {best_h}:00 and lowest "
                   f"around {low_h}:00. Above the red line means more likely High Demand.")


# ====================== DATA EXPLORE ======================
with tab_d:
    st.header("Data Explore")
    st.caption("Uses the full processed dataset, so sidebar filters do not apply.")

    # ---- 1. basic characteristics ----
    st.subheader("1. Basic characteristics")
    c = st.columns(4)
    c[0].metric("Rows (sessions)", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Missing values", int(df.isnull().sum().sum()))
    c[3].metric("Duplicate rows", int(df.duplicated().sum()))
    st.caption(
        f"Data covers {df['StartDatetime'].min():%d %b %Y} to "
        f"{df['StartDatetime'].max():%d %b %Y}. Each row is one charging session."
    )
    st.divider()

    # ---- 2. data dictionary ----
    st.subheader("2. What each column means")
    dictionary = pd.DataFrame([
        ("UserID", "Original", "Identifier of the user who charged"),
        ("ChargerID", "Original", "Identifier of the charger used"),
        ("ChargerCompany", "Original", "Company operating the charger (coded as a number)"),
        ("Location", "Original", "Type of place where the charger is installed"),
        ("ChargerType", "Original", "Type of charger (coded as a number)"),
        ("StartDatetime / EndDatetime", "Original", "Start and end time of the session"),
        ("Duration", "Original", "Length of the charging session"),
        ("Demand", "Original (target)", "Charging demand of the session (Regression target)"),
        ("Hour, Day, Month, DayOfWeek", "Engineered", "Parts of the start time (0 = Monday)"),
        ("IsWeekend", "Engineered", "1 if the session started on Saturday or Sunday"),
        ("TimeOfDay", "Engineered", "Morning, Afternoon, Evening or Night"),
        ("User_PrevSessions / User_AvgPastDemand", "Engineered", "User's earlier session count and average demand"),
        ("Charger_PrevSessions / Charger_AvgPastDemand", "Engineered", "Charger's earlier session count and average demand"),
        ("IsNewUser", "Engineered", "1 if this is the user's first session"),
        ("HighDemand", "Engineered (target)", "1 if Demand >= median, else 0 (Classification target)"),
    ], columns=["Column", "Type", "Meaning"])
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
    st.divider()

    # ---- 3. summary statistics ----
    st.subheader("3. Summary statistics")
    num_all = ["Demand", "Duration", "User_PrevSessions", "User_AvgPastDemand",
               "Charger_PrevSessions", "Charger_AvgPastDemand"]
    st.dataframe(df[num_all].describe().T.round(2), use_container_width=True)
    st.divider()

    
    
    # ---- 6. data preview ----
    st.subheader("6. Preview the data")
    st.caption("Showing all data. Use the filters to narrow it down "
               "(leave a filter empty to include everything).")

    f1, f2, f3, f4 = st.columns(4)
    sel_loc = f1.multiselect("Location", sorted(df["Location"].unique()), key="de_f_loc")
    sel_type = f2.multiselect("Charger type", sorted(df["ChargerType"].unique()), key="de_f_type")
    sel_comp = f3.multiselect("Charger company", sorted(df["ChargerCompany"].unique()), key="de_f_comp")
    sel_tod = f4.multiselect("Time of day", ["Morning", "Afternoon", "Evening", "Night"], key="de_f_tod")

    g1, g2, g3 = st.columns(3)
    sel_dates = g1.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax,
                              key="de_f_dates")
    d_lo, d_hi = float(df["Demand"].min()), float(df["Demand"].max())
    sel_dem = g2.slider("Demand range", d_lo, d_hi, (d_lo, d_hi), key="de_f_dem")
    sel_id = g3.text_input("User ID or Charger ID", key="de_f_id")

    view = df
    if sel_loc:
        view = view[view["Location"].isin(sel_loc)]
    if sel_type:
        view = view[view["ChargerType"].isin(sel_type)]
    if sel_comp:
        view = view[view["ChargerCompany"].isin(sel_comp)]
    if sel_tod:
        view = view[view["TimeOfDay"].isin(sel_tod)]
    if len(sel_dates) == 2:
        view = view[(view["StartDatetime"].dt.date >= sel_dates[0])
                    & (view["StartDatetime"].dt.date <= sel_dates[1])]
    view = view[view["Demand"].between(sel_dem[0], sel_dem[1])]
    if sel_id.strip().isdigit():
        n_id = int(sel_id.strip())
        view = view[(view["UserID"] == n_id) | (view["ChargerID"] == n_id)]

    st.write(f"Showing **{len(view):,}** of {len(df):,} rows.")
    if view.empty:
        st.warning("No rows match these filters.")
    else:
        st.dataframe(view, use_container_width=True, height=500)
        st.download_button("Download filtered data (CSV)",
                           view.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_data.csv", mime="text/csv",
                           key="de_download")