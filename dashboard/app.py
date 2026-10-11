"""EV Charging Demand Dashboard (Streamlit)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from sklearn.metrics import (confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score,
                             roc_curve)
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

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


@st.cache_data
def get_test_predictions():
    """Predictions on the same last-20% time-based test set used in training."""
    d = load_data()
    reg, clf = load_models()
    test = d.iloc[int(len(d) * 0.8):]
    X_test = test[FEATURES]
    out = pd.DataFrame({
        "Actual": test["Demand"].values,
        "Predicted": reg.predict(X_test),
        "ActualClass": test["HighDemand"].values,
        "ProbHigh": clf.predict_proba(X_test)[:, 1],
    })
    return out


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

def show_lime(key):
    """LIME explanation of one session, drawn as a native chart."""
    p = RES / "lime_example.csv"
    if not p.exists():
        st.info("Run the last cell of 08_explainability.ipynb to create lime_example.csv.")
        return
    lime = pd.read_csv(p)
    pred = lime["Predicted"].iloc[0]
    lime = lime.iloc[::-1].copy()
    lime["Direction"] = np.where(lime["Effect"] >= 0,
                                 "Pushes demand up", "Pushes demand down")
    fig = px.bar(lime, x="Effect", y="Condition", orientation="h", color="Direction",
                 color_discrete_map={"Pushes demand up": "#2ca02c",
                                     "Pushes demand down": "#d62728"},
                 title=f"LIME: top 10 factors for this session (predicted demand {pred:.2f})")
    fig.update_layout(yaxis_title="", xaxis_title="Effect on predicted demand")
    a, b = st.columns([3, 1])
    a.plotly_chart(fig, use_container_width=True, key=f"lime_chart_{key}")
    sp = RES / "lime_session.csv"
    if sp.exists():
        b.markdown("**Session values**")
        b.dataframe(pd.read_csv(sp), use_container_width=True, hide_index=True, height=480)
    st.caption("Green bars push the predicted demand up, red bars push it down. "
               "Conditions like 'Charger_AvgPastDemand > 0.56' are on scaled values, "
               "so it means above average.")

# ====================== Sidebar filters ======================
st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to Overview and Time Series. The Regression and "
                   "Classification tabs always use the fixed 80/20 time-based test set.")
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

tab_o, tab_d, tab_r, tab_c, tab_t, tab_m = st.tabs(
    ["Overview", "Data Explore", "1. Regression", "2. Classification",
     "3. Time Series", "Models & Explainability"])

# ====================== OVERVIEW ======================
with tab_o:
    st.markdown(
        "This page gives a quick summary of EV charging activity for the "
        "filters selected in the sidebar. **Demand** is the energy requested "
        "in a charging session."
    )
    c = st.columns(5)
    c[0].metric("Sessions", f"{len(fdf):,}")
    c[1].metric("Total demand", f"{fdf['Demand'].sum():,.0f}")
    c[2].metric("Avg demand / session", f"{fdf['Demand'].mean():.2f}")
    c[3].metric("Avg duration", f"{fdf['Duration'].mean():.1f}")
    c[4].metric("High-demand share", f"{fdf['HighDemand'].mean() * 100:.1f}%")

    peak_hour = int(fdf.groupby("Hour")["Demand"].sum().idxmax())
    busy_dow = int(fdf.groupby("DayOfWeek")["Demand"].sum().idxmax())
    c = st.columns(5)
    c[0].metric("Unique users", f"{fdf['UserID'].nunique():,}")
    c[1].metric("Unique chargers", f"{fdf['ChargerID'].nunique():,}")
    c[2].metric("Locations", fdf["Location"].nunique())
    c[3].metric("Peak hour", f"{peak_hour}:00")
    c[4].metric("Busiest day", DAYS[busy_dow])

    st.info(
        f"**Summary:** {len(fdf):,} sessions used {fdf['Demand'].sum():,.0f} "
        f"total demand. The average session needs {fdf['Demand'].mean():.2f}, "
        f"the median is {fdf['Demand'].median():.2f}. "
        f"{fdf['HighDemand'].mean() * 100:.1f}% of sessions are High Demand "
        f"(Demand >= overall median {median_demand:.2f})."
    )
    st.divider()

    daily = fdf.set_index("StartDatetime")["Demand"].resample("D").sum()
    daily_df = pd.DataFrame({"Daily demand": daily,
                             "7-day average": daily.rolling(7).mean()})
    st.plotly_chart(px.line(daily_df, title="Daily total demand with 7-day average"),
                    use_container_width=True)
    fw, lw = daily.head(7).mean(), daily.tail(7).mean()
    chg = (lw - fw) / fw * 100 if fw else 0
    st.caption(
        f"The smooth line is the 7-day average, which removes weekly ups and downs. "
        f"Average daily demand moved from {fw:,.0f} (first 7 days) to {lw:,.0f} "
        f"(last 7 days), a change of {chg:+.1f}%. Sudden drops can be outages or "
        f"missing data."
    )

    monthly = (fdf.groupby(fdf["StartDatetime"].dt.to_period("M"))["Demand"]
                  .sum().reset_index())
    monthly["StartDatetime"] = monthly["StartDatetime"].astype(str)
    monthly["Change %"] = monthly["Demand"].pct_change().mul(100).round(1)
    st.plotly_chart(px.bar(monthly, x="StartDatetime", y="Demand",
                           hover_data=["Change %"], labels={"StartDatetime": "Month"},
                           title="Monthly total demand"), use_container_width=True)
    top_m = monthly.loc[monthly["Demand"].idxmax()]
    st.caption(f"Highest month: {top_m['StartDatetime']} ({top_m['Demand']:,.0f}). "
               f"First and last months may be partial.")
    st.divider()

    a, b = st.columns(2)
    fig = px.histogram(fdf, x="Demand", nbins=50, title="Distribution of demand per session")
    fig.add_vline(x=median_demand, line_dash="dash", line_color="red",
                  annotation_text="median")
    a.plotly_chart(fig, use_container_width=True)
    a.caption(f"Most sessions are small, a few are very large, so the mean "
              f"({fdf['Demand'].mean():.2f}) is above the median "
              f"({fdf['Demand'].median():.2f}). Red line = High/Normal threshold.")

    heat = fdf.pivot_table(index="DayOfWeek", columns="Hour", values="Demand",
                           aggfunc="mean").reindex(range(7))
    heat.index = DAYS
    b.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                             labels=dict(x="Hour", y="Day", color="Avg demand"),
                             title="Avg demand by day and hour"),
                   use_container_width=True)
    b.caption("Darker cells = higher average demand. Use it for capacity planning.")
    st.divider()

    a, b = st.columns(2)
    sess_hour = fdf.groupby("Hour").size().reset_index(name="Sessions")
    a.plotly_chart(px.bar(sess_hour, x="Hour", y="Sessions",
                          title="Number of sessions by hour"), use_container_width=True)
    a.caption(f"Most sessions start around "
              f"{int(sess_hour.loc[sess_hour['Sessions'].idxmax(), 'Hour'])}:00.")

    tod = (fdf.groupby("TimeOfDay")["Demand"].agg(["count", "mean"])
              .reindex(["Morning", "Afternoon", "Evening", "Night"]).reset_index())
    tod.columns = ["TimeOfDay", "Sessions", "Avg demand"]
    b.plotly_chart(px.bar(tod, x="TimeOfDay", y="Avg demand", hover_data=["Sessions"],
                          title="Avg demand by time of day"), use_container_width=True)
    b.caption(f"{tod.loc[tod['Avg demand'].idxmax(), 'TimeOfDay']} has the highest "
              f"average demand. (Morning 5-12, Afternoon 12-17, Evening 17-21, Night 21-5.)")
    st.divider()

    a, b = st.columns(2)
    loc_tot = fdf.groupby("Location")["Demand"].sum().reset_index()
    a.plotly_chart(px.pie(loc_tot, names="Location", values="Demand", hole=0.4,
                          title="Share of total demand by location"),
                   use_container_width=True)
    tl = loc_tot.loc[loc_tot["Demand"].idxmax()]
    a.caption(f"{tl['Location']} contributes the most "
              f"({tl['Demand'] / loc_tot['Demand'].sum() * 100:.1f}% of total).")

    loc_hi = (fdf.groupby("Location")["HighDemand"].mean().mul(100)
                 .sort_values(ascending=False).reset_index())
    loc_hi.columns = ["Location", "High-demand %"]
    b.plotly_chart(px.bar(loc_hi, x="Location", y="High-demand %",
                          title="High-demand sessions (%) by location"),
                   use_container_width=True)
    b.caption(f"{loc_hi.iloc[0]['Location']} has the highest High-Demand share "
              f"({loc_hi.iloc[0]['High-demand %']:.1f}%). Overall is about 50% by definition.")
    st.divider()

    a, b = st.columns(2)
    ct = fdf.groupby("ChargerType").agg(Sessions=("Demand", "size"),
                                        AvgDemand=("Demand", "mean")).reset_index()
    ct["ChargerType"] = ct["ChargerType"].astype(str)
    a.plotly_chart(px.bar(ct, x="ChargerType", y="AvgDemand", hover_data=["Sessions"],
                          title="Avg demand by charger type"), use_container_width=True)
    cc = fdf.groupby("ChargerCompany").agg(Sessions=("Demand", "size"),
                                           AvgDemand=("Demand", "mean")).reset_index()
    cc["ChargerCompany"] = cc["ChargerCompany"].astype(str)
    b.plotly_chart(px.bar(cc, x="ChargerCompany", y="AvgDemand", hover_data=["Sessions"],
                          title="Avg demand by charger company"), use_container_width=True)
    st.caption("Hover a bar to see session counts. A high average with very few "
               "sessions is not reliable.")

    top_ch = fdf.groupby("ChargerID")["Demand"].sum().nlargest(10).reset_index()
    top_ch["ChargerID"] = "Charger " + top_ch["ChargerID"].astype(str)
    st.plotly_chart(px.bar(top_ch, x="Demand", y="ChargerID", orientation="h",
                           title="Top 10 chargers by total demand")
                    .update_yaxes(autorange="reversed"), use_container_width=True)
    st.caption("These chargers carry the most load: check them first for maintenance "
               "and extra capacity nearby.")


# ====================== 1. REGRESSION ======================
with tab_r:
    st.header("1. Regression: Charging Demand Prediction")
    st.markdown(
        """
**Goal:** predict the **charging demand** (a number) of an EV charging session.

**Inputs (features):** charger type, location, charger company, time of charging
(hour, day, month, day of week, weekend, time of day) and historical usage
(user and charger past sessions and past average demand).

**Models compared:** Linear Regression, Decision Tree, Random Forest, Gradient
Boosting, XGBoost, LightGBM.

**Evaluation:** MAE, MSE, RMSE and R². The split is time-based (first 80% train,
last 20% test) and `Duration` / end-time columns are **not** used, because they are
only known after charging ends (data leakage).
        """
    )
    with st.expander("How to read the metrics"):
        st.markdown(
            "- **MAE**: average error in demand units. Lower is better.\n"
            "- **MSE**: average squared error. Punishes big mistakes.\n"
            "- **RMSE**: square root of MSE, in demand units. Lower is better.\n"
            "- **R²**: share of the variation explained by the model. 1 = perfect, "
            "0 = no better than predicting the average."
        )
    st.divider()

    # ---- model comparison ----
    st.subheader("Model comparison")
    reg_res_path = RES / "regression_results.csv"
    if reg_res_path.exists():
        rr = pd.read_csv(reg_res_path)
        best_row = rr.sort_values("RMSE").iloc[0]
        st.success(f"Best model: **{best_row['Model']}** "
                   f"(RMSE {best_row['RMSE']:.2f}, R² {best_row['R2']:.3f})")
        st.dataframe(rr, use_container_width=True)
        a, b = st.columns(2)
        a.plotly_chart(px.bar(rr.sort_values("RMSE"), x="Model", y="RMSE",
                              title="RMSE by model (lower is better)"),
                       use_container_width=True)
        b.plotly_chart(px.bar(rr.sort_values("R2", ascending=False), x="Model", y="R2",
                              title="R² by model (higher is better)"),
                       use_container_width=True)
    else:
        st.info("regression_results.csv not found in results/.")

    # ---- best model on the test set ----
    st.subheader(f"Best model on the test set ({model_name(reg_model)})")
    tp = get_test_predictions()
    mse = mean_squared_error(tp["Actual"], tp["Predicted"])
    m = st.columns(4)
    m[0].metric("MAE", f"{mean_absolute_error(tp['Actual'], tp['Predicted']):.2f}")
    m[1].metric("MSE", f"{mse:.2f}")
    m[2].metric("RMSE", f"{np.sqrt(mse):.2f}")
    m[3].metric("R²", f"{r2_score(tp['Actual'], tp['Predicted']):.3f}")

    a, b = st.columns(2)
    samp = tp.sample(min(3000, len(tp)), random_state=42)
    fig = px.scatter(samp, x="Actual", y="Predicted", opacity=0.4,
                     title="Actual vs predicted demand (3,000 test sessions)")
    mx = float(max(samp["Actual"].max(), samp["Predicted"].max()))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                             name="Perfect prediction", line=dict(color="red", dash="dash")))
    a.plotly_chart(fig, use_container_width=True)
    a.caption("Points on the red line are perfect predictions. The spread around it "
              "shows the error. Large sessions are the hardest to predict.")

    resid = tp["Actual"] - tp["Predicted"]
    b.plotly_chart(px.histogram(resid, nbins=60, title="Prediction error (Actual - Predicted)")
                   .update_layout(showlegend=False), use_container_width=True)
    b.caption("Centered near 0 means the model is not biased. A long right tail means "
              "it under-predicts some very large sessions.")
    st.divider()

    # ---- important factors ----
    st.subheader("Important factors influencing demand")
    ri = RES / "reg_feature_importance.csv"
    a, b = st.columns(2)
    if ri.exists():
        imp = pd.read_csv(ri, index_col=0).head(15).sort_values("importance")
        a.plotly_chart(px.bar(imp, x="importance", y=imp.index, orientation="h",
                              title="Feature importance (how often used in splits)"),
                       use_container_width=True)
    if (RES / "shap_bar.png").exists():
        b.image(str(RES / "shap_bar.png"), caption="SHAP: average impact on predictions")
    st.markdown(
        "**Finding:** the history of the user and the charger "
        "(`User_AvgPastDemand`, `Charger_AvgPastDemand`) matters most, then the hour "
        "of day and the number of past sessions. Night sessions push demand up. "
        "Location, company and month have a small effect. SHAP is the main result "
        "because it measures the real impact on predictions."
    )
    if (RES / "shap_summary.png").exists():
        st.image(str(RES / "shap_summary.png"),
                 caption="SHAP summary: red = high feature value, blue = low. "
                         "Right side = pushes demand up.")
        st.markdown("**LIME: explanation of one session**")
    show_lime("reg")
    st.divider()

    # ---- predict ----
    st.subheader("Predict demand for a session")
    row_r = session_inputs("reg")
    if st.button("Predict demand", key="btn_reg"):
        pred = float(reg_model.predict(row_r)[0])
        st.metric("Predicted demand", f"{pred:.2f}")
        st.caption(f"Typical session (median) is {median_demand:.2f}.")


# ====================== 2. CLASSIFICATION ======================
with tab_c:
    st.header("2. Classification: High-Demand Session Prediction")
    st.markdown(
        f"""
**Goal:** decide whether a charging session is **High Demand** or **Normal Demand**.

**Target:**
- **High Demand:** Demand ≥ median demand ({median_demand:.2f})
- **Normal Demand:** Demand < median demand

**Models compared:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM,
XGBoost, LightGBM.

**Evaluation:** Accuracy, Precision, Recall, F1-score and ROC-AUC.

**Class imbalance:** checked first. Because the threshold is the median, the two
classes are about 50/50, so Random Oversampling, SMOTE and ADASYN are **not required**.
        """
    )
    with st.expander("How to read the metrics"):
        st.markdown(
            "- **Accuracy**: share of all sessions classified correctly.\n"
            "- **Precision**: of sessions predicted High, how many really were High.\n"
            "- **Recall**: of the real High sessions, how many the model found.\n"
            "- **F1**: balance of precision and recall.\n"
            "- **ROC-AUC**: how well the model ranks High above Normal. 0.5 = random, 1 = perfect."
        )
    st.divider()

    # ---- class balance ----
    st.subheader("Step 1: Check class balance")
    bal = df["HighDemand"].map({1: "High Demand", 0: "Normal Demand"}).value_counts().reset_index()
    bal.columns = ["Class", "Sessions"]
    a, b = st.columns([1, 2])
    a.plotly_chart(px.pie(bal, names="Class", values="Sessions", hole=0.4,
                          title="Class distribution"), use_container_width=True)
    b.dataframe(bal.assign(Share=(bal["Sessions"] / bal["Sessions"].sum() * 100)
                           .round(2).astype(str) + "%"), use_container_width=True)
    b.success("Classes are balanced (about 50/50), so no oversampling was needed.")
    st.divider()

    # ---- model comparison ----
    st.subheader("Step 2: Model comparison")
    clf_res_path = RES / "classification_results.csv"
    if clf_res_path.exists():
        cr = pd.read_csv(clf_res_path)
        best_c = cr.sort_values("ROC_AUC", ascending=False).iloc[0]
        st.success(f"Best model by ROC-AUC: **{best_c['Model']}** "
                   f"(ROC-AUC {best_c['ROC_AUC']:.3f}, F1 {best_c['F1']:.3f})")
        st.dataframe(cr, use_container_width=True)
        long = cr.melt(id_vars="Model", var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                               title="All metrics by model"), use_container_width=True)
        st.caption("The top models are very close. Pick on the metric that matters: "
                   "ROC-AUC for ranking quality, Recall if missing a High-demand "
                   "session is costly.")
    else:
        st.info("classification_results.csv not found in results/.")

    # ---- best model on test set ----
    st.subheader(f"Step 3: Best model on the test set ({model_name(clf_model)})")
    tp = get_test_predictions()
    y_true = tp["ActualClass"]
    thr = st.slider("Decision threshold for 'High Demand'", 0.1, 0.9, 0.5, 0.05, key="clf_thr")
    y_pred = (tp["ProbHigh"] >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp_ = cm.ravel()
    acc = (tn + tp_) / cm.sum()
    prec = tp_ / (tp_ + fp) if (tp_ + fp) else 0
    rec = tp_ / (tp_ + fn) if (tp_ + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    m = st.columns(5)
    m[0].metric("Accuracy", f"{acc:.3f}")
    m[1].metric("Precision", f"{prec:.3f}")
    m[2].metric("Recall", f"{rec:.3f}")
    m[3].metric("F1", f"{f1:.3f}")
    m[4].metric("ROC-AUC", f"{roc_auc_score(y_true, tp['ProbHigh']):.3f}")

    a, b = st.columns(2)
    a.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                             x=["Pred Normal", "Pred High"], y=["Actual Normal", "Actual High"],
                             title="Confusion matrix"), use_container_width=True)
    a.caption("Diagonal = correct predictions. Off-diagonal = mistakes. Moving the "
              "threshold trades precision against recall.")
    fpr, tpr, _ = roc_curve(y_true, tp["ProbHigh"])
    roc = go.Figure()
    roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model"))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color="red")))
    roc.update_layout(title="ROC curve", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    b.plotly_chart(roc, use_container_width=True)
    b.caption("The further the curve is above the red diagonal, the better the model.")
    st.divider()

    # ---- factors ----
    st.subheader("Step 4: Factors associated with High-Demand sessions")
    ci = RES / "clf_feature_importance.csv"
    if ci.exists():
        imp = pd.read_csv(ci, index_col=0).head(15).sort_values("importance")
        st.plotly_chart(px.bar(imp, x="importance", y=imp.index, orientation="h",
                               title="Feature importance (best classifier)"),
                        use_container_width=True)
    st.markdown(
        "**Finding:** the same factors as regression lead: the user's and the "
        "charger's past average demand, then the hour, past session counts and "
        "night-time charging. Location and company add little."
    )
    st.divider()

    # ---- predict ----
    st.subheader("Predict whether a session is High Demand")
    row_c = session_inputs("clf")
    if st.button("Classify session", key="btn_clf"):
        proba = float(clf_model.predict_proba(row_c)[0, 1])
        r = st.columns(2)
        r[0].metric("P(High demand)", f"{proba * 100:.1f}%")
        r[1].metric("Class", "High Demand" if proba >= 0.5 else "Normal Demand")
        st.caption(f"High Demand means Demand ≥ median ({median_demand:.2f}).")


# ====================== 3. TIME SERIES ======================
with tab_t:
    st.header("3. Time Series: Future Charging Demand Forecasting")
    st.markdown(
        """
**Goal:** understand how total demand changes over time and forecast future demand.

**Steps on this page:**
1. Convert timestamps into a time series and aggregate demand per day
2. Analyze **trend, seasonality and noise**
3. Check whether the series is **stationary** (ADF test)
4. Apply **Moving Average** and **Exponential Smoothing**, evaluated with **MAE and RMSE**
5. Identify periods of **increasing or decreasing** demand
6. Forecast future demand

This page follows the sidebar filters.
        """
    )
    drop_edges = st.checkbox("Drop the first and last day (they may be partial days)", value=True)

    daily_s = fdf.set_index("StartDatetime")["Demand"].resample("D").sum()
    if drop_edges and len(daily_s) > 2:
        daily_s = daily_s.iloc[1:-1]

    if len(daily_s) < 60:
        st.warning("Not enough days for time series analysis. Widen the filters (need 60+ days).")
    else:
        st.divider()
        # ---- step 1 ----
        st.subheader("Step 1: Daily demand time series")
        st.plotly_chart(px.line(daily_s.rename("Demand"), title="Daily total demand")
                        .update_layout(showlegend=False), use_container_width=True)
        hr = fdf.groupby("Hour")["Demand"].sum().reset_index()
        st.plotly_chart(px.bar(hr, x="Hour", y="Demand", title="Total demand by hour of day"),
                        use_container_width=True)
        st.caption(f"{len(daily_s)} daily points. Demand is aggregated by day (the hourly "
                   f"pattern is shown above).")
        st.divider()

        # ---- step 2 ----
        st.subheader("Step 2: Trend, seasonality and noise")
        dec = seasonal_decompose(daily_s, model="additive", period=7)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=["Observed", "Trend", "Seasonal (weekly)", "Residual (noise)"])
        for i, comp in enumerate([dec.observed, dec.trend, dec.seasonal, dec.resid], start=1):
            fig.add_trace(go.Scatter(x=comp.index, y=comp.values, mode="lines" if i < 4 else "markers",
                                     showlegend=False), row=i, col=1)
        fig.update_layout(height=750)
        st.plotly_chart(fig, use_container_width=True)
        seas_amp = float(dec.seasonal.max() - dec.seasonal.min())
        st.markdown(
            f"- **Trend:** the long-term direction (smooth line).\n"
            f"- **Seasonality:** a repeating weekly cycle, swinging about {seas_amp:,.0f} "
            f"from lowest to highest day.\n"
            f"- **Noise:** what is left after removing trend and season. Very large "
            f"residuals are outlier days (outages or missing data)."
        )
        st.divider()

        # ---- step 3 ----
        st.subheader("Step 3: Stationarity check (ADF test)")
        p_raw = adfuller(daily_s.dropna())[1]
        p_diff = adfuller(daily_s.diff().dropna())[1]
        a, b = st.columns(2)
        a.metric("p-value (original)", f"{p_raw:.4f}")
        a.write("Stationary" if p_raw < 0.05 else "**Not stationary** (has a trend)")
        b.metric("p-value (after differencing)", f"{p_diff:.4f}")
        b.write("Stationary" if p_diff < 0.05 else "Still not stationary")
        st.caption("p-value < 0.05 means stationary (stable mean over time). A trending "
                   "series is not stationary, but differencing once usually fixes it.")
        st.divider()

        # ---- step 4 ----
        st.subheader("Step 4: Forecasting models (Moving Average and Exponential Smoothing)")
        test_pct = st.slider("Test period (% of days, from the end)", 10, 30, 20, 5)
        n_test = max(int(len(daily_s) * test_pct / 100), 7)
        train, test = daily_s.iloc[:-n_test], daily_s.iloc[-n_test:]
        c1, c2 = st.columns(2)
        w = c1.slider("Moving average window (days)", 2, 30, 7)
        alpha = c2.slider("Exponential smoothing alpha", 0.05, 1.0, 0.3, 0.05)

        preds = {}
        for win in sorted({3, 7, 14, w}):
            preds[f"Moving Average ({win}d)"] = (
                daily_s.rolling(win).mean().shift(1).loc[test.index])
        ses = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
        preds[f"Simple Exp Smoothing (alpha={alpha})"] = pd.Series(
            ses.forecast(len(test)).values, index=test.index)
        try:
            hw = ExponentialSmoothing(train, trend="add", seasonal="add",
                                      seasonal_periods=7).fit()
            preds["Holt-Winters"] = pd.Series(hw.forecast(len(test)).values, index=test.index)
        except Exception:
            st.caption("Holt-Winters could not be fitted on this selection.")

        rows = []
        for name, p in preds.items():
            ok = p.notna()
            rows.append({"Model": name,
                         "MAE": mean_absolute_error(test[ok], p[ok]),
                         "RMSE": np.sqrt(mean_squared_error(test[ok], p[ok]))})
        score_df = pd.DataFrame(rows).sort_values("RMSE").round(2)
        st.success(f"Best model by RMSE: **{score_df.iloc[0]['Model']}**")
        a, b = st.columns([1, 1])
        a.dataframe(score_df, use_container_width=True)
        b.plotly_chart(px.bar(score_df, x="Model", y=["MAE", "RMSE"], barmode="group",
                              title="Forecast error (lower is better)"),
                       use_container_width=True)

        show = st.multiselect("Models to plot", list(preds.keys()),
                              default=list(preds.keys())[:1] + list(preds.keys())[-1:])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual",
                                 line=dict(color="black")))
        for name in show:
            fig.add_trace(go.Scatter(x=preds[name].index, y=preds[name].values, name=name))
        fig.update_layout(title="Forecast vs actual (test period)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Moving Average is one-step-ahead (each day is predicted from the "
                   "previous days), while Exponential Smoothing and Holt-Winters forecast "
                   "the whole test period at once, which is harder. Keep this in mind "
                   "when comparing them.")
        st.divider()

        # ---- step 5 ----
        st.subheader("Step 5: Increasing and decreasing periods")
        mon = daily_s.resample("MS").sum().reset_index()
        mon.columns = ["Month", "Demand"]
        mon["Change %"] = mon["Demand"].pct_change().mul(100).round(1)
        mon["Direction"] = np.where(mon["Change %"] >= 0, "Increase", "Decrease")
        mon["Month"] = mon["Month"].dt.strftime("%Y-%m")
        mc = mon.dropna()
        st.plotly_chart(px.bar(mc, x="Month", y="Change %", color="Direction",
                               color_discrete_map={"Increase": "green", "Decrease": "red"},
                               title="Month-on-month change in total demand (%)"),
                        use_container_width=True)
        if not mc.empty:
            up, dn = mc.loc[mc["Change %"].idxmax()], mc.loc[mc["Change %"].idxmin()]
            st.markdown(f"- Biggest increase: **{up['Month']}** ({up['Change %']:+.1f}%)\n"
                        f"- Biggest decrease: **{dn['Month']}** ({dn['Change %']:+.1f}%)")
        st.caption("The first and last months may be partial months.")
        st.divider()

        # ---- step 6 ----
        st.subheader("Step 6: Future demand forecast")
        horizon = st.slider("Forecast horizon (days)", 7, 60, 14)
        try:
            final = ExponentialSmoothing(daily_s, trend="add", seasonal="add",
                                         seasonal_periods=7).fit()
            label = "Holt-Winters"
        except Exception:
            final = SimpleExpSmoothing(daily_s).fit()
            label = "Simple Exp Smoothing"
        fc = final.forecast(horizon)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_s.tail(90).index, y=daily_s.tail(90).values,
                                 name="Actual (last 90 days)"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name=f"{label} forecast",
                                 line=dict(dash="dash", color="red")))
        fig.update_layout(title=f"{horizon}-day demand forecast")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Forecast average: {fc.mean():,.0f} per day, compared with "
                   f"{daily_s.tail(14).mean():,.0f} per day over the last 14 days. "
                   f"Forecasts get less reliable the further ahead they go.")
        # ====================== DATA EXPLORE ======================
with tab_d:
    st.header("Data Explore")
    st.markdown(
        "This page explains the dataset before any modeling: what each column means, "
        "its quality (missing values, duplicates, outliers) and how the columns relate. "
        "It uses the full processed dataset, so sidebar filters do not apply."
    )
    st.divider()

    # ---- 1. basic characteristics ----
    st.subheader("1. Basic characteristics")
    c = st.columns(5)
    c[0].metric("Rows (sessions)", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Missing values", int(df.isnull().sum().sum()))
    c[3].metric("Duplicate rows", int(df.duplicated().sum()))
    c[4].metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
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
        ("StartDay / StartTime", "Original", "Date and clock time charging started"),
        ("EndDay / EndTime", "Original", "Date and clock time charging ended"),
        ("StartDatetime / EndDatetime", "Original", "Start and end combined into one timestamp"),
        ("Duration", "Original", "Length of the charging session"),
        ("Demand", "Original (target)", "Charging demand of the session. Predicted in Regression"),
        ("Hour, Day, Month, DayOfWeek", "Engineered", "Parts of the start time (DayOfWeek: 0 = Monday)"),
        ("IsWeekend", "Engineered", "1 if the session started on Saturday or Sunday"),
        ("TimeOfDay", "Engineered", "Morning 5-12, Afternoon 12-17, Evening 17-21, Night 21-5"),
        ("User_PrevSessions", "Engineered", "Number of earlier sessions by this user"),
        ("User_AvgPastDemand", "Engineered", "User's average demand over earlier sessions only"),
        ("Charger_PrevSessions", "Engineered", "Number of earlier sessions on this charger"),
        ("Charger_AvgPastDemand", "Engineered", "Charger's average demand over earlier sessions only"),
        ("IsNewUser", "Engineered", "1 if this is the user's first session"),
        ("HighDemand", "Engineered (target)", "1 if Demand >= median demand, else 0. Predicted in Classification"),
    ], columns=["Column", "Type", "Meaning"])
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
    st.caption("History features use only earlier sessions (shifted by one), so the "
               "current session's own demand never leaks into its features.")
    st.divider()

    # ---- 3. data quality ----
    st.subheader("3. Data types, missing values and unique values")
    quality = pd.DataFrame({
        "Data type": df.dtypes.astype(str),
        "Missing": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2),
        "Unique values": df.nunique(),
    })
    st.dataframe(quality, use_container_width=True)
    st.caption("No missing values remain after cleaning. `ChargerCompany` and "
               "`ChargerType` are numbers but are really categories, so the models "
               "treat them as categories, not as quantities.")
    st.divider()

    # ---- 4. summary statistics ----
    st.subheader("4. Summary statistics")
    num_all = ["Demand", "Duration", "User_PrevSessions", "User_AvgPastDemand",
               "Charger_PrevSessions", "Charger_AvgPastDemand"]
    st.dataframe(df[num_all].describe().T.round(2), use_container_width=True)
    sk = df[["Demand", "Duration"]].skew().round(2)
    st.caption(
        f"Skewness: Demand {sk['Demand']}, Duration {sk['Duration']}. A value well "
        f"above 0 means a long right tail: most sessions are small and a few are "
        f"very large, so the mean is higher than the median."
    )
    st.divider()

    # ---- 5. distribution explorer ----
    st.subheader("5. Distribution of a numeric column")
    col = st.selectbox("Choose a column", num_all, key="de_num")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x=col, nbins=60, title=f"Histogram of {col}"),
                   use_container_width=True)
    b.plotly_chart(px.box(df, y=col, title=f"Box plot of {col}"),
                   use_container_width=True)
    st.caption("The histogram shows how values are spread. In the box plot, dots "
               "above the whisker are outliers.")
    st.divider()

    # ---- 6. outliers ----
    st.subheader("6. Outliers (IQR method)")
    rows = []
    for cname in ["Demand", "Duration"]:
        q1, q3 = df[cname].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[cname] < lo) | (df[cname] > hi)).sum())
        rows.append({"Column": cname, "Lower limit": round(lo, 2),
                     "Upper limit": round(hi, 2), "Outliers": n,
                     "Outlier %": round(n / len(df) * 100, 2)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("A value is an outlier if it is more than 1.5 x IQR beyond Q1 or Q3. "
               "These were kept in the data because large sessions are real charging "
               "events. Only impossible values (negative, end before start) are errors.")
    st.divider()

    # ---- 7. categorical explorer ----
    st.subheader("7. Categorical columns")
    ccol = st.selectbox("Choose a column",
                        ["Location", "ChargerType", "ChargerCompany", "TimeOfDay"],
                        key="de_cat")
    g = (df.groupby(ccol).agg(Sessions=("Demand", "size"), AvgDemand=("Demand", "mean"),
                              HighDemandPct=("HighDemand", "mean")).reset_index())
    g["HighDemandPct"] = (g["HighDemandPct"] * 100).round(1)
    g[ccol] = g[ccol].astype(str)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(g, x=ccol, y="Sessions", title=f"Sessions by {ccol}"),
                   use_container_width=True)
    b.plotly_chart(px.bar(g, x=ccol, y="AvgDemand", hover_data=["HighDemandPct"],
                          title=f"Average demand by {ccol}"), use_container_width=True)
    st.dataframe(g.round(2), use_container_width=True, hide_index=True)
    st.caption("Left: how common each category is. Right: how much demand it "
               "typically has. Compare both before drawing conclusions.")
    st.divider()

    # ---- 8. correlation ----
    st.subheader("8. Correlation between numeric columns")
    corr_cols = ["Demand", "Duration", "Hour", "Month", "DayOfWeek", "IsWeekend",
                 "User_PrevSessions", "User_AvgPastDemand",
                 "Charger_PrevSessions", "Charger_AvgPastDemand"]
    corr = df[corr_cols].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto",
                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Correlation matrix"), use_container_width=True)
    dc = corr["Demand"].drop("Demand").sort_values(key=abs, ascending=False)
    st.caption(
        f"Values near +1 or -1 mean a strong linear link, near 0 means none. "
        f"The columns most related to Demand are {dc.index[0]} ({dc.iloc[0]:+.2f}) "
        f"and {dc.index[1]} ({dc.iloc[1]:+.2f}). `Duration` is only known after "
        f"charging ends, so it is not used as a model input."
    )
    st.divider()

     # ---- 9. raw data ----
    st.subheader("9. Preview the data")
    mode = st.radio("Show", ["First rows", "Random sample", "Full data"],
                    horizontal=True, key="de_mode")
    if mode == "First rows":
        n = st.slider("Rows to show", 10, 5000, 100, 10, key="de_n")
        view = df.head(n)
    elif mode == "Random sample":
        n = st.slider("Rows to show", 10, 5000, 100, 10, key="de_n2")
        view = df.sample(n, random_state=42).sort_values("StartDatetime")
    else:
        view = df
        st.warning(f"Showing all {len(df):,} rows. It may load slowly.")
    st.dataframe(view, use_container_width=True, height=500)
    st.caption(f"Showing {len(view):,} of {len(df):,} rows. Click a column header to sort, "
               f"and scroll inside the table to see more.")
    st.download_button("Download full processed data (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="featured_data.csv", mime="text/csv")


# ====================== MODELS & EXPLAINABILITY ======================
with tab_m:
    st.header("Models & Explainability")
    st.markdown(
        "This page puts all model results in one place and explains **why** the "
        "models predict what they do, using Feature Importance, SHAP and LIME."
    )
    st.divider()

    # ---- 1. summary ----
    st.subheader("1. Best model for each task")
    summary = []
    if (RES / "regression_results.csv").exists():
        r = pd.read_csv(RES / "regression_results.csv").sort_values("RMSE").iloc[0]
        summary.append({"Task": "Regression (Demand)", "Best model": r["Model"],
                        "Key metrics": f"RMSE {r['RMSE']:.2f}, MAE {r['MAE']:.2f}, R² {r['R2']:.3f}"})
    if (RES / "classification_results.csv").exists():
        r = pd.read_csv(RES / "classification_results.csv").sort_values(
            "ROC_AUC", ascending=False).iloc[0]
        summary.append({"Task": "Classification (High/Normal)", "Best model": r["Model"],
                        "Key metrics": f"ROC-AUC {r['ROC_AUC']:.3f}, F1 {r['F1']:.3f}, Accuracy {r['Accuracy']:.3f}"})
    if (RES / "timeseries_results.csv").exists():
        r = pd.read_csv(RES / "timeseries_results.csv").sort_values("RMSE").iloc[0]
        summary.append({"Task": "Time Series (Daily demand)", "Best model": r["Model"],
                        "Key metrics": f"RMSE {r['RMSE']:.2f}, MAE {r['MAE']:.2f}"})
    if summary:
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    else:
        st.info("No result files found in results/.")
    st.caption("Regression and classification use a time-based split (first 80% "
               "train, last 20% test). Time series results use the last 20% of days.")
    st.divider()

    # ---- 2. full comparison ----
    st.subheader("2. Full model comparison")
    t1, t2, t3 = st.tabs(["Regression", "Classification", "Time Series"])
    with t1:
        p = RES / "regression_results.csv"
        if p.exists():
            rr = pd.read_csv(p).sort_values("RMSE")
            st.dataframe(rr, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(rr, x="Model", y=["MAE", "RMSE"], barmode="group",
                                   title="Regression error (lower is better)"),
                            use_container_width=True)
            st.caption("R² near 0.35 means the features explain about a third of the "
                       "variation in demand. The rest is individual behavior that the "
                       "dataset does not capture.")
    with t2:
        p = RES / "classification_results.csv"
        if p.exists():
            cr = pd.read_csv(p).sort_values("ROC_AUC", ascending=False)
            st.dataframe(cr, use_container_width=True, hide_index=True)
            long = cr.melt(id_vars="Model", var_name="Metric", value_name="Score")
            st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric",
                                   barmode="group", title="Classification metrics"),
                            use_container_width=True)
            st.caption("The top models are very close, so differences of 0.01 are "
                       "small. Classes are about 50/50, so no oversampling was needed.")
    with t3:
        p = RES / "timeseries_results.csv"
        if p.exists():
            tr = pd.read_csv(p).sort_values("RMSE")
            st.dataframe(tr, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(tr, x="Model", y=["MAE", "RMSE"], barmode="group",
                                   title="Forecast error (lower is better)"),
                            use_container_width=True)
            st.caption("Moving Average is one-step-ahead, while Exponential Smoothing "
                       "and Holt-Winters forecast the whole test period at once, so "
                       "the comparison favors Moving Average.")
    st.divider()

    # ---- 3. feature importance ----
    st.subheader("3. Feature importance")
    st.markdown("Shows which inputs the models rely on most. Longer bar = more important.")
    a, b = st.columns(2)
    for colw, fname, title in [(a, "reg_feature_importance.csv", "Regression model"),
                               (b, "clf_feature_importance.csv", "Classification model")]:
        p = RES / fname
        if p.exists():
            imp = pd.read_csv(p, index_col=0).head(15).sort_values("importance")
            colw.plotly_chart(px.bar(imp, x="importance", y=imp.index, orientation="h",
                                     title=title), use_container_width=True)
    st.caption("Feature importance counts how much a feature is used inside the model. "
               "It does not say whether a feature pushes demand up or down. SHAP does.")
    st.divider()

    # ---- 4. SHAP ----
    st.subheader("4. SHAP: how each feature changes the prediction")
    s1, s2 = st.columns(2)
    if (RES / "shap_summary.png").exists():
        s1.image(str(RES / "shap_summary.png"), caption="SHAP summary (each dot is one session)")
    if (RES / "shap_bar.png").exists():
        s2.image(str(RES / "shap_bar.png"), caption="Average impact of each feature")
    st.markdown(
        "**How to read it:**\n"
        "- **Left chart:** each dot is a session. Dots on the right push predicted "
        "demand up, dots on the left push it down. **Red** = high feature value, "
        "**blue** = low.\n"
        "- **Right chart:** the average size of each feature's impact. Longer = "
        "more influence overall."
    )
    st.divider()

    # ---- 5. LIME ----
    st.subheader("5. LIME: explaining one single session")
    show_lime("models")
    st.divider()

    # ---- 6. key findings ----
    st.subheader("6. Key findings")
    st.success(
        "**What drives charging demand**\n\n"
        "1. **Past behavior is the strongest signal:** the user's and the charger's "
        "average past demand matter most.\n"
        "2. **Hour of day matters next,** and **night** sessions push demand up.\n"
        "3. **Experience counts:** the number of past sessions of the user and "
        "charger has a moderate effect.\n"
        "4. **Location, company, charger type, month and weekday** have a small effect.\n"
        "5. Regression and classification agree on the same top factors, which makes "
        "the result more trustworthy."
    )
    st.warning(
        "**Limits to mention:** one UserID (likely a guest/default ID) holds a large "
        "share of sessions, so its history is a group average. R² is moderate because "
        "individual charging needs are not in the data. Data from the first and last "
        "partial days is edge noise."
    )