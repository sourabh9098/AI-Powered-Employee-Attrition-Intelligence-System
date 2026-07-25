# conda = employee_attrition 


import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# ── Page Config 
st.set_page_config(
    page_title = "HR Attrition Analytics",
    page_icon  = "👥",
    layout     = "wide",
    initial_sidebar_state = "collapsed"
)


# ── Load Model & Preprocessors 
@st.cache_resource
def load_artifacts():
    model    = keras.models.load_model('employee_attrition_ann.keras')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('selected_features.pkl')
    return model, scaler, features

model, scaler, feature_cols = load_artifacts()

# ── Preprocessing Function 
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Label Encoding
    df['Gender']  = df['Gender'].map({'Male': 1, 'Female': 0})
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

    # One Hot Encoding
    ohe_cols = ['BusinessTravel', 'Department',
                'EducationField', 'JobRole', 'MaritalStatus']
    df = pd.get_dummies(df, columns=ohe_cols)

    # Align with training features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    # Scale
    df_scaled = scaler.transform(df)
    return df_scaled

# ── Predict Function 
def predict_attrition(data):
    processed = preprocess_input(data)
    prob      = model.predict(processed, verbose=0)[0][0]
    label     = 'YES' if prob >= 0.5 else 'NO'
    risk_score = int(prob * 100)
    return label, float(prob), risk_score

# ── CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #050D18;
    color: #E2E8F0;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Header */
.hero {
    padding: 36px 0 24px 0;
    text-align: center;
}

.hero-badge {
    display: inline-block;
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6366F1;
    margin-bottom: 16px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 48px;
    font-weight: 700;
    letter-spacing: -2px;
    color: #F8FAFC;
    margin: 0 0 8px 0;
}

.hero-sub {
    font-size: 15px;
    color: #64748B;
    margin: 0;
}

/* Divider */
.divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg,
    transparent, #1E293B, transparent);
    margin: 28px 0;
}

/* Metric Cards */
.metric-card {
    background: #0D1424;
    border: 1px solid #1E293B;
    border-top: 3px solid #6366F1;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #6366F1;
    margin: 0;
}

.metric-lbl {
    font-size: 11px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 4px 0 0 0;
}

/* Section Header */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #475569;
    border-left: 3px solid #6366F1;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* Input Card */
.input-card {
    background: #0D1424;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Result Cards */
.result-yes {
    background: linear-gradient(135deg, #2D0A0A, #450A0A);
    border: 2px solid #EF4444;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-no {
    background: linear-gradient(135deg, #052E16, #064E3B);
    border: 2px solid #10B981;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-icon { font-size: 52px; display: block; margin-bottom: 8px; }
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 4px;
    margin: 0;
}

.result-sub {
    font-size: 13px;
    color: #94A3B8;
    margin-top: 8px;
    letter-spacing: 1px;
}

/* Risk Meter */
.risk-card {
    background: #0D1424;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 20px 24px;
    margin-top: 16px;
}

.risk-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 12px;
}

.risk-bar-track {
    background: #1E293B;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 8px 0;
}

/* Info Box */
.info-box {
    background: #0D1424;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 18px 20px;
    margin-top: 16px;
}

.info-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 10px;
}

.info-text {
    font-size: 14px;
    color: #94A3B8;
    line-height: 1.7;
    margin: 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1424;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1E293B;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #475569;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.stTabs [aria-selected="true"] {
    background: #6366F1 !important;
    color: white !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.35) !important;
}

/* Selectbox & Slider */
.stSelectbox > div > div {
    background: #0D1424 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}

label {
    color: #64748B !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

.stSlider > div > div > div {
    background: #6366F1 !important;
}

/* Overview cards */
.overview-card {
    background: #0D1424;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 4px 0;
}

.overview-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
}

.overview-lbl {
    font-size: 11px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 4px 0 0 0;
}

.stSpinner > div {
    border-top-color: #6366F1 !important;
}
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown("""
<div class='hero'>
    <div class='hero-badge'>ANN · HR Analytics · IBM Dataset</div>
    <h1 class='hero-title'>Employee Attrition Predictor</h1>
    <p class='hero-sub'>
    Predict which employees are at risk of leaving
    using a deep learning model trained on 1,470 IBM HR records.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stats Bar 
s1, s2, s3, s4 = st.columns(4)
for col, val, lbl in zip(
    [s1, s2, s3, s4],
    ["1,470", "85.7%", "25", "ANN"],
    ["Employees Trained", "Model Accuracy",
     "Features Used", "Architecture"]
):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-val'>{val}</p>
            <p class='metric-lbl'>{lbl}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# TABS

tab1, tab2, tab3 = st.tabs([
    "🔮  Predict Employee",
    "📊  HR Overview",
    "ℹ️  Model Info"
])


# TAB 1 — Predict Single Employee

with tab1:
    st.markdown("<div class='section-title'>Employee Details</div>",
                unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Personal Info**")
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status",
            ["Single", "Married", "Divorced"])
        distance = st.slider("Distance From Home (km)", 1, 30, 5)

    with c2:
        st.markdown("**Job Details**")
        department = st.selectbox("Department",
            ["Sales", "Research & Development",
             "Human Resources"])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist",
            "Laboratory Technician", "Manufacturing Director",
            "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director",
            "Human Resources"
        ])
        job_level     = st.slider("Job Level", 1, 5, 2)
        job_sat       = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        env_sat       = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        overtime      = st.selectbox("OverTime", ["Yes", "No"])

    with c3:
        st.markdown("**Experience & Compensation**")
        monthly_income    = st.slider("Monthly Income ($)",
                                      1000, 20000, 5000, step=500)
        years_at_company  = st.slider("Years at Company", 0, 40, 5)
        years_in_role     = st.slider("Years in Current Role", 0, 18, 3)
        years_promotion   = st.slider("Years Since Last Promotion",
                                      0, 15, 2)
        work_life         = st.slider("Work Life Balance (1-4)", 1, 4, 3)
        num_companies     = st.slider("Num Companies Worked", 0, 9, 2)
        business_travel   = st.selectbox("Business Travel", [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ])

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Additional Fields ─────────────────────────────
    with st.expander("More Details (Optional)"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            education      = st.slider("Education (1-5)", 1, 5, 3)
            edu_field      = st.selectbox("Education Field", [
                "Life Sciences", "Medical", "Marketing",
                "Technical Degree", "Human Resources", "Other"
            ])
        with col_b:
            perf_rating    = st.slider("Performance Rating (1-4)", 1, 4, 3)
            rel_sat        = st.slider("Relationship Satisfaction (1-4)",
                                       1, 4, 3)
            job_involve    = st.slider("Job Involvement (1-4)", 1, 4, 3)
        with col_c:
            stock_option   = st.slider("Stock Option Level (0-3)", 0, 3, 1)
            training_times = st.slider("Training Times Last Year", 0, 6, 3)
            total_working  = st.slider("Total Working Years", 0, 40, 8)
            years_manager  = st.slider("Years With Current Manager",
                                       0, 17, 4)
            daily_rate     = st.slider("Daily Rate", 100, 1500, 800)
            hourly_rate    = st.slider("Hourly Rate", 30, 100, 65)
            monthly_rate   = st.slider("Monthly Rate",
                                       2000, 27000, 14000, step=500)
            salary_hike    = st.slider("Percent Salary Hike", 11, 25, 15)

    st.markdown("")
    predict_btn = st.button("🔮  Predict Attrition Risk", type="primary")

    # ── Prediction Result ─────────────────────────────
    if predict_btn:
        employee_data = {
            'Age'                      : age,
            'BusinessTravel'           : business_travel,
            'DailyRate'                : daily_rate,
            'Department'               : department,
            'DistanceFromHome'         : distance,
            'Education'                : education,
            'EducationField'           : edu_field,
            'EnvironmentSatisfaction'  : env_sat,
            'Gender'                   : gender,
            'HourlyRate'               : hourly_rate,
            'JobInvolvement'           : job_involve,
            'JobLevel'                 : job_level,
            'JobRole'                  : job_role,
            'JobSatisfaction'          : job_sat,
            'MaritalStatus'            : marital,
            'MonthlyIncome'            : monthly_income,
            'MonthlyRate'              : monthly_rate,
            'NumCompaniesWorked'       : num_companies,
            'OverTime'                 : overtime,
            'PercentSalaryHike'        : salary_hike,
            'PerformanceRating'        : perf_rating,
            'RelationshipSatisfaction' : rel_sat,
            'StockOptionLevel'         : stock_option,
            'TotalWorkingYears'        : total_working,
            'TrainingTimesLastYear'    : training_times,
            'WorkLifeBalance'          : work_life,
            'YearsAtCompany'           : years_at_company,
            'YearsInCurrentRole'       : years_in_role,
            'YearsSinceLastPromotion'  : years_promotion,
            'YearsWithCurrManager'     : years_manager,
        }

        with st.spinner("Analyzing employee profile..."):
            label, prob, risk_score = predict_attrition(employee_data)

        st.markdown("<div class='divider'></div>",
                    unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1])

        with r1:
            if label == 'YES':
                st.markdown(f"""
                <div class='result-yes'>
                    <span class='result-icon'>⚠️</span>
                    <p class='result-label'
                    style='color:#EF4444;'>LEAVING</p>
                    <p class='result-sub'>
                    This employee is at risk of attrition
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-no'>
                    <span class='result-icon'>✅</span>
                    <p class='result-label'
                    style='color:#10B981;'>STAYING</p>
                    <p class='result-sub'>
                    This employee is likely to stay
                    </p>
                </div>""", unsafe_allow_html=True)

        with r2:
            risk_color = '#EF4444' if label=='YES' else '#10B981'
            stay_pct   = round((1-prob)*100, 1)
            leave_pct  = round(prob*100, 1)

            st.markdown(f"""
            <div class='risk-card'>
                <div class='risk-title'>Risk Score</div>
                <div style='font-family:Space Grotesk,sans-serif;
                font-size:52px; font-weight:800;
                color:{risk_color}; line-height:1;'>
                {risk_score}<span style='font-size:24px;'>/100</span>
                </div>
                <div class='risk-bar-track' style='margin-top:12px;'>
                    <div style='height:100%;
                    width:{risk_score}%;
                    background:{risk_color};
                    border-radius:999px;'></div>
                </div>
                <div style='display:flex;
                justify-content:space-between;
                margin-top:16px;'>
                    <div style='text-align:center;'>
                        <div style='color:#10B981;
                        font-size:22px; font-weight:700;'>
                        {stay_pct}%</div>
                        <div style='color:#475569;
                        font-size:11px; letter-spacing:1px;'>
                        STAY</div>
                    </div>
                    <div style='text-align:center;'>
                        <div style='color:#EF4444;
                        font-size:22px; font-weight:700;'>
                        {leave_pct}%</div>
                        <div style='color:#475569;
                        font-size:11px; letter-spacing:1px;'>
                        LEAVE</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Recommendation
            if label == 'YES':
                tips = []
                if overtime == 'Yes':
                    tips.append("Reduce overtime workload")
                if job_sat <= 2:
                    tips.append("Improve job satisfaction")
                if years_promotion >= 3:
                    tips.append("Consider promotion or salary hike")
                if work_life <= 2:
                    tips.append("Improve work-life balance")
                if monthly_income < 5000:
                    tips.append("Review compensation package")
                if not tips:
                    tips.append("Conduct retention interview")

                tips_html = ''.join(
                    [f"→ {t}<br>" for t in tips[:3]]
                )
                st.markdown(f"""
                <div class='info-box' style='border-color:#EF444433;
                margin-top:12px;'>
                    <div class='info-title'
                    style='color:#EF4444;'>
                    HR Action Required</div>
                    <p class='info-text'>{tips_html}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='info-box' style='border-color:#10B98133;
                margin-top:12px;'>
                    <div class='info-title'
                    style='color:#10B981;'>
                    Employee Status</div>
                    <p class='info-text'>
                    → Employee profile looks stable<br>
                    → Continue regular check-ins<br>
                    → Maintain current engagement level
                    </p>
                </div>""", unsafe_allow_html=True)



# TAB 2 — HR Overview

with tab2:
    try:
        df = pd.read_csv(
            'WA_Fn-UseC_-HR-Employee-Attrition.csv'
        )

        st.markdown(
            "<div class='section-title'>Company Overview</div>",
            unsafe_allow_html=True
        )

        total     = len(df)
        left      = (df['Attrition']=='Yes').sum()
        stayed    = (df['Attrition']=='No').sum()
        attr_rate = round(left/total*100, 1)

        o1, o2, o3, o4 = st.columns(4)
        for col, val, lbl, color in zip(
            [o1, o2, o3, o4],
            [total, stayed, left, f"{attr_rate}%"],
            ["Total Employees", "Active Employees",
             "Employees Left", "Attrition Rate"],
            ["#6366F1", "#10B981", "#EF4444", "#F59E0B"]
        ):
            with col:
                st.markdown(f"""
                <div class='overview-card'>
                    <p class='overview-val'
                    style='color:{color};'>{val}</p>
                    <p class='overview-lbl'>{lbl}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>",
                    unsafe_allow_html=True)

        import plotly.graph_objects as go
        import plotly.express as px

        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown(
                "<div class='section-title'>"
                "Attrition by Department</div>",
                unsafe_allow_html=True
            )
            dept = df.groupby('Department')['Attrition'].apply(
                lambda x: round((x=='Yes').sum()/len(x)*100, 1)
            ).reset_index()
            dept.columns = ['Department', 'Attrition Rate']

            fig1 = px.bar(
                dept, x='Department', y='Attrition Rate',
                color='Attrition Rate',
                color_continuous_scale=['#10B981','#F59E0B','#EF4444'],
                text='Attrition Rate'
            )
            fig1.update_traces(texttemplate='%{text}%',
                               textposition='outside')
            fig1.update_layout(
                plot_bgcolor ="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color   ="#E2E8F0",
                height       = 300,
                showlegend   = False,
                coloraxis_showscale=False,
                margin=dict(l=0,r=0,t=20,b=0)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with ch2:
            st.markdown(
                "<div class='section-title'>"
                "Overtime vs Attrition</div>",
                unsafe_allow_html=True
            )
            ot = df.groupby(['OverTime','Attrition']).size()\
                   .reset_index(name='Count')
            fig2 = px.bar(
                ot, x='OverTime', y='Count',
                color='Attrition',
                color_discrete_map={
                    'Yes':'#EF4444','No':'#10B981'
                },
                barmode='group'
            )
            fig2.update_layout(
                plot_bgcolor ="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color   ="#E2E8F0",
                height       = 300,
                margin=dict(l=0,r=0,t=20,b=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

        ch3, ch4 = st.columns(2)

        with ch3:
            st.markdown(
                "<div class='section-title'>"
                "Age Distribution</div>",
                unsafe_allow_html=True
            )
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=df[df['Attrition']=='Yes']['Age'],
                name='Left', marker_color='#EF4444',
                opacity=0.75, nbinsx=20
            ))
            fig3.add_trace(go.Histogram(
                x=df[df['Attrition']=='No']['Age'],
                name='Stayed', marker_color='#10B981',
                opacity=0.75, nbinsx=20
            ))
            fig3.update_layout(
                barmode='overlay',
                plot_bgcolor ="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color   ="#E2E8F0",
                height       = 300,
                margin=dict(l=0,r=0,t=20,b=0)
            )
            st.plotly_chart(fig3, use_container_width=True)

        with ch4:
            st.markdown(
                "<div class='section-title'>"
                "Income vs Attrition</div>",
                unsafe_allow_html=True
            )
            fig4 = px.box(
                df, x='Attrition', y='MonthlyIncome',
                color='Attrition',
                color_discrete_map={
                    'Yes':'#EF4444','No':'#10B981'
                }
            )
            fig4.update_layout(
                plot_bgcolor ="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color   ="#E2E8F0",
                height       = 300,
                showlegend   = False,
                margin=dict(l=0,r=0,t=20,b=0)
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Top Attrition Job Roles
        st.markdown(
            "<div class='section-title'>"
            "Attrition Rate by Job Role</div>",
            unsafe_allow_html=True
        )
        role_attr = df.groupby('JobRole')['Attrition'].apply(
            lambda x: round((x=='Yes').sum()/len(x)*100, 1)
        ).sort_values(ascending=True).reset_index()
        role_attr.columns = ['JobRole','Attrition Rate']

        fig5 = px.bar(
            role_attr, x='Attrition Rate', y='JobRole',
            orientation='h',
            color='Attrition Rate',
            color_continuous_scale=['#10B981','#F59E0B','#EF4444'],
            text='Attrition Rate'
        )
        fig5.update_traces(texttemplate='%{text}%',
                           textposition='outside')
        fig5.update_layout(
            plot_bgcolor ="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color   ="#E2E8F0",
            height       = 360,
            showlegend   = False,
            coloraxis_showscale=False,
            margin=dict(l=0,r=20,t=20,b=0)
        )
        st.plotly_chart(fig5, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.info("Place WA_Fn-UseC_-HR-Employee-Attrition.csv "
                "in the same folder as app.py")



# TAB 3 — Model Info

with tab3:
    st.markdown(
        "<div class='section-title'>Model Architecture</div>",
        unsafe_allow_html=True
    )

    mi1, mi2 = st.columns(2)

    with mi1:
        st.markdown("""
        <div class='info-box'>
            <div class='info-title'>ANN Architecture</div>
            <p class='info-text'>
            Input Layer  → 25 features<br>
            Dense(256)   → ReLU + BatchNorm + Dropout(0.3)<br>
            Dense(128)   → ReLU + BatchNorm + Dropout(0.3)<br>
            Dense(64)    → ReLU + Dropout(0.2)<br>
            Dense(1)     → Sigmoid output
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box'>
            <div class='info-title'>Training Config</div>
            <p class='info-text'>
            Optimizer    → Adam<br>
            Loss         → Binary Crossentropy<br>
            Epochs       → 100 (Early Stopping)<br>
            Batch Size   → 32<br>
            Class Weight → Balanced (handles imbalance)
            </p>
        </div>""", unsafe_allow_html=True)

    with mi2:
        st.markdown("""
        <div class='info-box'>
            <div class='info-title'>Model Performance</div>
            <p class='info-text'>
            Accuracy     → 85.7%<br>
            Precision    → 55.6%<br>
            Recall       → 53.2%<br>
            F1 Score     → 0.54<br><br>
            ANN outperforms Random Forest & XGBoost
            on Recall — catching more employees at
            risk of leaving.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box'>
            <div class='info-title'>Preprocessing</div>
            <p class='info-text'>
            Label Encoding  → Gender, OverTime<br>
            One-Hot Encoding → BusinessTravel,
            Department, EducationField,
            JobRole, MaritalStatus<br>
            Scaling → StandardScaler<br>
            Imbalance → Class Weight Balancing
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='margin-top:16px;'>
        <div class='info-title'>Disclaimer</div>
        <p class='info-text'>
        This tool is for educational purposes only.
        Built on IBM HR Analytics dataset (1,470 records, 2017).
        Predictions should not be used as the sole basis
        for HR decisions. Always combine with human judgment
        and direct employee conversations.
        </p>
    </div>""", unsafe_allow_html=True)

# ── Footer 
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:12px 0 20px 0;'>
    <p style='color:#1E293B; font-size:12px;
    letter-spacing:2px; text-transform:uppercase; margin:0;'>
    HR Attrition Predictor · ANN · Keras ·
    Streamlit · Built by Sourabh
    </p>
</div>""", unsafe_allow_html=True)