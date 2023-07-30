import streamlit as st
import pandas as pd
import itertools

import datetime
import numpy as np
import io
import statsmodels.formula.api as smf
from scipy.stats import shapiro
# from statsmodels.graphics.gofplots import qqplot


import plotly.express as px
# from plotly.subplots import make_subplots
import plotly.graph_objects as go
#


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')

def ols_reg(formula, df):

    model = smf.ols(formula, df)
    res = model.fit()
    df_result = df.copy()
    df_result['yhat'] = res.fittedvalues
    df_result['resid'] = res.resid

    #   print(df_result.head())

    return res, df_result, model


def summary_model(result):
    results_summary = result.summary()
    results_as_html = results_summary.tables[1].as_html()
    model = pd.read_html(results_as_html, header=0, index_col=0)[0]

    model_as_html = results_summary.tables[0].as_html()
    model2 = pd.read_html(model_as_html, header=0, index_col=0)[0]

    r2 = model2.iloc[0,2]
    modulus = model.iloc[1,0]
    p_val = model.iloc[1,3]
    intercept = model.iloc[0,0]

    return r2, modulus, p_val, intercept



def aquire_partial(df_result, criteria_column, judge_ratio):
    df_reg_2nd = df_result.reset_index(drop=True)
    err_max = df_reg_2nd[criteria_column].abs().max()
    err_maxid = df_reg_2nd[criteria_column].abs().idxmax()
    size = df_reg_2nd.shape[0]
    # print(err_max)
    # print(err_maxid)
    low_row = int(judge_ratio * size)
    up_row = int((1-judge_ratio) * size)

    if err_maxid >= up_row:
        cut_row = int((1-cut_ratio) * size)
        df_reg_trial = df_reg_2nd.iloc[0: cut_row,:]
        # print("Uprow")
    elif err_maxid <= low_row:
        cut_row = int(cut_ratio * size)
        df_reg_trial = df_reg_2nd.iloc[cut_row :,:]
        # print("Lowrow")
    else:
        st.markdown("-----------------------")
        st.markdown("Not Meet R^2 Criteria")
        st.markdown("Please Adjust Initial Ratio")
        st.markdown("-----------------------")
        df_reg_trial=pd.DataFrame()
    return df_reg_trial


def figure_plot(df_stress, df_result):

    fig_line = px.line(df_stress, x="Strain", y="Stress")
    fig_line.update_traces(
        line=dict(width=1, color="gray")
    )
    # print(df_result)
    fig_reg_line = px.line(df_result, x="Strain", y="yhat")

    fig_reg_line.update_traces(
        line=dict(dash="dot", width=4, color="red")
    )

    return fig_line, fig_reg_line



## Program Start:

## Not change parameter
# fig_size = [1280, 960]
judge_ratio = 0.05

st.title('Modulus (Slope) Tool')

# st.markdown("#### Author & License:")

# st.markdown("**Kurt Su** (phononobserver@gmail.com)")

# st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

# st.markdown("               ")
# st.markdown("               ")


# Provide dataframe example & relative url
data_ex_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQzc-xaGomUO81MiJ7lyQ__FhbPIGK4YTvUjoE76BglXWj2XLIqSc-9-Mrlq9P2iuYeqRhgRJTgn1QW/pub?gid=0&single=true&output=csv"
# st.write("Factor Format Example File [link](%s)" % factor_ex_url)
st.markdown("### **Data Format Example File [Demo File](%s)**" % data_ex_url)

uploaded_csv = st.file_uploader('#### 選擇您要上傳的CSV檔')

if uploaded_csv is not None:
    df_raw = pd.read_csv(uploaded_csv, encoding="utf-8")
    st.header('您所上傳的CSV檔內容：')
    st.dataframe(df_raw)

    select_list = list(df_raw.columns)
    # select_list
    force = st.selectbox("### Choose Force(y)", select_list)
    f_unit = st.selectbox("### Choose Force Unit", ["kgf", "kN", "N"])

    # response
    deform_list = select_list.copy()
    deform_list.remove(force)
    deform = st.selectbox(
        "### Choose Deformation(x), (Unit-mm)", deform_list)
    if not deform:
        st.error("Please select at least one factor.")

    st.markdown("----------------")  

ratio_col1, ratio_col2 = st.columns(2)
with ratio_col1:
    st.markdown("#### **Choose lower ratio of max. stress**")
    low_stress_ratio = st.number_input('Low Stress Ratio', min_value=0.0, value=0.05, max_value=0.15, step=0.01)  
with ratio_col2:
    st.markdown("#### **Choose upper ratio of max. stress**")
    up_stress_ratio = st.number_input('Up Stress Ratio', min_value=0.4, value=0.9, max_value=0.96, step=0.05) 

st.markdown("----------------")  

set_col1, set_col2, set_col3 = st.columns(3)
with set_col1:
    st.markdown("##### Drop portion each reg.")
    cut_ratio = st.number_input('Cut Ratio', min_value=0.0, value=0.05, max_value=0.10, step=0.01)  
with set_col2:
    st.markdown("##### Max. Re-run times")
    re_run = st.number_input('Re-run', min_value=5, value=20, max_value=100, step=5) 
with set_col3:
    st.markdown("###### $R^2$ Criteria")
    r2_criteria = st.number_input('Min. Criteria:', min_value=0.95, value=0.990, max_value=0.999, step=0.001, format="%1f") 

st.markdown("----------------")  

# select_list
test_type = st.selectbox("### Choose Test Type", ["3PT Bending", "Tensile", "Slope Only"])

dim_col1, dim_col2, dim_col3 = st.columns(3)
if test_type == "3PT Bending":
    with dim_col1:
        st.markdown("##### Specimen Span(L)")
        L = st.number_input('mm', min_value=10.0, value=25.4)  
    with dim_col2:
        st.markdown("##### Specimen Width(b)")
        b = st.number_input("mm", min_value=5.0, value=12.76) 
    with dim_col3:
        st.markdown("##### Specimen Thickness(d)")
        d = st.number_input('mm', min_value=0.5, value=1.0) 
elif test_type == "Tensile":
    with dim_col1:
        st.markdown("##### Specimen Length(L)")
        L = st.number_input('mm', min_value=10.0, value=50.0)  
    with dim_col2:
        st.markdown("##### Specimen Width(b)")
        b = st.number_input("mm", min_value=5.0, value=25.4) 
    with dim_col3:
        st.markdown("##### Specimen Thickness(t)")
        t = st.number_input('mm', min_value=0.5, value=1.5) 

size_col1, size_col2 = st.columns(2)
with size_col1:
    fig_width = st.number_input('Figure Width', min_value=640, value=1280, max_value=5120, step=320) 
    
with size_col2:
    fig_height = st.number_input('Figure Height', min_value=480, value=960, max_value=3840, step=240) 

if st.checkbox('Perform Analysis'):
# For single file design
    df_stress = df_raw.copy()

    
    if f_unit == "kN":
        df_stress["Force_N"] = df_stress[force] * 1000
    elif f_unit == "kgf":
        df_stress["Force_N"] = df_stress[force] * 9.81
    elif f_unit == "N":
        df_stress["Force_N"] = df_stress[force]

    if test_type == "3PT Bending":

        L2 = L**2
        bd2 = b*d**2 

        df_stress["Strain"] = 6 * (df_stress[deform] * d) / L2   
        df_stress["Stress"] = 1.5 * (df_stress["Force_N"] * L) / bd2  
    
    elif test_type == "Tensile":

        area = b*t
        df_stress["Strain"] = df_stress[deform] / L   
        df_stress["Stress"] = df_stress["Force_N"] / area

    elif test_type == "Slope Only":

        df_stress["Strain"] = df_stress[deform]   
        df_stress["Stress"] = df_stress["Force_N"]


    stress_max = df_stress["Stress"].max()
    stress_maxid = df_stress["Stress"].idxmax()

    up_filter_stress = up_stress_ratio * stress_max
    low_filter_stress = low_stress_ratio * stress_max     

    df_modulus_reg = df_stress[(df_stress.index < stress_maxid) & 
                    (df_stress["Stress"] <= up_filter_stress) & 
                    (df_stress["Stress"] >= low_filter_stress)
                    ]
    
    formula = "Stress ~ Strain"

    result, df_result, model = ols_reg(formula, df_modulus_reg)

    r2, modulus, p_val, intercept = summary_model(result)
    # df_result

    for j in range(re_run):
        df_reg_trial = aquire_partial(df_result, "resid", judge_ratio)
        if df_reg_trial.shape[0] == 0:
            break

        else:
            result, df_result, model = ols_reg(formula, df_reg_trial)
            r2, modulus, p_val, intercept = summary_model(result)
    
        if r2 >= r2_criteria:
            break

    if r2 < r2_criteria:
        st.markdown("            ")
        st.markdown("-----------------------")
        st.markdown("Already Reach Re-run Limitation")
        st.markdown("Please Increase Re-run")

    # df_yield, yield_stress, yield_strain, intercept_x = find_yield(df_result, df_modulus_reg, offset_strain)

    st.markdown("                ")
    st.markdown("-----------------------")
    st.markdown("#### Intercept: %s" % intercept)
    st.markdown("#### Modulus: %s" % modulus)
    st.markdown("P Value: %s" % p_val)
    st.markdown("Adjust $R^2$ is: %s" % r2)
    st.markdown("-----------------------")

    fig_line, fig_reg_line = figure_plot(df_stress, df_result)
    fig_interact = go.Figure(data=fig_line.data + 
                fig_reg_line.data 
                )
    if test_type == "Slope Only":
            
            fig_interact.update_layout(
                autosize=False,
                width=fig_width,
                height=fig_height,
                xaxis_title=deform,
                yaxis_title=force,
                title= force + " vs. " + deform)
    else:

        fig_interact.update_layout(
                autosize=False,
                width=fig_width,
                height=fig_height,
                xaxis_title="Strain",
                yaxis_title="Stress(MPa)",
                title="Strian vs. Stress")
    st.plotly_chart(fig_interact, use_container_width=True)

    date = str(datetime.datetime.now()).split(" ")[0]
    mybuff = io.StringIO()
    fig_file_name = date + "_" + test_type + ".html"
    # fig_html = fig_pair.write_html(fig_file_name)
    fig_interact.write_html(mybuff, include_plotlyjs='cdn')
    html_bytes = mybuff.getvalue().encode()

    st.download_button(label="Download figure",
                        data=html_bytes,
                        file_name=fig_file_name,
                        mime='text/html'
                        )