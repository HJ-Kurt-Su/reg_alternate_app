import streamlit as st
import pandas as pd
# import itertools

# import datetime
import numpy as np
# import io
# import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.signal import argrelmax
# from statsmodels.graphics.gofplots import qqplot


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#


# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv(index=False).encode('utf-8')

# # @st.cache_data
# # def convert_df(df):
# #     # IMPORTANT: Cache the conversion to prevent computation on every rerun
# #     return df.to_csv().encode('utf-8')
  
# def ols_reg(formula, df):

#     model = smf.ols(formula, df)
#     res = model.fit()
#     df_result = df.copy()
#     df_result['yhat'] = res.fittedvalues
#     df_result['resid'] = res.resid

#     #   print(df_result.head())

#     return res, df_result, model


# def summary_model(result):
#     results_summary = result.summary()
#     results_as_html = results_summary.tables[1].as_html()
#     model = pd.read_html(results_as_html, header=0, index_col=0)[0]

#     model_as_html = results_summary.tables[0].as_html()
#     model2 = pd.read_html(model_as_html, header=0, index_col=0)[0]

#     r2 = model2.iloc[0,2]
#     modulus = model.iloc[1,0]
#     p_val = model.iloc[1,3]
#     intercept = model.iloc[0,0]

#     return r2, modulus, p_val, intercept



# def aquire_partial(df_result, criteria_column, judge_ratio):
#     df_reg_2nd = df_result.reset_index(drop=True)
#     err_max = df_reg_2nd[criteria_column].abs().max()
#     err_maxid = df_reg_2nd[criteria_column].abs().idxmax()
#     size = df_reg_2nd.shape[0]
#     # print(err_max)
#     # print(err_maxid)
#     low_row = int(judge_ratio * size)
#     up_row = int((1-judge_ratio) * size)

#     if err_maxid >= up_row:
#         cut_row = int((1-cut_ratio) * size)
#         df_reg_trial = df_reg_2nd.iloc[0: cut_row,:]
#         # print("Uprow")
#     elif err_maxid <= low_row:
#         cut_row = int(cut_ratio * size)
#         df_reg_trial = df_reg_2nd.iloc[cut_row :,:]
#         # print("Lowrow")
#     else:
#         print("-----------------------")
#         print("If Not Meet R^2 Criteria")
#         print("Please Adjust Initial Ratio")
#         print("-----------------------")
#     return df_reg_trial


# def figure_plot(df_stress, df_result):

#     fig_line = px.line(df_stress, x="Strain", y="Stress")
#     fig_line.update_traces(
#         line=dict(width=1, color="gray")
#     )
#     # print(df_result)
#     fig_reg_line = px.line(df_result, x="Strain", y="yhat")

#     fig_reg_line.update_traces(
#         line=dict(dash="dot", width=4, color="red")
#     )

#     return fig_line, fig_reg_line



## Program Start:

## Not change parameter
fig_size = [1280, 960]
# judge_ratio = 0.05

st.title('Damping Ratio Calculation Tool')


# Provide dataframe example & relative url
data_ex_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTM1DtHKHzS8v78tP4rMTxB8OL4vsjgdKMQ0GBxlAMPlh8Rp0jBn0qngW6Q5DWhlmIClav7DrP-_sNR/pub?output=csv"
# st.write("Factor Format Example File [link](%s)" % factor_ex_url)
st.markdown("#### Data Format Example File [Demo File](%s)" % data_ex_url)

uploaded_csv = st.file_uploader('#### 選擇您要上傳的CSV檔')

if uploaded_csv is not None:
    df_raw = pd.read_csv(uploaded_csv, encoding="utf-8")
    st.header('您所上傳的CSV檔內容：')
    st.dataframe(df_raw)

    select_list = list(df_raw.columns)
    # select_list
    time = st.selectbox("### Choose Time(s)", select_list)

    # response
    response_list = select_list.copy()
    response_list.remove(time)
    response = st.multiselect(
        "### Choose Response(y)", response_list)
    if not response:
        st.error("Please select at least one factor.")

    st.markdown("----------------")  

    ratio_col1, ratio_col2 = st.columns(2)
    with ratio_col1:
        st.markdown("#### **Choose local max/min criteria**")
        local_extrem_order = st.number_input('Local extream order', min_value=1, value=5, max_value=15, step=1)  
    with ratio_col2:
        st.markdown("#### **Choose min. filter response**")
        criteria_level = st.number_input('Min Criteria Leve', min_value=0.0, value=0.05, step=0.01) 

    st.markdown("----------------")  

    # set_col1, set_col2, set_col3 = st.columns(3)
    # with set_col1:
    #     st.markdown("##### Drop portion each reg.")
    #     cut_ratio = st.number_input('Cut Ratio', min_value=0.0, value=0.05, max_value=0.10, step=0.01)  
    # with set_col2:
    #     st.markdown("##### Max. Re-run times")
    #     re_run = st.number_input('Re-run', min_value=5, value=20, max_value=100, step=5) 
    # with set_col3:
    #     st.markdown("###### $R^2$ Criteria")
    #     r2_criteria = st.number_input('Min. Criteria:', min_value=0.985, value=0.990, max_value=0.999, step=0.001, format="%1f") 

    # st.markdown("----------------")  

    # dim_col1, dim_col2, dim_col3 = st.columns(3)
    # with dim_col1:
    #     st.markdown("##### Specimen Span(L)")
    #     L = st.number_input('mm', min_value=10.0, value=25.4)  
    # with dim_col2:
    #     st.markdown("##### Specimen Width(b)")
    #     b = st.number_input("mm", min_value=5.0, value=12.76) 
    # with dim_col3:
    #     st.markdown("###### Specimen Thickness(d)")
    #     d = st.number_input('mm', min_value=0.5, value=1.0) 
    
    if st.button('Perform Analysis'):
        # For single file design

        # df = df_raw.copy()[response]
        
        fig = make_subplots(
            rows=len(response), cols=1, shared_xaxes=True,
            subplot_titles=response
            )

        fig_reg = make_subplots(
            rows=len(response), cols=1, shared_xaxes=True,
            subplot_titles=response
            )
        
        j = 1
        for i in response:

            # Plot raw data
            fig.add_trace(go.Scatter(x=df_raw[time], y=df_raw[i], mode="lines"),
                        row=j, col=1)
            
            local_max_loc = argrelmax(np.array(df_raw[i]), order=local_extrem_order)
            y = df_raw[i].loc[local_max_loc]
            local_time = df_raw[time].loc[y.index]

            fig.add_trace(go.Scatter(x=local_time, y=y, mode="markers"),
            row=j, col=1
            )
            
            y_reg = np.log(y[y >= criteria_level])
            time_reg = df_raw[time].loc[y_reg.index]

            y_reg = y_reg.reset_index(drop=True)
            time_reg = time_reg.reset_index(drop=True)  

            time_reg = sm.add_constant(time_reg)
            model = sm.OLS(y_reg, time_reg)
            results = model.fit()
            fit_result = results.fittedvalues
            res_result = results.resid
            adj_rsqrt = results.rsquared_adj

            frequency = (len(time_reg[time])-1)/(time_reg[time].max()-time_reg[time].min())
            beta1 = results.params[1] 
            damping_ratio = -beta1/(2*frequency*np.pi)


            st.markdown("                ")
            # st.markdown("-----------------------")
            st.markdown("#### Channel: %s" % i)
            st.markdown("###### Frequency: %s" % round(frequency, 2))
            st.markdown("###### Damping Ratio: %s" % round(damping_ratio, 4))
            st.markdown("Adjust $R^2$ is: %s" % round(adj_rsqrt, 4))
            st.markdown("-----------------------")
            # print("                ")
            # print("------------------------------")
            # print("Channel is: ", i)
            # print("Frequency: ", round(frequency, 2))
            # print("Damping Ratio: ", round(damping_ratio, 4))
            # print("Adj R^2: ", round(adj_rsqrt, 4))

            # df_result.loc[j-1, "Frequency"] = frequency
            # df_result.loc[j-1, "Damping"] = damping_ratio
            # df_result.loc[j-1, "R2"] = adj_rsqrt


            # if calculate_settle_time == True:
            #     settle_time = np.log(settle_pct)/beta1
            #     print("Settling Time: ", round(settle_time, 2))
            #     df_result.loc[j-1, settle_name] = settle_time
            # print("------------------------------")
            # print("                ")

            # fig_reg = px.scatter(x=time_reg[x_variable], y=y_reg, 
            #           color_discrete_sequence=color_sequence,
            #           range_x=x_range, range_y=y_range, template=template,
            #           # log_x=xlog_scale, log_y=ylog_scale,
            #           width=fig_size[0], height=fig_size[1])
            
            fig_reg.add_trace(go.Scatter(x=time_reg[time], y=y_reg, mode="markers"),
                        row=j, col=1)
            fig_reg.add_trace(go.Scatter(x=time_reg[time], y=fit_result, mode="lines"),
                        row=j, col=1)

            j+=1
        
        fig
        fig_reg
   










        # df_stress = df_raw.copy()

        # L2 = L**2
        # bd2 = b*d**2 

        # df_stress["Strain"] = 6 * (df_stress[deform] * d) / L2   
        # df_stress["Stress"] = 1.5 * (df_stress[force] * 9.81 * L) / bd2  

        # stress_max = df_stress["Stress"].max()
        # stress_maxid = df_stress["Stress"].idxmax()

        # up_filter_stress = up_stress_ratio * stress_max
        # low_filter_stress = low_stress_ratio * stress_max     

        # df_modulus_reg = df_stress[(df_stress.index < stress_maxid) & 
        #                 (df_stress["Stress"] <= up_filter_stress) & 
        #                 (df_stress["Stress"] >= low_filter_stress)
        #                 ]
        
        # formula = "Stress ~ Strain"

        # result, df_result, model = ols_reg(formula, df_modulus_reg)

        # r2, modulus, p_val, intercept = summary_model(result)

        # for j in range(re_run):
        #     df_reg_trial = aquire_partial(df_result, "resid", judge_ratio)
        #     result, df_result, model = ols_reg(formula, df_reg_trial)
        #     r2, modulus, p_val, intercept = summary_model(result)
       
        #     if r2 >= r2_criteria:
        #        break

        # if r2 < r2_criteria:
        #     print("            ")
        #     print("-----------------------")
        #     print("Already Reach Re-run Limitation")
        #     print("Please Increase Re-run")

        # df_yield, yield_stress, yield_strain, intercept_x = find_yield(df_result, df_modulus_reg, offset_strain)

        # st.markdown("                ")
        # st.markdown("-----------------------")
        # st.markdown("#### Intercept: %s" % intercept)
        # st.markdown("#### Modulus: %s" % modulus)
        # st.markdown("P Value: %s" % p_val)
        # st.markdown("Adjust $R^2$ is: %s" % r2)
        # st.markdown("-----------------------")

        # fig_line, fig_reg_line = figure_plot(df_stress, df_result)
        # fig_interact = go.Figure(data=fig_line.data + 
        #           fig_reg_line.data 
        #           )
        # fig_interact.update_layout(
        #         autosize=False,
        #         width=fig_size[0],
        #         height=fig_size[1],
        #         xaxis_title="Strain",
        #         yaxis_title="Stress(MPa)",
        #         title="Strian vs. Stress")
        # st.plotly_chart(fig_interact, use_container_width=True)
