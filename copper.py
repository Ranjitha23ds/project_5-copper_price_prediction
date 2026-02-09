import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import date
from streamlit_option_menu import option_menu

#load model and scaller object for  regression
with open("num_scaller_v2.pkl","rb")as f:
    scaler=pickle.load(f)
with open("width_scaler_v2.pkl","rb")as f:
    width_scaler=pickle.load(f)
with open("cat_encoding_v2.pkl","rb")as f:
    ohe_encoding=pickle.load(f)
with open("country_list_v2.pkl","rb")as f:
    country_list=pickle.load(f)
with open("application_list_v2.pkl","rb")as f:
    application_list=pickle.load(f)
with open("item_type_v2.pkl","rb")as f:
    item_list=pickle.load(f)
with open("product_ref_v2.pkl","rb")as f:
    product_ref_list=pickle.load(f)
with open("extratree_reg_model_v2.pkl","rb")as f:
    model_et=pickle.load(f)
with open("random_forest_clas_model_v2.pkl","rb")as f:
    rf_clf=pickle.load(f)

#from streamlit_option_menu import option_menu
st.title("Copper Price Prediction")
st.header("enter product features")
selected=option_menu("Main menu",["Price prediction","Status prediction"],menu_icon="menu_up",default_index=0,orientation="horizontal")
st.write("you selected",selected)


#get input from the user
quantity_tons=st.number_input("quantity_tons",min_value=0.0)
thickness=st.number_input("Thickness",min_value=0.0)
item_date=st.date_input("Item Date",value=date.today())
delivery_date=st.date_input("Delivery date",value=date.today())
width=st.number_input("width",min_value=0.0)
country=st.selectbox("select country",country_list)
application=st.selectbox("select application",application_list)
item_type=st.selectbox("select item_type",item_list)
product_ref=st.selectbox("select product_ref",product_ref_list)
delivery_days=(delivery_date-item_date).days

input_data=pd.DataFrame({"quantity_tons":[quantity_tons],
                        "thickness":[thickness],
                        "delivery_days":[delivery_days],
                        "width":[width],
                        "country":[country],
                        "application":[application],
                        "item_type":[item_type],
                        "product_ref":[product_ref]})
num_cols=["quantity_tons","thickness","delivery_days"]
num_log=np.log1p(input_data[num_cols])
num_scaled=scaler.transform(num_log)
width_scaled=width_scaler.transform(input_data[["width"]])
cat_cols=["country","application","item_type","product_ref"]
cat_encoded=ohe_encoding.transform(input_data[cat_cols])
x_input = np.hstack([num_scaled, width_scaled, cat_encoded])


#if delivery_days<=0:
  #  st.

if selected=="Price prediction":
    #st.subheader("price prediction page")
    if st.button("selling price prediction"):
        pred_price=model_et.predict(x_input)
        st.success(f"predict selling price{pred_price[0]:,.2f}")
elif selected=="Status prediction":
    #st.subheader("status prediction")
    if st.button(" status prediction"):
        pred_status=rf_clf.predict(x_input)[0]
        #st.success(f"Status prediction is {pred_status[0]}")
        if pred_status==1:
            st.success("WON")
        else:
            st.error("LOST")
            
