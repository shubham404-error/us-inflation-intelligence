import os
import streamlit as st
from dotenv import load_dotenv

from data import fetch_model_data

load_dotenv()

st.set_page_config(
    page_title="US Inflation Intelligence",
    layout="wide",
)

st.title("US Inflation Intelligence")
st.caption("Data pipeline test")

if not os.getenv("FRED_API_KEY"):
    st.error("FRED_API_KEY is missing from Streamlit Secrets.")
    st.stop()

try:
    data = fetch_model_data()

    st.success("FRED data loaded successfully.")

    st.write("Columns returned by data.py:")
    st.write(list(data.columns))

    st.write("Shape:")
    st.write(data.shape)

    st.write("Latest observations:")
    st.dataframe(
        data.tail(10),
        use_container_width=True,
    )

except Exception as e:
    st.exception(e)
