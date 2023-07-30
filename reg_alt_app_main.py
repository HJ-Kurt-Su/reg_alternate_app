import streamlit as st 

def main():
    st.header("Page Purpose & Description")
    st.markdown("**modulus cal**: Calculate modulus (slope) by linear regression")
    st.markdown("**1dof damping cal**: Calculate damping ratios by linear regression")
    # st.markdown("**regression**: Regression tool with linear & taguchi method")
    # st.markdown("**predict**: Predict result with load trained model")

if __name__ == '__main__':

    # st.title('Modulus (Slope) Tool')

    st.title("Author & License:")

    st.markdown("**Kurt Su** (phononobserver@gmail.com)")

    st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

    st.markdown("               ")
    st.markdown("               ")

    main()