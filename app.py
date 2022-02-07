import pickle 
import streamlit as st

model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizr.pkl","rb"))

def main():
    st.title("Detektor Spam Chat")
    st.subheader("Dibangun dengan Python dan Streamlit")
    msg=st.text_area("Masukan pesan teks: ")
    if st.button("Prediksi"):
        data=[msg]
        data=cv.transform(data)
        prediksi = model.predict(data)
        result=prediksi[0]
        if result==1:
            st.error("Ini adalah pesan spam/penipuan")
        elif result==2:
            st.warning("Ini adalah pesan promo")
        else:
            st.success("Ini adalah pesan normal")
main()