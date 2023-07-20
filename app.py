import streamlit as st
import requests
from model import getGeneratedCaption, getModelList

def main():
    st.set_page_config(layout="wide")  
    st.title("Image Captioning App")
    caption = None 

    with st.sidebar:
        modelList = ["All"] + getModelList()
        modelName = st.selectbox('Select Model', (modelList))
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        st.header("OR")
        txt = st.text_area('Enter Image URL', '')
        gc = st.button("Generate Caption")

    col1, col2 = st.columns([2,1], gap="large")

    with col1:        
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        elif txt.strip() != '':
            try:
                image = requests.get(txt)
                st.image(image.content, caption='Uploaded Image', use_column_width=True)
            except Exception as e:
                st.write("Error: Unable to fetch image from the provided URL." , e)
    with col2:                 
        if gc:            
            st.subheader("Generated Caption:")
            if modelName == "All":                
                for name in modelList[1:]:
                    st.caption(name)
                    if uploaded_file is not None:   
                        caption = getGeneratedCaption(uploaded_file, name,"FILE")

                    elif txt.strip() != '':
                        try:
                            image = requests.get(txt)
                            image_content = image.content
                            caption = getGeneratedCaption(image_content, name,"URL")
                        except Exception as e:
                            st.write("Error: Unable to fetch image from the provided URL." , e)
                    if caption is not None:
                        st.write(caption)
            else:
                if uploaded_file is not None:   
                    caption = getGeneratedCaption(uploaded_file, modelName,"FILE")

                elif txt.strip() != '':
                    try:
                        image = requests.get(txt)
                        image_content = image.content
                        caption = getGeneratedCaption(image_content, modelName,"URL")
                    except Exception as e:
                        st.write("Error: Unable to fetch image from the provided URL." , e)
        
                if caption is not None:
                    st.write(caption)

if __name__ == "__main__":
    main()
