# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="AI Club -- ISIC Challenge",
    )
    
    st.title("Welcome to our Skin Lesion Detection Website!")


    st.markdown(
        """
        This tool is designed to assist in the detection and analysis of skin lesions using advanced machine learning techniques.
        """
    )

    st.header("How Does This Tool Work?")
    st.write(
        """
        - On the top left choose the model you would want to run
        - Upload a image of the skin lesion.
        - Our AI model analyzes the image and provides an assessment.
        """
    )

    st.header("About the ISIC challenge")
    st.write(
        """
        This project is inspired by the International Skin Imaging Collaboration (ISIC) Challenge, 
        which is an annual competition focused on the automatic analysis of skin lesions using dermoscopic images. 
        This challenge aims to improve and evaluate computer algorithms for the diagnosis of melanoma, 
        a lethal form of skin cancer, as well as other skin disorders.
        """
    )
    
    
    st.header("Credits")
    st.markdown(        
        """
        Made possible by
        - Luan Nguyen
        - Joseph Giordano
        - Kyle Phillips
        - Aron Chen
        - Alan Yin
        """
    )

    st.write(
        """
        DISCLAIMER: This tool is not a substitute for professional medical advice, diagnosis, or treatment.
        """
    )


if __name__ == "__main__":
    run()