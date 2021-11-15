# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:46:05 2021

@author: Pascal
"""

import streamlit as st 
from PIL import Image 
from datetime import date 
import time 
import cv2 
import os
import object_detection as detect
import snapshot as snap


def write_image(out_image):
    today = date.today()
    d = today.strftime("%b-%d-%Y")
    t = time.localtime()
    current_time = time.strftime("%H-%M-%S", t)
    
    file_name = "tempDir/photo_" + d + "_" + current_time + ".jpg"
    cv2.imwrite(file_name, out_image)
    
    return(file_name)



def main():
    
    # ===================== Set page config and background =======================
    st.set_page_config(page_title ="Objekterkennung", 
                       page_icon=':camera:', 
                       layout='centered')
    

    # ===================== Set header and site info =============================
    text = """
    <center> <br>Testumgebung für Objektklassifizierungen mit Webcam. </br> </center>
    </center>
    """
    html_temp = f"""
    <p style = "color:#1F4E79; text_align:justify;"> {text} </p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    
    # ======================= Get tf lite model details ==========================
    labels, colors, height, width, interpreter = detect.define_tf_lite_model()
    
    # ============================= Main app =====================================
    option = st.selectbox('Wähle Eingabeart aus',
                         ('Nichts', 
                          'Webcam', 
                        #  'Upload photo'
                          ))
    
    if option == 'Webcam':
        
        out_image = snap.streamlit_webrtc_snapshot()
        
        if out_image is not None:
            st.header("Dein Bild")
            st.image(out_image, channels="BGR")
            # Speichere Bild temporär
            file_name = write_image(out_image)
            object_detection = detect.display_results(labels, 
                                                      colors, 
                                                      height, 
                                                      width,
                                                      file_name, 
                                                      interpreter, 
                                                      threshold=0.5)
            st.image(Image.fromarray(object_detection), use_column_width=True)
            
   #     else:
      #      st.warning('Kalibriere...')
           

if __name__ == "__main__":
    main()
    