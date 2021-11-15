# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:46:05 2021

@author: Pascal
"""

import streamlit as st 
from PIL import Image # PIL is used to display images 
from datetime import date # used in write_image for file name
import time # used in write_image for file name
import cv2 # used in write_image for writing images
import os # used to save images in a directory
import object_detection as detect
import snapshot as snap
#import helper as help


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
    # Main panel setup
    # Set website details
    st.set_page_config(page_title ="Objekterkennung", 
                       page_icon=':camera:', 
                       layout='centered')
    
    # Set the background
 #   help.set_bg_hack()
    
    # ===================== Set header and site info =============================
    # Set app header
#    help.header('Object detection app')
    
    # Set text and pass to sub_text function
    text = """
    <center> <br>Testumgebung für Objektklassifizierungen mit Webcam. </br> </center>
    </center>
    """
    html_temp = f"""
    <p style = "color:#1F4E79; text_align:justify;"> {text} </p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
#    help.sub_text(text)
    
    # Set expander with references and special mentions
 #   help.expander()
   
    
    # ======================= Get tf lite model details ==========================
    labels, colors, height, width, interpreter = detect.define_tf_lite_model()
    
    # ============================= Main app =====================================
    option = st.selectbox('Wähle Eingabeart aus',
                         ('Nichts', 
                          'Webcam', 
                        #  'Upload photo'
                          ))
    
    # Start with app logic:
    if option == 'Webcam':
        
        # In case Take photo is selected, run the webrtc component, 
        # save photo and pass it to the object detection model
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
           
      
        
      
# =============================================================================
#     # If option is upload photo, allow upload and pass to model
#     elif option == 'Upload photo':
#         
#         uploaded_file = st.file_uploader("Upload a photo", type=["jpg","png"])
#         
#         if uploaded_file is not None:
#             
#             st.image(uploaded_file)
#             
#             with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
#                 f.write(uploaded_file.getbuffer())  
#              
#             resultant_image = detect.display_results(labels, 
#                                                      colors, 
#                                                      height, 
#                                                      width,
#                                                      "tempDir/" + uploaded_file.name, 
#                                                      interpreter, 
#                                                      threshold=0.5)
#             
#             st.image(Image.fromarray(resultant_image), use_column_width=True)
#         
#     else:
#         help.header("Please select the type of photo you would like to classify.")
# =============================================================================


if __name__ == "__main__":
    main()
    