# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:20:48 2021

@author: Pascal
"""
import threading
from typing import Union
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings

def streamlit_webrtc_snapshot():
    '''
    
    A function to take snapshots through webrtc component.
    Also uses client settings for streamlit sharing deployment.
    
    '''
    
    # Hier werden client einstellungen getroffen
    WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    # Wir wollen ja nur Video...
    media_stream_constraints={"video": True, "audio": False},
    )
    
    # Über wrapper Klasse greifen wir auf den Stream zu
    class VideoTransformer(VideoTransformerBase):
            frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
            in_image: Union[np.ndarray, None]
            out_image: Union[np.ndarray, None]
    
            def __init__(self) -> None:
                self.frame_lock = threading.Lock()
                self.in_image = None
                self.out_image = None
    
            def transform(self, frame: av.VideoFrame) -> np.ndarray:
                in_image = frame.to_ndarray(format="bgr24")
                out_image = in_image[:, ::-1, :]  # Simple flipping for example.
                with self.frame_lock:
                    self.in_image = in_image
                    self.out_image = out_image
    
                return out_image
    
    # Hier rufen wir das Streamer Objekt
    ctx = webrtc_streamer(key="snapshot", 
                          client_settings=WEBRTC_CLIENT_SETTINGS,
                          video_transformer_factory=VideoTransformer)
    
    # Snapshot button, nur wenn tatsächlich Stream läuft
    if ctx.video_transformer:
        if st.button("Snapshot"):
            with ctx.video_transformer.frame_lock:
                out_image = ctx.video_transformer.out_image
            if out_image is not None:
                return(out_image)
            else:
                st.warning("No frames available yet.")