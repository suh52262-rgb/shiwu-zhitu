"""Local drag selection in original image pixels; no third-party web assets."""
import base64
import math
from pathlib import Path
import streamlit.components.v1 as components

_component=components.declare_component('shiwu_crop_selector',path=str(Path(__file__).parent/'components/crop_selector'))


def validated_box(value,width,height):
    if not isinstance(value,list) or len(value)!=4:return None
    if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in value):return None
    x0,y0,x1,y1=map(int,value)
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):return None
    return [x0,y0,x1,y1]


def select_crop(image,mime='image/png',*,key):
    encoded=base64.b64encode(image).decode('ascii')
    return _component(image_data=f'data:{mime};base64,{encoded}',key=key,default=None)
