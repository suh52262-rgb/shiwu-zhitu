import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontend'))
from crop_selector import validated_box


def test_only_finite_original_pixel_rectangles_are_accepted():
    assert validated_box([10,20,110,220],1000,2000)==[10,20,110,220]
    assert validated_box([-1,0,10,20],1000,2000) is None
    assert validated_box([0,0,1001,20],1000,2000) is None
    assert validated_box([20,10,10,20],1000,2000) is None
    assert validated_box([0,0,float('nan'),20],1000,2000) is None
    assert validated_box([False,0,10,20],1000,2000) is None
