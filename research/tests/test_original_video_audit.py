import cv2
import numpy as np

from research.audit_original_video import object_number, video_family
from research.measure_visual_centers import silhouette_proxy


def test_video_families_preserve_object_linkage():
    assert video_family("새30(1)_tracksequence.json") == video_family("새_30(객체두개).mp4")
    assert video_family("동규_새_10_위에 새.json") == video_family("동규_새_10(객체두개).MOV")
    assert video_family("드론21_track_sequence.json") == video_family("드론_21.mov")
    assert object_number("새36(2)_tracksequence.json") == object_number("새_36(2번객체).mp4")


def test_silhouette_proxy_is_a_local_candidate_not_distant_clutter():
    image=np.full((128,128,3),190,dtype=np.uint8)
    cv2.circle(image,(65,63),4,(10,10,10),-1)
    cv2.rectangle(image,(80,80),(90,90),(0,0,0),-1)
    result=silhouette_proxy(image,64,64)
    assert result is not None
    assert abs(result["x"]-65)<.5 and abs(result["y"]-63)<.5
    assert 20 < result["area_px"] < 100
