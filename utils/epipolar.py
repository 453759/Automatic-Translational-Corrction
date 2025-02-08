import numpy as np
import utils.camera as camera
from utils.camera import CameraWorldPixel

class EpipolarLine(object):
    def __init__(self, query, ref, ppa_query, psa_query, dsp_query, dsd_query, ppa_ref, psa_ref, dsp_ref, dsd_ref):
        self.query = query
        self.ref = ref
        self.ppa_query = ppa_query
        self.psa_query = psa_query
        self.dsp_query = dsp_query
        self.dsd_query = dsd_query
        self.ppa_ref = ppa_ref
        self.psa_ref = psa_ref
        self.dsp_ref = dsp_ref
        self.dsd_ref = dsd_ref
        self.camera_pose_query = CameraWorldPixel(ppa_query, psa_query, dsp_query).cam_pose
        self.camera_pose_ref = CameraWorldPixel(ppa_ref, psa_ref, dsp_ref).cam_pose
        self.line_and_point_data = self.get_epipolar_line(self.query, self.ref, self.dsd_query, self.dsd_ref)

    def get_epipolar_line(self, query, ref, dsd_query, dsd_ref):
        line_and_point_data = []
        for (q, r) in zip(query, ref):
            x1, y1 = q
            x2, y2 = r
            p_w = camera.pixel2world(q, self.camera_pose_query, dsd_query)
            p_w = p_w[:3]
            p_s = self.camera_pose_query[:3, 3]
            p_pix_b = camera.world2pixel(p_w, self.camera_pose_ref, dsd_ref)
            p_pix_s = camera.world2pixel(p_s, self.camera_pose_ref, dsd_ref)
            A = p_pix_b[1] - p_pix_s[1]
            B = p_pix_b[0] - p_pix_s[0]
            k = A / B
            b = p_pix_b[1] - k * p_pix_b[0]
            line_and_point_data.append([-k, 1, -b, x2, y2])
        np.array(line_and_point_data)
        return line_and_point_data

