import numpy as np

class CameraWorldPixel(object):
    def __init__(self, ppa, psa, dsp):
        self.ppa = ppa
        self.psa = psa
        self.dsp = dsp
        self.cam_pose = self.campolar2rotation(self.ppa, self.psa, self.dsp)
        
    def campolar2rotation(self, a, b, r):
        r = float(r) / 157.7
        theta = np.deg2rad(float(b) - 90)  # Polar angle
        phi = np.deg2rad(float(a))  # Polar coordinates

        x_camera = r * np.sin(theta) * np.cos(phi)
        y_camera = r * np.sin(theta) * np.sin(phi)
        z_camera = r * np.cos(theta)

        # Directions of the three camera axes in the world coordinate system
        z_xc, z_yc, z_zc = -x_camera, -y_camera, -z_camera
        ###########################
        # x-axis parallel to the y-x plane
        x_xc, x_yc, x_zc = -1 / (x_camera + 10e-6), 1 / (y_camera + 10e-6), 0
        y_xc, y_yc, y_zc = ((z_yc * x_zc - x_yc * z_zc), -(z_xc * x_zc - z_zc * x_xc), (z_xc * x_yc - z_yc * x_xc))
        ##################################
        #
        if y_zc > 0:
            x_xc, x_yc, x_zc = -x_xc, -x_yc, -x_zc
            y_xc, y_yc, y_zc = -y_xc, -y_yc, -y_zc

        # Calculate the direction matrix D
        D = np.array([[x_xc, y_xc, z_xc],
                      [x_yc, y_yc, z_yc],
                      [x_zc, y_zc, z_zc]])

        # Normalize the columns of D
        D_prime = D / (np.linalg.norm(D, axis=0) + 10e-6)

        # Compute the rotation matrix R
        R = D_prime
        # Compute the translation matrix T
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x_camera, y_camera, z_camera]
        return T


def pixel2world(P, cam2world, focal_length):
    K = [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]]
    P_cam = (np.dot(np.linalg.inv(K), np.concatenate([P / 512, [1]], axis=0))) * 829.4015277 / 157.7
    # print(f'P_cam={P_cam}')
    P_world = np.dot(cam2world, np.concatenate([P_cam, [1]], axis=0))
    return P_world


def world2pixel(P_world, camera2world, focal_length):
    P_cam = np.dot(np.linalg.inv(camera2world), np.concatenate([P_world, [1]], axis=0))
    K = [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]]
    P = np.dot(K, P_cam[:3])
    P = P / P[2] * 512
    return P[:2]
