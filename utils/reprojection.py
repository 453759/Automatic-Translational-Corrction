'''
假设已知六个机位下的某个点的6个对应点（手动标出）
求出这6条极线在三维空间中的交点，并将交点重投影回到6个机位下，求得重投影点和原来点的距离
'''

import numpy as np

class Reprojection(object):
    def __init__(self, points, ppas, psas, dsps, dsds):
        self.points = points    # 6个
        self.ppas = ppas
        self.psas = psas
        self.dsps = dsps
        self.dsds = dsds
        self.Ts = []
        self.Ks = []
        for ppa, psa, dsp in zip(self.ppas, self.psas, self.dsps):
            self.Ts.append(self.camera_mat(ppa, psa, dsp))

        for dsd in self.dsds:
            self.Ks.append(np.array([[dsd/157.5, 0, 0.5], [0, dsd/157.5, 0.5], [0, 0, 1]]))

        self.epipolar_lines = []
        for point, T, K in zip(self.points, self.Ts, self.Ks):
            self.epipolar_lines.append(self.get_epipolar_line(point, T, K))
        self.intersection_point = self.get_intersection_point(self.epipolar_lines)
        self.reproject_points = self.reproject(self.intersection_point, self.Ts, self.Ks)
        self.error = self.get_reproject_error(self.points, self.reproject_points)

    def camera_mat(self, a, b, r):
        r = float(r) / 157.7
        theta = np.deg2rad(float(b) - 90)  # 极角
        phi = np.deg2rad(float(a))  # 极坐标

        x_camera = r * np.sin(theta) * np.cos(phi)
        y_camera = r * np.sin(theta) * np.sin(phi)
        z_camera = r * np.cos(theta)

        # 相机坐标系的三个轴在世界坐标系中的方向
        z_xc, z_yc, z_zc = -x_camera, -y_camera, -z_camera
        ###########################
        # x轴 平行yox平面
        x_xc, x_yc, x_zc = -1 / (x_camera + 10e-6), 1 / (y_camera + 10e-6), 0
        y_xc, y_yc, y_zc = ((z_yc * x_zc - x_yc * z_zc), -(z_xc * x_zc - z_zc * x_xc), (z_xc * x_yc - z_yc * x_xc))
        ##################################
        #
        if y_zc > 0:
            x_xc, x_yc, x_zc = -x_xc, -x_yc, -x_zc
            y_xc, y_yc, y_zc = -y_xc, -y_yc, -y_zc

        # 计算方向矩阵 D
        D = np.array([[x_xc, y_xc, z_xc],
                      [x_yc, y_yc, z_yc],
                      [x_zc, y_zc, z_zc]])

        # 单位化 D 的列向量
        D_prime = D / (np.linalg.norm(D, axis=0) + 10e-6)

        # 计算旋转矩阵 R
        R = D_prime
        # 计算平移矩阵 T
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x_camera, y_camera, z_camera]
        return T

    def get_epipolar_line(self, P, camera_pose, K):
        P_cam = np.dot(np.linalg.inv(K), np.concatenate([P / 512, [1]], axis=0))
        P_world = np.dot(camera_pose, np.concatenate([P_cam, [1]], axis=0))
        P_world = P_world[:3]
        P_s = camera_pose[:3, 3]
        direction = (P_world - P_s) / np.linalg.norm(P_world - P_s)
        return (P_s, direction)

    def get_intersection_point(self, epipolar_lines):
        A = []
        b = []

        for P_s, direction in epipolar_lines:
            # 方向向量必须是归一化的
            direction = direction / np.linalg.norm(direction)
            # 构建约束矩阵 A 和偏移量 b
            I = np.eye(3)  # 单位矩阵
            A.append(I - np.outer(direction, direction))  # 生成 (I - d*d^T)
            b.append((I - np.outer(direction, direction)) @ P_s)

        # 堆叠矩阵
        A = np.vstack(A)
        b = np.hstack(b)
        # 求解最小二乘问题 Ax = b
        intersection_point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return intersection_point

    def reproject(self, intersection_point, Ts, Ks):
        reproject_points = []
        for T, K in zip(Ts, Ks):
            # print(f'T={T}, K={K}')
            P_cam = np.dot(np.linalg.inv(T), np.concatenate([intersection_point, [1]], axis=0))
            # print(f'P_cam={P_cam}')
            P = np.dot(K, P_cam[:3])
            P = P / P[2] * 512
            P = P[:2]
            reproject_points.append(P)
            # print(f'P={P}')
        # print(f'reproject_points={reproject_points}')
        return reproject_points

    def get_reproject_error(self, points, reproject_points):
        error = []
        for point, reproject_point in zip(points, reproject_points):
            diff = np.linalg.norm(np.array(point) - np.array(reproject_point))
            error.append(diff)
        return error
