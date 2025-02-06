import numpy as np

class DiastolicFrameDetector(object):
    def __init__(self, points):
        '''
            points是一个字典，格式为
            图片名：points坐标
        '''
        self.points = points
        self.centroid = self.get_centroid(self.points)
        self.average_distance = self.get_average_distance(self.points, self.centroid)
        self.diastolic_frame = self.detect_diastolic_frame(self.average_distance)

    def get_centroid(self, points):
        """
        计算每一张图片的点的质心。

        :param points: 字典，键为图片名，值为坐标列表 [(x1, y1), (x2, y2), ...]
        :return: 字典，键为图片名，值为质心坐标 (x_centroid, y_centroid)
        """
        centroids = {}
        for image_name, coordinates in points.items():
            if coordinates is None or len(coordinates) == 0:
                # 如果某图片没有点，返回质心为 (0, 0) 或 None，可根据需求调整
                centroids[image_name] = (0, 0)
                continue
            # 分别提取 x 和 y 坐标
            x_coords = [x for x, y in coordinates]
            y_coords = [y for x, y in coordinates]
            # 计算质心
            x_centroid = sum(x_coords) / len(x_coords)
            y_centroid = sum(y_coords) / len(y_coords)
            centroids[image_name] = (x_centroid, y_centroid)
        return centroids

    def get_average_distance(self, points, centroid):
        """
        计算每一张图片的点到质心的平均距离。

        :param points: 字典，键为图片名，值为坐标列表 [(x1, y1), (x2, y2), ...]
        :param centroid: 字典，键为图片名，值为质心坐标 (x_centroid, y_centroid)
        :return: 字典，键为图片名，值为平均距离
        """
        average_distances = {}
        for image_name, coordinates in points.items():
            if coordinates.size == 0 or image_name not in centroid:
                # 如果没有点，或者没有质心数据，平均距离为 0 或其他默认值
                average_distances[image_name] = 0
                continue
            # 获取质心
            x_centroid, y_centroid = centroid[image_name]
            # 计算每个点到质心的欧几里得距离
            distances = [
                ((x - x_centroid) ** 2 + (y - y_centroid) ** 2) ** 0.5
                for x, y in coordinates
            ]
            # 计算平均距离
            average_distances[image_name] = sum(distances) / len(distances)
        return average_distances

    def detect_diastolic_frame(self, average_distance):
        """
        返回平均距离最大的图片名。

        :param average_distance: 字典，键为图片名，值为平均距离
        :return: 平均距离最大的图片名
        """
        if not average_distance:
            return None  # 如果字典为空，返回 None

        # 使用 max 函数找到平均距离最大的图片名
        diastolic_frame = max(average_distance, key=average_distance.get)
        return diastolic_frame





