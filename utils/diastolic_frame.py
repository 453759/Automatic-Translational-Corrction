import numpy as np

class DiastolicFrameDetector(object):
    def __init__(self, points):
        '''
            points is a dictionary in the format:
            Image name: points coordinates
        '''
        self.points = points
        self.centroid = self.get_centroid(self.points)
        self.average_distance = self.get_average_distance(self.points, self.centroid)
        self.diastolic_frame = self.detect_diastolic_frame(self.average_distance)

    def get_centroid(self, points):
        """
        Calculate the centroid of the points for each image.

        :param points: Dictionary where the key is the image name and the value is a list of coordinates [(x1, y1), (x2, y2), ...]
        :return: Dictionary where the key is the image name and the value is the centroid coordinates (x_centroid, y_centroid)
        """
        centroids = {}
        for image_name, coordinates in points.items():
            if coordinates is None or len(coordinates) == 0:
                # If there are no points for an image, return centroid as (0, 0) or None, can be adjusted as needed
                centroids[image_name] = (0, 0)
                continue
            # Extract x and y coordinates separately
            x_coords = [x for x, y in coordinates]
            y_coords = [y for x, y in coordinates]
            # Calculate the centroid
            x_centroid = sum(x_coords) / len(x_coords)
            y_centroid = sum(y_coords) / len(y_coords)
            centroids[image_name] = (x_centroid, y_centroid)
        return centroids

    def get_average_distance(self, points, centroid):
        """
        Calculate the average distance from each point to the centroid for each image.

        :param points: Dictionary where the key is the image name and the value is a list of coordinates [(x1, y1), (x2, y2), ...]
        :param centroid: Dictionary where the key is the image name and the value is the centroid coordinates (x_centroid, y_centroid)
        :return: Dictionary where the key is the image name and the value is the average distance
        """
        average_distances = {}
        for image_name, coordinates in points.items():
            if coordinates.size == 0 or image_name not in centroid:
                # If there are no points, or if there is no centroid data, set the average distance to 0 or another default value
                average_distances[image_name] = 0
                continue
            # Get the centroid
            x_centroid, y_centroid = centroid[image_name]
            # Calculate the Euclidean distance from each point to the centroid
            distances = [
                ((x - x_centroid) ** 2 + (y - y_centroid) ** 2) ** 0.5
                for x, y in coordinates
            ]
            # Calculate the average distance
            average_distances[image_name] = sum(distances) / len(distances)
        return average_distances

    def detect_diastolic_frame(self, average_distance):
        """
        Return the image name with the largest average distance.

        :param average_distance: Dictionary where the key is the image name and the value is the average distance
        :return: The image name with the largest average distance
        """
        if not average_distance:
            return None  # Return None if the dictionary is empty

        # Use max function to find the image name with the largest average distance
        diastolic_frame = max(average_distance, key=average_distance.get)
        return diastolic_frame
