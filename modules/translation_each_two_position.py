from project_config.translation_config import get_args
import numpy as np
import os
from utils.epipolar import EpipolarLine
from modules import key_point_match as kpt_match
import pandas as pd
import utils.optimization as optimization

def extract_position_from_filename(filename):
    """
    Extract position information (ppa, psa) from the image filename.
    Example filename format: c00942_SE02_L_1_18_-2.8_27.7.png
    Returns: (ppa, psa)
    """
    try:
        parts = filename.split("_")
        ppa = float(parts[-2])  # The second-to-last part is ppa
        psa = float(parts[-1].split(".png")[0])  # The last part is psa (remove the extension)
        return (ppa, psa)
    except (IndexError, ValueError):
        print(f"Invalid filename format: {filename}")
        return None

def process_txt_file(args, k):
    txt_file_path = args.txt_path
    # Define position mappings
    position_mapping = {
        "30_20": 1,
        "0_30": 2,
        "-30_20": 3,
        "-30_-20": 4,
        "0_-30": 5,
        "45_-30": 6,
    }
    records = []
    with open(txt_file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()[6*k:6*(k+1)]]  # Only take the first 6 lines and remove extra spaces and newlines

    case_num = 1
    for i in range(0, len(lines), 6):
        group = lines[i:i + 6]  # Group every 6 lines together

        for line in group:
            line = line.strip()  # Remove extra spaces and newlines
            if not line:
                continue
            # Get the camera position directory and find the corresponding mark
            try:
                filepath = line
                filename = os.path.basename(filepath)

                position = extract_position_from_filename(filename)
                if position is None:
                    continue
                mark = position_mapping.get(line.split("/")[-2], "Unknown")  # Get the mark, default is "Unknown"
                records.append((line, position, mark))
            except IndexError:
                print(f"Invalid line format: {line}")
        case_num += 1
    return records

def translation(args, records):
    cam_info = pd.read_csv(args.data_csv)
    # Sort by marks
    sorted_combined = sorted(records, key=lambda x: x[2])
    # Unpack the sorted results
    im_pths, positions, marks = zip(*sorted_combined)
    translation_forward = []
    for i, (im_pth, position) in enumerate(zip(im_pths, positions)):
        print(f'i={i}')
        query_im_pth = im_pth
        if i == len(im_pths) - 1:
            ref_im_pth = im_pths[0]
        else:
            ref_im_pth = im_pths[i+1]
        print(f'query_pth={query_im_pth},ref_pth={ref_im_pth}')

        query_name_list = query_im_pth.split('/')[-1].split('_')
        query_img_key = '_'.join(query_name_list[:2])
        info_query = cam_info[cam_info['id'] == query_img_key]
        ppa_query, psa_query, dsp_query, dsd_query = info_query['PositionerPrimaryAngle'].values[0], \
            info_query['PositionerSecondaryAngle'].values[0],info_query['DistanceSourceToPatient'].values[0], info_query['DistanceSourceToDetector'].values[0]

        ref_name_list = ref_im_pth.split('/')[-1].split('_')
        ref_img_key = '_'.join(ref_name_list[:2])
        info_ref = cam_info[cam_info['id'] == ref_img_key]
        ppa_ref, psa_ref, dsp_ref, dsd_ref = info_ref['PositionerPrimaryAngle'].values[0], \
            info_ref['PositionerSecondaryAngle'].values[0], info_ref['DistanceSourceToPatient'].values[0], info_ref['DistanceSourceToDetector'].values[0]

        query, ref = kpt_match.get_match_points(query_im_pth, ref_im_pth)
        epipolar_line = EpipolarLine(query, ref, ppa_query, psa_query, dsp_query, dsd_query, ppa_ref, psa_ref, dsp_ref, dsd_ref)
        line_and_point_data = epipolar_line.line_and_point_data
        line_and_point_data = np.array(line_and_point_data)
        mask = (line_and_point_data[:, 3] >= args.x_min) & (line_and_point_data[:, 3] <= args.x_max) & \
               (line_and_point_data[:, 4] >= args.y_min) & (line_and_point_data[:, 4] <= args.y_max)
        line_and_point_data = line_and_point_data[mask]
        translation_forward.append(optimization.differential_evolution(optimization.fitness_function, args.num_generations, line_and_point_data[:, :3],
                                                          line_and_point_data[:, 3:], 10))

    translation_backward = []
    # print(f'im_pths={im_pths},positions={positions}')
    for i, (im_pth, position) in enumerate(zip(im_pths, positions)):
        print(f'i={i}')
        query_im_pth = im_pth
        if i == 0:
            ref_im_pth = im_pths[-1]
        else:
            ref_im_pth = im_pths[i - 1]
        print(f'query_pth={query_im_pth},ref_pth={ref_im_pth}')

        query_name_list = query_im_pth.split('/')[-1].split('_')
        query_img_key = '_'.join(query_name_list[:2])
        info_query = cam_info[cam_info['id'] == query_img_key]
        ppa_query, psa_query, dsp_query, dsd_query = info_query['PositionerPrimaryAngle'].values[0], \
            info_query['PositionerSecondaryAngle'].values[0], info_query['DistanceSourceToPatient'].values[0], \
        info_query['DistanceSourceToDetector'].values[0]

        ref_name_list = ref_im_pth.split('/')[-1].split('_')
        ref_img_key = '_'.join(ref_name_list[:2])
        info_ref = cam_info[cam_info['id'] == ref_img_key]
        ppa_ref, psa_ref, dsp_ref, dsd_ref = info_ref['PositionerPrimaryAngle'].values[0], \
            info_ref['PositionerSecondaryAngle'].values[0], info_ref['DistanceSourceToPatient'].values[0], \
        info_ref['DistanceSourceToDetector'].values[0]

        query, ref = kpt_match.get_match_points(query_im_pth, ref_im_pth)
        epipolar_line = EpipolarLine(query, ref, ppa_query, psa_query, dsp_query, dsd_query, ppa_ref, psa_ref,
                                         dsp_ref, dsd_ref)
        line_and_point_data = epipolar_line.line_and_point_data
        line_and_point_data = np.array(line_and_point_data)
        mask = (line_and_point_data[:, 3] >= args.x_min) & (line_and_point_data[:, 3] <= args.x_max) & \
                (line_and_point_data[:, 4] >= args.y_min) & (line_and_point_data[:, 4] <= args.y_max)
        line_and_point_data = line_and_point_data[mask]
        # print(f'line_and_point_data={line_and_point_data}')
        translation_backward.append(
            optimization.differential_evolution(optimization.fitness_function, args.num_generations,
                                                line_and_point_data[:, :3],
                                                line_and_point_data[:, 3:], 10))

    print(f'translation_forward={translation_forward},translation_backward={translation_backward}')

    # After computing the forward and backward translation between each pair, calculate the offset of other positions relative to the (30, 20) reference angle
    translation_total = []
    for i in range(0, 6):
        if i != 5:
            value = (translation_forward[i] - translation_backward[i + 1]) / 2
        else:
            value = (translation_forward[i] - translation_backward[0]) / 2
        translation_total.append(value)

    print(f'translation_total={translation_total}')

    # Initialize offset and forward_offset
    offset = [(0, 0)]
    forward_offset = np.array([0.0, 0.0])  # Explicitly use numpy array

    # Calculate the offset
    for i in range(0, 5):
        print(f'forward_offset_before={forward_offset}')
        forward_offset += translation_total[i]  # Accumulate translation_total
        print(f'forward_offset_after={forward_offset}')
        offset.append(tuple(forward_offset))  # Convert to regular tuple

    print(f'offset={offset}')

    '''# Calculate other offsets
    for i in range(1, 6):
        forward_offset = 0
        backward_offset = 0
        for j in range(i):
            forward_offset = forward_offset + translation_forward[j]
        for k in range(6 - i):
            backward_offset = backward_offset + translation_backward[k]
        # Convert offset to regular list and convert np.float64 to float
        offset_value = (forward_offset + backward_offset) / 2
        if isinstance(offset_value, np.ndarray):  # Check if it's an array
            offset_value = offset_value.tolist()
        offset.append([float(value) for value in offset_value])  # Convert to regular float'''

    output_txt_path = args.output_txt_path
    case_name = im_pths[0].split('/')[-1].split('_')[0]
    offsets_cleaned = [(float(x), float(y)) for x, y in offset]  # Convert to pure float
    with open(output_txt_path, 'a') as f:
        f.write(f"{case_name} {offsets_cleaned}\n")

    # Return results, remove `np.float64`, and convert to regular float
    print(im_pths[0].split('/')[-1].split('_')[0], [(list(map(float, item)) if isinstance(item, list) else item) for
                                                     item in offset])


if __name__=="__main__":
    args = get_args()
    for i in range(224):
        records = process_txt_file(args, i)
        translation(args, records)
