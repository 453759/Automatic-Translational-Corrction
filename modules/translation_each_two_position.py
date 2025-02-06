from project_config.translation_config import get_args
import numpy as np
import os
from utils.epipolar import EpipolarLine
from modules import key_point_match as kpt_match
import pandas as pd
import utils.optimization as optimization

def extract_position_from_filename(filename):
    """
    从图片名中提取位置信息（ppa, psa）。
    图片名格式示例：c00942_SE02_L_1_18_-2.8_27.7.png
    返回：(ppa, psa)
    """
    try:
        parts = filename.split("_")
        ppa = float(parts[-2])  # 倒数第二个部分是 ppa
        psa = float(parts[-1].split(".png")[0])  # 倒数第一个部分是 psa（去掉扩展名）
        return (ppa, psa)
    except (IndexError, ValueError):
        print(f"Invalid filename format: {filename}")
        return None

def process_txt_file(args, k):
    txt_file_path = args.txt_path
    # 定义机位对应的标号
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
        lines = [line.strip() for line in file.readlines()[6*k:6*(k+1)]]  # 只取前 6 行并去掉多余空格和换行符

    case_num = 1
    for i in range(0, len(lines), 6):
        group = lines[i:i + 6]  # 每 6 行一组

        for line in group:
            line = line.strip()  # 去掉多余空格和换行符
            if not line:
                continue
            # 获取机位目录并找到对应标号
            try:
                filepath = line
                filename = os.path.basename(filepath)

                position = extract_position_from_filename(filename)
                if position is None:
                    continue
                mark = position_mapping.get(line.split("/")[-2], "Unknown")  # 获取标号，默认 "Unknown"
                records.append((line, position, mark))
            except IndexError:
                print(f"Invalid line format: {line}")
        case_num += 1
    return records

def translation(args, records):
    cam_info = pd.read_csv(args.data_csv)
    # 按 marks 排序
    sorted_combined = sorted(records, key=lambda x: x[2])
    # 解包排序后的结果
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

    # 已经算出正向和反向两两间的平移量，现以(30, 20)为基准图，计算其余机位相对这个角度的偏移
    translation_total = []
    for i in range(0, 6):
        if i != 5:
            value = (translation_forward[i] - translation_backward[i + 1]) / 2
        else:
            value = (translation_forward[i] - translation_backward[0]) / 2
        translation_total.append(value)

    print(f'translation_total={translation_total}')

    # 初始化 offset 和 forward_offset
    offset = [(0, 0)]
    forward_offset = np.array([0.0, 0.0])  # 显式使用 numpy 数组

    # 计算 offset
    for i in range(0, 5):
        print(f'forward_offset_before={forward_offset}')
        forward_offset += translation_total[i]  # 累加 translation_total
        print(f'forward_offset_after={forward_offset}')
        offset.append(tuple(forward_offset))  # 转换为普通 tuple

    print(f'offset={offset}')

    '''# 计算其他偏移量
    for i in range(1, 6):
        forward_offset = 0
        backward_offset = 0
        for j in range(i):
            forward_offset = forward_offset + translation_forward[j]
        for k in range(6 - i):
            backward_offset = backward_offset + translation_backward[k]
        # 转换偏移量为普通列表并将 np.float64 转换为 float
        offset_value = (forward_offset + backward_offset) / 2
        if isinstance(offset_value, np.ndarray):  # 检查是否是数组
            offset_value = offset_value.tolist()
        offset.append([float(value) for value in offset_value])  # 转换为普通 float'''

    output_txt_path = args.output_txt_path
    case_name = im_pths[0].split('/')[-1].split('_')[0]
    offsets_cleaned = [(float(x), float(y)) for x, y in offset]  # 转换为纯 float
    with open(output_txt_path, 'a') as f:
        f.write(f"{case_name} {offsets_cleaned}\n")

    # 返回结果，去掉 `np.float64`，转换为普通 float
    print(im_pths[0].split('/')[-1].split('_')[0], [(list(map(float, item)) if isinstance(item, list) else item) for
                                                     item in offset])


if __name__=="__main__":
    args = get_args()
    for i in range(224):
        records = process_txt_file(args, i)
        translation(args, records)

