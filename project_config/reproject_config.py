import configargparse

def get_args():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--csv_path', type=str,
                        default="E:/PycharmProjects/pythonProject/AutomaticTranslateCorrection/data/data_attribute.csv",
                        help="Path to the input text file.")
    parser.add_argument('--txt_path', type=str,
                        default=r"E:\PycharmProjects\pythonProject\AutomaticTranslateCorrection\data_3\comparative_experiment\caps_translation.txt",
                        help="Path to the input text file.")
    parser.add_argument('--root_folder', type=str,
                        default="E:/keyan/diastolic_six_position",
                        help="Path to the input text file.")
    args = parser.parse_known_args()[0]

    return args