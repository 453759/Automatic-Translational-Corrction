import configargparse

def get_args():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--txt_path', type=str,
                        default="/home/star/Projects/g4/cy/AutomaticTranslateCorrection/data/diastolic_frames_R.txt",
                        help="Path to the input text file.")
    parser.add_argument('--img_root', type=str,
                        default='/home/star/Data/g4/two_position/',
                        help="Root directory of images.")
    parser.add_argument('--data_csv', type=str,
                        default="/home/star/Projects/g4/cy/AutomaticTranslateCorrection/data/data_attribute.csv",
                        help="Path to the output CSV file.")
    parser.add_argument('--num_generations', type=int, default=50,
                        help="Number of generations for processing.")
    parser.add_argument('--x_min', type=int, default=20,
                        help="Minimum x value (default: 20).")
    parser.add_argument('--x_max', type=int, default=492,
                        help="Maximum x value (default: 492).")
    parser.add_argument('--y_min', type=int, default=20,
                        help="Minimum y value (default: 20).")
    parser.add_argument('--y_max', type=int, default=492,
                        help="Maximum y value (default: 492).")
    parser.add_argument('--output_img_path', type=str,
                        default="/home/star/Data/g4/six_position_move/")
    parser.add_argument('--output_txt_path', type=str,
                        default='/home/star/Projects/g4/cy/AutomaticTranslateCorrection/data_R/ablation_experiment/different_frames/translation_12_frames.txt')
    args = parser.parse_known_args()[0]

    return args
