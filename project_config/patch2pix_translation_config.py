import configargparse

def get_args():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--txt_path', type=str,
                        default="/home/star/Projects/g4/cy/AutomaticTranslateCorrection/patch2pix/match_pairs.txt",
                        help="Path to the input text file.")
    parser.add_argument('--data_csv', type=str,
                        default="/home/star/Projects/g4/cy/AutomaticTranslateCorrection/data/data_attribute.csv",
                        help="Path to the output CSV file.")
    parser.add_argument('--num_generations', type=int, default=50,
                        help="Number of generations for processing.")
    parser.add_argument('--output_txt_path', type=str,
                        default='/home/star/Projects/g4/cy/AutomaticTranslateCorrection/data/patch2pix_translation.txt')
    args = parser.parse_known_args()[0]

    return args

