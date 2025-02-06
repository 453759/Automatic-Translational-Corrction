import configargparse


def get_args():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--checkpoint', type=str,
                        default="")
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--root', type=str, default='../../storage/',
                        help='to the parent folder of hpatches_sequences folder')
    parser.add_argument('--sequence_list', type=str, default='')
    parser.add_argument('--image_size', type=int, default=512, help='image size used')
    parser.add_argument('--Npts', type=int, default=70, help='how many matches selected')
    parser.add_argument('--iter_step', type=int, default=20)
    parser.add_argument('--im_fe_ratio', type=int, default=16)
    parser.add_argument('--device', type=int, default=0, help='which gpu should the experiment be run on')
    parser.add_argument('--benchmark', type=bool, default=False,
                        help='whether to benchmark the speed. If does, it will use the first image for 20 times')
    parser.add_argument('--select_position', type=bool, default=True, help='whether to match each two position or not')
    args = parser.parse_known_args()[0]

    return args
