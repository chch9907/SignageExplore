import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', 
                        type=bool,
                        default=True)
    parser.add_argument('--config_path',
                        type=str,
                        default='config/scene1.yaml')
    parser.add_argument('--ros_config_path',
                        type=str,
                        default='config/ros.yaml')
    parser.add_argument('--OGmap_path',
                        type=str,
                        default=None)
    parser.add_argument('--ocr_type',
                        type=str,
                        default='ESTS')
    parser.add_argument('--plot',
                        action='store_true')
    parser.add_argument('--show',
                        action='store_true')
    parser.add_argument('--debug',
                        action='store_true')
    parser.add_argument('--record',
                        action='store_true')
    parser.add_argument('--data_save_path',
                        type=str,
                        default='./data')
    parser.add_argument('--rate',
                        type=int,
                        default=4)
    parser.add_argument('--scene',
                        type=str,
                        default=1)
    parser.add_argument('--max_buffer_size',
                        type=int,
                        default=2)
    parser.add_argument('--rateHz',
                        type=int,
                        default=2)
    parser.add_argument('--use_camera_api',
                        action='store_true')
    parser.add_argument('--use_camera_topic',
                        action='store_true')
    args = parser.parse_args()
    return args