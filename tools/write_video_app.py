import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Merge a batch of videos after anonymization')
parser.add_argument('--base_dir', default='../../data/tennis', help='datasets')
args = parser.parse_args()


if __name__ == '__main__':
    base_command = "python ../../tools/video_writer.py"
    for subfile in os.listdir(args.base_dir):
        command = f"{base_command} --base_path '{args.base_dir}/{subfile}' --target_path '../../{subfile[:-4]}/'"
        try:
            if os.path.isdir(f'{args.base_dir}/{subfile}') or not os.path.isdir(f'../../{subfile[:-4]}/'):
                continue
            if 'mp4' in subfile.lower() and "result.mp4" not in os.listdir(f'../../{subfile[:-4]}/'):
                print(command)
                os.system(command)
        except Exception as e:
            print(e)
            pass
