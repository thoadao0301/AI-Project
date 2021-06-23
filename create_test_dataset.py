import sys
import os
import argparse
from tqdm import tqdm
from shutil import move

'''
Split dataset from src directory to des directory with the size 20% of the src directory
'''
def move_data_to_path(src, des,size):
    list_sub_dir = os.listdir(src)
    try:
        os.mkdir(des)
    except:
        pass
    n = len(list_sub_dir)
    for i in range(n):
        print(i,'/',n,':',list_sub_dir[i])
        sub_src_path = os.path.join(src, list_sub_dir[i])
        sub_des_path = os.path.join(des, list_sub_dir[i])
        try:
            os.mkdir(sub_des_path)
        except:
            continue
        images_src = os.listdir(sub_src_path)
        num = int(len(images_src)*size)
        count = 1
        with tqdm(total=num, file=sys.stdout) as pbar:
            for image in images_src:
                image_src_path = os.path.join(sub_src_path, image)
                image_des_path = os.path.join(sub_des_path, image)
                if os.path.exists(image_des_path):
                    continue
                move(image_src_path, image_des_path)

                pbar.update(1)
                count += 1
                if count > num:
                    break

def main(args):
    move_data_to_path(args.src, args.des, args.size)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('src', type=str,
                        help='Path to the data directory')

    parser.add_argument('des', type=str,
                        help='Path to the data directory')

    parser.add_argument('--size', type=float,
                        help='', default=0.2)


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))