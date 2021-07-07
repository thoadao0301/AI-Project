"""Create a test dataset from dataset."""
#   MIT License
#  Copyright (c) 2021. TranPhuongNam,DaoLeBaoThoa,NguyenDiemUyenPhuong
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

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
    move_data_to_path(args.input_dir, args.output_dir, args.size)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str,
                        help='Directory of raw dataset.')

    parser.add_argument('output_dir', type=str,
                        help='Directory of test dataset.')

    parser.add_argument('--size', type=float,
                        help='', default=0.2)


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))