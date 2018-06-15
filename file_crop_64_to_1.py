import os
import cv2

dir = "/Users/lastland/datasets/dcgan_64_64_64/"
output_dir = "/Users/lastland/datasets/dcgan_1_64_64/"
file_list = os.listdir(dir)
wh_list = [x*64 for x in range(0,8)]

cnt = 0
for i in range(1,16):
    print(i)
    new_dir = dir + str(i) + '/'
    print(new_dir)
    for file_name in file_list:
        for x in wh_list:
            for y in wh_list:
                # print(x, x+63, y, y+63)
                origin_img = cv2.imread(dir+file_name)
                crop_img = origin_img[x:x+64,y:y+64]
                cv2.imwrite(output_dir+'img_{:07d}.png'.format(cnt), crop_img)
                cnt += 1

                # '{:04d}.png'.format
