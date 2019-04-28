#coding=utf-8
import cv2
import os
import numpy as np
import argparse
import _thread
from multiprocessing import Pool
import time
import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Torch')
    parser.add_argument('-d','--dcm_path',default='/media/renyz/data8g/Dicom/2fy/', type=str)
    parser.add_argument('-p','--png_path',default='/media/renyz/data8g/Dicom/png_2/', type=str)
    parser.add_argument('-c','--cpu_core',default=4, type=int)
    parser.add_argument('-f','--multiprocessing_flag',default=False, type=bool)
    
    args = parser.parse_args()
    return args


def HE(image_path): 
    ori_image = cv2.imread(image_path,0)
    try:
        hist,_ = np.histogram(ori_image.flatten(),256,[0,256]) 
    except:
        return 0
    cdf = hist.cumsum() #计算累积直方图
    cdf_m = np.ma.masked_equal(cdf,0) #除去直方图中的0值
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#等同于前面介绍的lut[i] = int(255.0 *p[i])公式
    cdf = np.ma.filled(cdf_m,0).astype('uint8') #将掩模处理掉的元素补为0
    
    he_img = cdf[ori_image]
    tmp = np.where(he_img>254)[0]
    if len(tmp) < 70000:
        cv2.imwrite(image_path, he_img)

def covert(dcmlist, args): 
    for fullname in tqdm.tqdm(dcmlist):
        dest_png_path = fullname.replace(args.dcm_path,args.png_path).replace(fullname.split('/')[-2]+'/','').replace('dcm','png')
        os.system('dcmj2pnm +Wi 1 +Sxv 1024 -mf +on '+fullname+' '+dest_png_path)
        HE(dest_png_path)

def main():
    dcm_list = []
    args = arg_parse()
    floder_list = os.listdir(args.dcm_path)
    for flodername in floder_list:
        for dcm_name in os.listdir(args.dcm_path+flodername):
            dcm_list.append(args.dcm_path+flodername+'/'+dcm_name)

    new_list = []
    total_count = len(dcm_list)

    if not args.multiprocessing_flag:
        covert(dcm_list,args)
    else:
        p = Pool(args.cpu_core)
        for i in range(0,len(dcm_list)):
            new_list.append(dcm_list[i])
            if len(new_list) == int(total_count/args.cpu_core):
                p.apply_async(covert,args = (dcm_list,args,))
                new_list = []
                time.sleep(20)
            if total_count - i < int(total_count/args.cpu_core):
                p.apply_async(covert,args = (new_list,args,))

        p.close()
        p.join()

if __name__=="__main__":
    main()
