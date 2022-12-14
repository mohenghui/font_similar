from http.client import CONTINUE
from importlib.resources import path
import os
from tkinter.tix import Tree
import cv2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import shutil
import zipfile
import math
import platform
from skimage.metrics import structural_similarity
img_type = ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP']  # 可继续添加图片类型

# 输入几行几列
ROW = 77
COL = 77


def resize_blank_img(srcimg, dstpath):
    img = cv2.imread(srcimg)
    blankimg = np.zeros(np.shape(img), np.uint8)
    blankimg[:, :, 0] = 255
    blankimg[:, :, 1] = 255
    blankimg[:, :, 2] = 255
    num = [os.path.join(dstpath, imgpath) for imgpath in os.listdir(
        dstpath) if os.path.splitext(imgpath)[1] in img_type]
    cv2.imwrite(dstpath + "\\" + str(len(num)+1) + ".jpg", blankimg)


def image_compose(image_path, save_path):
    if not os.path.isdir(image_path):
        return -1
    imgpath_vec = [os.path.join(image_path, imgpath) for imgpath in os.listdir(
        image_path) if os.path.splitext(imgpath)[1] in img_type]
    font_type = ""
    font_content = ""
    if len(imgpath_vec):
        filepath, tempfilename = os.path.split(imgpath_vec[0])
        filename, extension = os.path.splitext(tempfilename)
        font_type = filename.split('|')[0]
        font_content = filename.split('|')[1]
    # 1、使用平均的width，heigth或者可以自定义width，heigth
    avg_width = 0
    avg_heigth = 0
    if avg_width == 0 or avg_heigth == 0:
        size = []
        for item in imgpath_vec:
            size.append((Image.open(item)).size)
        sum_width = sum_heigth = 0
        for item in size:
            sum_width += item[0]
            sum_heigth += item[1]
        avg_width = int(sum_width/(len(size)))
        avg_heigth = int(sum_heigth/(len(size)))
    avg_size = (avg_width, avg_heigth)

    # 2、resize图片大小
    vec = [os.path.join(image_path, imgpath) for imgpath in os.listdir(image_path) if
           os.path.splitext(imgpath)[1] in img_type]
    while (len(vec)) < COL * ROW:
        vec = [os.path.join(image_path, imgpath) for imgpath in os.listdir(image_path) if
               os.path.splitext(imgpath)[1] in img_type]
        resize_blank_img(vec[0], image_path)

    imgs = []
    for item in vec:
        imgs.append((Image.open(item)).resize(avg_size, Image.BILINEAR))

    # 3、拼接成大图
    result_img = Image.new(imgs[0].mode, (avg_width * COL, avg_heigth * ROW))
    index = 0
    for i in range(COL):
        for j in range(ROW):
            result_img.paste(imgs[index], (i * avg_width, j * avg_heigth))
            index += 1

    # 4、显示拼接结果
    # plt.imshow(result_img)
    # plt.show()
    # print(os.path.join(save_path,font_type+img_type[2]))
    result_img.save(os.path.join(save_path, font_type+img_type[2]))  # 保存结果


def white2alpha(image_path, save_path):
    # ans=[]
    for path in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, path))  # 读取照片
        img = img.convert("RGBA")    # 转换格式，确保像素包含alpha通道
        width, height = img.size     # 长度和宽度
        for i in range(0, width):     # 遍历所有长度的点
            for j in range(0, height):       # 遍历所有宽度的点
                data = img.getpixel((i, j))  # 获取一个像素
                black_idx = 0
                white_idx = 0
                for d in data:
                    if d <= 255//2:
                        black_idx += 1
                    else:
                        white_idx += 1
                # ans.append(data)
                # if (data.count(0) == 4):  # RGBA都是255，改成透明色
                #     continue
                if black_idx >= 3:
                    # img.putpixel((i,j),(0,0,0,255))
                    continue
                else:
                    img.putpixel((i, j), (255, 255, 255, 0))
        # print(ans)
        img.save(os.path.join(save_path, os.path.split(path)[1]))  # 保存图片


# def makedir(c_path):
#     if os.path.exists(c_path):
#         shutil.rmtree(c_path)
#     # else:
#     os.mkdir(c_path)
# def makedirR(c_path):
#     if not os.path.exists(c_path):
#         # shutil.rmtree(c_path)
#     # else:
#         os.mkdir(c_path)


def makedirR(c_path, is_dir=True):
    if is_dir and not os.path.exists(c_path):
        os.mkdir(c_path)
    elif not is_dir and not os.path.exists(c_path):  # 文件新建上一级目录
        if platform.system().lower() == 'windows':
            tmp = '\\'.join(c_path.split('\\')[:-1])
        elif platform.system().lower() == 'linux':
            tmp = '/'.join(c_path.split('/')[:-1])
        if not os.path.exists(tmp):
            os.mkdir(tmp)


def fontlist():
    target_img = ""
    find_dir = "./datasets/font/train/source"
    copy_dir = "./datasets/font/train/copysource"
    makedir(copy_dir)
    for line in open('./datasets/sortfont20902.txt', encoding='utf-8'):
        target_img += line.strip()
    source_list = os.listdir(find_dir)
    source_tail = source_list[0].split('.')[1]
    for i in target_img:
        source_font = i+'.'+source_tail
        if source_font in source_list:
            src = os.path.join(find_dir, source_font)
            dst = os.path.join(copy_dir, source_font)
            shutil.copy(src, dst)

    print("over")


def rename_inference(sortfontlist, find_dir, targetpath=None):
    target_img = ""
    target_path = find_dir if not targetpath else targetpath
    # makedir(copy_dir)
    for line in open(sortfontlist, encoding='utf-8'):
        target_img += line.strip()
    source_list = os.listdir(find_dir)
    source_tail = source_list[0].split('.')[1]
    for idx, i in enumerate(target_img):
        source_font = i+'.'+source_tail
        new_name = str(idx)+'.'+source_tail
        if source_font in source_list:
            src = os.path.join(find_dir, source_font)
            dst = os.path.join(target_path, new_name)
            os.rename(src, dst)


def find_size(input_dir=None):
    # input_dir=
    # fileSize = os.path.getsize("results/B814-B812/test_unknown_style_latest/images/爱你是会呼吸的甜/阿.png")
    # print(fileSize)
    chooses = []
    input_dir = "results/B814-B812/test_unknown_style_latest/images/"
    for i in os.listdir(input_dir):
        second = os.path.join(input_dir, i)
        for j in os.listdir(second):
            file_name = os.path.join(second, j)
            if cv2.imread(file_name).shape[0] == 512:
                chooses.append(file_name)
        print(chooses, len(chooses))


def rename():
    target_img = ""
    find_dir = "./results/0705/test_unknown_style_latest/0705_og"
    copy_dir = "./results/0705/test_unknown_style_latest/copyimages"
    target_path = "./results/0705/test_unknown_style_latest/reimages"
    makedir(copy_dir)
    for line in open('./datasets/sortfont.txt', encoding='utf-8'):
        target_img += line.strip()
    source_list = os.listdir(find_dir)
    source_tail = source_list[0].split('.')[1]
    for idx, i in enumerate(target_img):
        source_font = i+'.'+source_tail
        new_name = str(idx)+'.'+source_tail
        if source_font in source_list:
            src = os.path.join(find_dir, source_font)
            dst = os.path.join(copy_dir, new_name)
            shutil.copy(src, dst)
    resize(copy_dir, target_path)


def resize(input_dir, output_dir, target_hsize=64, target_wsize=64):
    outtype = '.png'  # <---------- 输出的统一格式
    image_size_h = target_hsize
    image_size_w = target_wsize
    # source_path="./results/0705_og/test_unknown_style_latest/images"
    source_path = input_dir
    target_path = output_dir
    # target_path="./results/0705/test_unknown_style_latest/reimages"
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    image_list = os.listdir(source_path)  # 获得文件名

    i = 0
    for file in image_list:
        i = i + 1
        # print(os.path(source_path,file))
        if not os.path.splitext(file)[1].lower() in [".png", ".jpg", "bmp"]:
            return False
        image_source = cv2.imread(os.path.join(source_path, file), 0)  # 读取图片d
        # print("处理中-->",file)
        if image_source.shape[0] == image_source.shape[1]:  # 图片是正方形
            image_size_h = image_size_w
            image = cv2.resize(image_source, (image_size_w,
                               image_size_h), 0, 0, cv2.INTER_LINEAR)  # 修改尺寸
            # cv2.imwrite(target_path + str(i) + outtype, image)  # 重命名并且保存 (统一图片格式)
            cv2.imwrite(os.path.join(target_path, str(file)), image)  # 保留原命名
        else:  # 图片是非方形
            sizenum = image_source.shape[0]/image_source.shape[1]
            image_size_h = sizenum * image_size_w
            image = cv2.resize(image_source, (image_size_w, int(
                image_size_h)), 0, 0, cv2.INTER_LINEAR)  # 修改尺寸
            # cv2.imwrite(target_path + str(i) + outtype, image)  # 重命名并且保存 (统一图片格式)
            cv2.imwrite(os.path.join(target_path, str(file)), image)  # 保留原命名
    # print("批量处理完成")
    return True


def makedir(c_path, file_flag=False):
    if file_flag:
        c_path = os.path.dirname(c_path)
    if not os.path.exists(c_path):
        father_dir = os.path.dirname(c_path)
        if not os.path.exists(father_dir):
            makedir(father_dir)
        os.mkdir(c_path)


def judge(filepath, tail):
    if os.path.splitext(filepath) in tail:
        return True
    else:
        return False


def uzip(file_path, save_path, old_list):
    f = zipfile.ZipFile(file_path, 'r')  # 压缩文件位置
    makedir(save_path)
    for file in f.namelist():
        try:
            new_file_name = file.encode('cp437').decode('gbk')
        except:
            new_file_name = file.encode('utf-8').decode('utf-8')
        f.extract(file, save_path)               # 解压位置
        os.rename(os.path.join(save_path, file),
                  os.path.join(save_path, new_file_name))
        # judge()
    f.close()
    dir_list = []
    for _, dirnames, _ in os.walk(save_path):
        if dirnames:
            for i in dirnames:
                dir_list.append(i)    
    for i in old_list:
        if i in dir_list:
            dir_list.remove(i)
    new_dir = os.path.join(save_path, dir_list[0])

    for dirpath, dir, filenames in os.walk(save_path):
        # if old_list and old_list[0] in dir:continue
        if set(dir_list)>=set(dir) :
            if not dir_list :
                new_dir = os.path.join(dirpath, "myfont")
                makedir(new_dir)
                for file_name in filenames:
                    old_file_path = os.path.join(dirpath, file_name)
                    new_file_path = os.path.join(new_dir, file_name)
                    os.rename(old_file_path, new_file_path)
            elif len(dir_list) > 1 and filenames and not [i for i in old_list if i in dirpath]:
                new_dir = os.path.join(save_path, dir_list[0])
                for file_name in filenames:
                    old_file_path = os.path.join(dirpath, file_name)
                    new_file_path = os.path.join(new_dir, file_name)
                    os.rename(old_file_path, new_file_path)
    if len(dir_list) > 1:
        shutil.rmtree(os.path.join(save_path, dir_list[0], dir_list[1]))
    if not os.listdir(os.path.join(save_path, dir_list[0])) or (new_dir and not resize(new_dir, new_dir, 64)):
        return False
    else:
        return True

def new_zip(file_path, save_path, old_list):
    f = zipfile.ZipFile(file_path, 'r')  # 压缩文件位置
    makedir(save_path)
    for file in f.namelist():
        try:
            new_file_name = file.encode('cp437').decode('gbk')
        except:
            new_file_name = file.encode('utf-8').decode('utf-8')
        f.extract(file, save_path)               # 解压位置
        os.rename(os.path.join(save_path, file),
                  os.path.join(save_path, new_file_name))
        # judge()
    f.close()
    dir_list = []
    for _, dirnames, _ in os.walk(save_path):
        if dirnames:
            for i in dirnames:
                dir_list.append(i)    
    for i in old_list:
        if i in dir_list:
            dir_list.remove(i)
    new_dir = os.path.join(save_path, dir_list[0])

    for dirpath, dir, filenames in os.walk(save_path):
        # if old_list and old_list[0] in dir:continue
        if set(dir_list)>=set(dir) :
            if not dir_list :
                new_dir = os.path.join(dirpath, "myfont")
                makedir(new_dir)
                for file_name in filenames:
                    old_file_path = os.path.join(dirpath, file_name)
                    new_file_path = os.path.join(new_dir, file_name)
                    os.rename(old_file_path, new_file_path)
            elif len(dir_list) > 1 and filenames and not [i for i in old_list if i in dirpath]:
                new_dir = os.path.join(save_path, dir_list[0])
                for file_name in filenames:
                    old_file_path = os.path.join(dirpath, file_name)
                    new_file_path = os.path.join(new_dir, file_name)
                    os.rename(old_file_path, new_file_path)
    if len(dir_list) > 1:
        shutil.rmtree(os.path.join(save_path, dir_list[0], dir_list[1]))
    if not os.listdir(os.path.join(save_path, dir_list[0])) or (new_dir and not resize(new_dir, new_dir, 64)):
        return False
    else:
        return True

def pHash(path):  # 感知哈希算法
    image = cv2.imread(path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(image))
    dct_roi = dct[0:8, 0:8]
    average = np.mean(dct_roi)
    hash = []
    dct_roi_copy=dct_roi.copy()
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > average:
                hash.append(1)
            else:
                hash.append(0)
    # cv2.imshow("2",image)
    # cv2.waitKey(0)
    return hash
# 均值哈希算法


def aHash(path):
    image = cv2.imread(path)
    # 缩放为8*8
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash
# 差值感知算法


def dHash(path):
    image = cv2.imread(path)
    # 缩放9*8
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     print(image.shape)
    hash = []
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if image[i, j] > image[i, j+1]:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return (1-(num*1.0)/64)*100
def ssim_cla(path1,path2):
    img1=cv2.imread(path1)
    img2=cv2.imread(path2)
    # print(path1,path2)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    ssim_score = structural_similarity(img1, img2, data_range=255, multichannel=True)
    #范围是[0,1]
    return ssim_score*100
def psnr_cla(path1,path2):
    img1=cv2.imread(path1)
    img2=cv2.imread(path2)
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
if __name__ == "__main__":
    # import os

    # file_path = "D:/test/test.py"
    # filepath, tempfilename = os.path.split(file_path)
    # filename, extension = os.path.splitext(tempfilename)

    # path = 'results/MLANH/test_unknown_style_latest/images'#输入你的图片路径
    # save_path='results/MLANH/test_unknown_style_latest/allimg'
    # ap_save_path='results/MLANH/test_unknown_style_latest/apallimg'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # if not os.path.exists(ap_save_path):
    #     os.mkdir(ap_save_path)
    # image_compose(path,save_path)
    # white2alpha(path,ap_save_path)
    # rename()
    # resize()
    # find_size()
    pHash("./吖.png")
    # fontlist()
