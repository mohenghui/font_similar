from cmath import inf
from ctypes import c_void_p
import os
from random import shuffle
from tkinter import font
from util import html
import zipfile
import json
import cv2
import shutil
from miximg import  judge, rename, rename_inference,makedir,resize, uzip,pHash,Hamming_distance,aHash,dHash,ssim_cla,psnr_cla
from flask import Flask, render_template, Response, request, redirect, url_for,send_file
from datasets.font2image_app import ttf2img
# python test.py --dataroot ./datasets/font  --model font_translator_gan  --eval --name MLANH --no_dropout 
import sys
from ocr.test import ocr
from util.visualizer import Visualizer
import torch
import time
import lpips
root_path=sys.path[0]
save_dir=os.path.join(root_path,"compare")
solo_dir=os.path.join(root_path,"compare","solo")
source_dir=os.path.join(root_path,"compare","source")
reference_dir=os.path.join(root_path,"compare","reference")
train_dir=os.path.join(root_path,"datasets/finetune/train/")
database_dir=os.path.join(root_path,"database")
#需求
#1张图片对多个文件夹里面相应图片的相似度
#一个文件夹里面的图片标签与数据库里面的相似度
#一个文件夹里面的图片标签与另一个上传的文件夹做相似度对比
#一个图片与一个图片进行对比
#
use_gpu=False

loss_fn = lpips.LPIPS(net='squeeze',version="0.1")
if(use_gpu):
    loss_fn.cuda()
    device=torch.device("cuda:0")
else:
    device = torch.device("cpu")
loss_fn.eval()

def font_img(file_path):
    # img=cv2.imread(file_path)
    return inference_solo_img(file_path)

def font_ttf(file_path):
    ttf2img(file_path,solo_dir)
    if not os.listdir(os.path.join(solo_dir,os.listdir(solo_dir)[0])):return redirect(url_for('error'))
    return inference_solo_t_z(solo_dir)

def font_zip(file_path):
    global ERROR_FLAG
    if not uzip(file_path,solo_dir,[]):return redirect(url_for('error'))
    
    # ocr(root_path,new_dir) #ocr对压缩内容打上标签
    return inference_solo_t_z(solo_dir)

def font_pair_tzi(file_path1,file_path2):
    global ERROR_FLAG
    file_tail_list=[".ttf",".otf"]
    imgfile_tail_list=[".png",".jpg","jpeg",".bmp"]
    file_tail1=os.path.splitext(os.path.basename(file_path1))[1]

    file_tail2=os.path.splitext(os.path.basename(file_path2))[1]
    old_list=[]
    # old_list.append(os.path.splitext(os.path.basename(file_path1))[0])
    # if os.path.exists(reference_dir):shutil.rmtree(reference_dir)
    # if os.path.exists(source_dir):shutil.rmtree(source_dir)
    if file_tail1==".zip" and not uzip(file_path1,reference_dir,[]):
        return redirect(url_for('error'))
    if file_tail2==".zip" and not uzip(file_path2,source_dir,[]):
        return redirect(url_for('error'))
    # old_list.append(os.listdir(reference_dir)[0])
    if file_tail1.lower() in file_tail_list:
        ttf2img(file_path1,reference_dir)
    if file_tail2.lower() in file_tail_list:
        ttf2img(file_path2,source_dir)
    if file_tail1.lower() in imgfile_tail_list:
        new_path1=os.path.join(reference_dir,"1",os.path.basename(file_path1))
        makedir(new_path1,True)
        os.rename(file_path1,new_path1)
    if file_tail2.lower() in imgfile_tail_list:
        new_path2=os.path.join(source_dir,"2",os.path.basename(file_path2))
        makedir(new_path2,True)
        os.rename(file_path2,new_path2)

    # ocr(root_path,os.path.join(source_dir,os.listdir(source_dir)[0]))
    if not os.listdir(os.path.join(source_dir,os.listdir(source_dir)[0])):return redirect(url_for('error'))
    if not os.listdir(os.path.join(reference_dir,os.listdir(reference_dir)[0])):return redirect(url_for('error'))
    return inference_pair_t_z(reference_dir,source_dir)

def font_pair_zip(file_path1,file_path2):
    global ERROR_FLAG
    old_list=[]
    # old_list.append(os.path.splitext(os.path.basename(file_path1))[0])
    if os.path.exists(reference_dir):shutil.rmtree(reference_dir)
    if not uzip(file_path1,reference_dir,[]):
        return redirect(url_for('error'))
    # old_list.append(os.listdir(reference_dir)[0])
    if not uzip(file_path2,source_dir,[]):
        return redirect(url_for('error'))
    return inference_pair_t_z(reference_dir,source_dir)

def font_pair_ttf(file_path1,file_path2):
    if os.path.exists(reference_dir):shutil.rmtree(reference_dir)
    if os.path.exists(source_dir):shutil.rmtree(source_dir)
    ttf2img(file_path1,reference_dir)
    ttf2img(file_path2,source_dir)
    if not os.listdir(os.path.join(source_dir,os.listdir(source_dir)[0])):return redirect(url_for('error'))
    if not os.listdir(os.path.join(reference_dir,os.listdir(reference_dir)[0])):return redirect(url_for('error'))
    return inference_pair_t_z(reference_dir,source_dir)


def inference_solo_t_z(file_path):
    file_dir=os.path.join(file_path,os.listdir(file_path)[0])

    max_dis=0
    similar_list=[]
    base_list=os.listdir(database_dir)
    n=len(base_list)
    index=0
    while index<n:
        sum_dist=0
        sum_compare=0
        for file in os.listdir(file_dir):
            file_path=os.path.join(file_dir,file)
            label,_=os.path.splitext(os.path.basename(file_path))
            img0 = lpips.im2tensor(lpips.load_image(file_path)) # RGB image from [-1,1]
            
            d_file_path=os.path.join(database_dir,base_list[index])
            # if label+".png" in os.listdir(d_file_path):
            target_path=os.path.join(d_file_path,label+".png")
            if os.path.exists(target_path):
                # print(os.path.join(d_file_path,label+".png"))
                img1 = lpips.im2tensor(lpips.load_image(target_path))
                img0 = img0.to(device)
                img1 = img1.to(device)
                sum_dist += (1-loss_fn.forward(img0,img1))
                # print('%s: %.3f'%(file,dist01))
                sum_compare+=1
        dist01=sum_dist/sum_compare
        if dist01>=max_dis:
            if max_dis==dist01:similar_list.append(base_list[index])
            else:
                similar_list=[]
                similar_list.append(base_list[index])
                max_dis=dist01
        print(base_list[index])
        print(max_dis,similar_list)
        index+=1
    if sum_compare==0:return 0
    # tmp_percentage=1-(max_dis).item()
    tmp_percentage=(max_dis).item()
    if tmp_percentage<0.80:
        sum_percentage=tmp_percentage*pow(tmp_percentage,2)*100
    else:
        sum_percentage=tmp_percentage*pow(tmp_percentage,1/2)*100
    ph,ah,dh,ssim,psnr=other_similar_solo(file_dir,database_dir,similar_list,file_flag=False)
    return sum_percentage,similar_list,ph,ah,dh,ssim,psnr


def inference_solo_img(file_path):
    label,_=os.path.splitext(os.path.basename(file_path))
    # p1=pHash(file_path)
    img0 = lpips.im2tensor(lpips.load_image(file_path)) # RGB image from [-1,1]
    sum_compare=0
    max_dis=0
    similar_list=[]
    for d_file in os.listdir(database_dir):
        d_file_path=os.path.join(database_dir,d_file)
        # if label+".png" in os.listdir(d_file_path):
        if os.path.exists(os.path.join(d_file_path,label+".png")):
            # os.path.exists(os.path.join(d_file_path,label+".png"))
            # print(os.path.join(d_file_path,label+".png"))
            label_path=os.path.join(d_file_path,label+".png")
            # p2=pHash(label_path)
            img1 = lpips.im2tensor(lpips.load_image(label_path))
            img0 = img0.to(device)
            img1 = img1.to(device)
            dist01 = 1-loss_fn.forward(img0,img1)
            # print('%s: %.3f'%(file,dist01))
            if dist01>=max_dis:
                if max_dis==dist01:similar_list.append(d_file)
                else:
                    similar_list=[]
                    similar_list.append(d_file)
                max_dis=dist01
            sum_compare+=1
    if sum_compare==0:return 0
    # tmp_percentage=1-(max_dis).item()
    tmp_percentage=(max_dis).item()
    if tmp_percentage<0.80:
        sum_percentage=tmp_percentage*pow(tmp_percentage,2)*100
    else:
        sum_percentage=tmp_percentage*pow(tmp_percentage,1/2)*100
    ph,ah,dh,ssim,psnr=other_similar_solo(file_path,database_dir,similar_list)
    return sum_percentage,similar_list,ph,ah,dh,ssim,psnr


def other_similar_solo(file_path1,file_path2,compare_list,file_flag=True,pair=True,img_flag=False):
    ph_similar=[]
    ah_similar=[]
    dh_similar=[]
    ssim_similar=[]
    psnr_similar=[]
    for similar in compare_list:
        if pair:other_compare=os.listdir(os.path.join(file_path2,similar))
        else:other_compare=[str(file_path2)]
        if file_flag:
            basename=os.path.splitext(os.path.basename(file_path1))[0]
            for other in other_compare:
                if  basename in other or img_flag:
                    p1=pHash(file_path1)
                    a1=aHash(file_path1)
                    d1= dHash(file_path1)
                    # s1=ssim_cla(file_path1)
                    label_path=os.path.join(file_path2,similar,other)
                    p2=pHash(label_path)
                    a2=aHash(label_path)
                    d2= dHash(label_path)
                    s_out=ssim_cla(file_path1,label_path)
                    ssim_similar.append(s_out)
                    p_out=psnr_cla(file_path1,label_path)
                    psnr_similar.append(p_out)
                    ph_similar.append(Hamming_distance(p1,p2))
                    ah_similar.append(Hamming_distance(a1,a2))
                    dh_similar.append(Hamming_distance(d1,d2))
        else:
            for file in os.listdir(file_path1):
                file_path_tmp=os.path.join(file_path1,file)
                basename=os.path.splitext(os.path.basename(file_path_tmp))[0]
                for other in other_compare:
                    if  basename in other:
                        p1=pHash(file_path_tmp)
                        a1=aHash(file_path_tmp)
                        d1= dHash(file_path_tmp)
                        label_path=os.path.join(file_path2,similar,other)
                        p2=pHash(label_path)
                        a2=aHash(label_path)
                        d2= dHash(label_path)
                        s_out=ssim_cla(file_path_tmp,label_path)
                        ssim_similar.append(s_out)
                        p_out=psnr_cla(file_path_tmp,label_path)
                        psnr_similar.append(p_out)
                        ph_similar.append(Hamming_distance(p1,p2))
                        ah_similar.append(Hamming_distance(a1,a2))
                        dh_similar.append(Hamming_distance(d1,d2))
        ph_similar.append(sum(ph_similar)/len(ph_similar))
        ah_similar.append(sum(ah_similar)/len(ah_similar))
        dh_similar.append(sum(dh_similar)/len(dh_similar))
        ssim_similar.append(sum(ssim_similar)/len(ssim_similar))
        psnr_similar.append(sum(psnr_similar)/len(psnr_similar))
    return ph_similar,ah_similar,dh_similar,ssim_similar,psnr_similar



def inference_pair_t_z(file_path1,file_path2):
    file_dir1=os.path.join(file_path1,os.listdir(file_path1)[0])
    file_dir2=os.path.join(file_path2,os.listdir(file_path2)[0])
    sum_percentage=similari_list=ph=ah=dh=ssim=psnr=0
    similari_list=[]
    name_list=os.listdir(file_dir2)
    sum_dist=0
    sum_compare=0
    eps=0.000000001
    for file in os.listdir(file_dir1):
        file_path=os.path.join(file_dir1,file)
        label,_=os.path.splitext(os.path.basename(file_path))
        for name in name_list:
            if label in name :
                img0 = lpips.im2tensor(lpips.load_image(file_path)) # RGB image from [-1,1]
                
                img1 = lpips.im2tensor(lpips.load_image(os.path.join(file_dir2,name))) # RGB image from [-1,1]
                img0 = img0.to(device)
                img1 = img1.to(device)
                sum_dist += (1-loss_fn.forward(img0,img1))
                # print('%s: %.3f'%(file,dist01))
                sum_compare+=1
    dist01=sum_dist/(sum_compare+eps)
    if sum_compare==0:return sum_percentage,similari_list,ph,ah,dh,ssim,psnr
    tmp_percentage=(dist01).item()
    if tmp_percentage<0.80:
        sum_percentage=tmp_percentage*pow(tmp_percentage,2)*100
    else:
        sum_percentage=tmp_percentage*pow(tmp_percentage,1/2)*100
    ph,ah,dh,ssim,psnr=other_similar_solo(file_dir1,file_dir2,[""],file_flag=False)
    return sum_percentage,similari_list,ph,ah,dh,ssim,psnr


def inference_pair_img(file_path1,file_path2):
    img0 = lpips.im2tensor(lpips.load_image(file_path1)) # RGB image from [-1,1]
    img1=lpips.im2tensor(lpips.load_image(file_path2))
    d_file,_=os.path.splitext(os.path.basename(file_path2))
    img0 = img0.to(device)
    img1 = img1.to(device)
    dist01 = 1-loss_fn.forward(img0,img1)
    similari_list=[]
    tmp_percentage=(dist01).item()
    if tmp_percentage<0.80:
        sum_percentage=tmp_percentage*pow(tmp_percentage,2)*100
    else:
        sum_percentage=tmp_percentage*pow(tmp_percentage,1/2)*100
    ph,ah,dh,ssim,psnr=other_similar_solo(file_path1,file_path2,[""],file_flag=True,pair=False,img_flag=True)
    return sum_percentage,similari_list,ph,ah,dh,ssim,psnr

def font_pair_img(file_path1,file_path2):
    old_file_name1=file_path1
    new_file_name1=os.path.join(reference_dir,os.path.basename(file_path1))
    old_file_name2=file_path2
    new_file_name2=os.path.join(source_dir,os.path.basename(file_path2))
    os.rename(old_file_name1,new_file_name1)
    os.rename(old_file_name2,new_file_name2)
    return inference_pair_img(new_file_name1,new_file_name2)

app = Flask(__name__)
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template("index.html")

@app.route('/error')
def error():
    """Video streaming home page."""
    return render_template("error.html")


@app.route('/font_upload', methods=['POST'])
def font_upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        if os.path.exists(solo_dir):shutil.rmtree(solo_dir) 
        os.mkdir(solo_dir)
        upload_file_path = os.path.join(basepath, solo_dir, (f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_file_path)
        global  GENERATE_FLAG,ERROR_FLAG
        file_name,file_tail=os.path.splitext(f.filename)
        ERROR_FLAG=False
        ttffile_tail_list=[".ttf",".otf"]
        imgfile_tail_list=[".png",".jpg","jpeg",".bmp"]
        if file_tail==".zip":
            accurate,similari_list,ph,ah,dh,ssim,psnr=font_zip(upload_file_path)
            
        elif file_tail.lower() in ttffile_tail_list:
            accurate,similari_list,ph,ah,dh,ssim,psnr=font_ttf(upload_file_path)
        elif file_tail.lower() in imgfile_tail_list:
            accurate,similari_list,ph,ah,dh,ssim,psnr=font_img(upload_file_path)
        else:
            return redirect(url_for('error'))
        if accurate:
            # print(ph,ah,dh)
            return "与{}相似度最高，神经网络相似程度为{},感知哈希相似程度为{},均值哈希相似程度为{},差值感知相似程度为{},ssim相似程度为{},psnr相似程度为{}".format(str(similari_list),str(accurate),str(ph[0]),str(ah[0]),str(dh[0]),str(ssim[0]),str(psnr[0]))
        else:
            return "没有在字库中找到相应的汉字"
    else:return redirect(url_for('index'))



@app.route('/font_upload1', methods=['POST'])
def font_upload1():
    if request.method == 'POST':
        f1 = request.files['file1']
        f2 = request.files['file2']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(root_path,'static/uploads')
        if not os.path.exists(upload_path):
            os.mkdir(upload_path)
        else:
            shutil.rmtree(upload_path) 
            os.mkdir(upload_path)
        upload_file_path1 = os.path.join(basepath, upload_path,"1", (f1.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        upload_file_path2 = os.path.join(basepath, upload_path,"2",(f2.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        makedir(upload_file_path1,True)
        makedir(upload_file_path2,True)
        f1.save(upload_file_path1)
        f2.save(upload_file_path2)
        global ERROR_FLAG
        file_name1,file_tail1=os.path.splitext(f1.filename)
        file_name2,file_tail2=os.path.splitext(f2.filename)
        ERROR_FLAG=False
        file_tail_list=[".ttf",".otf"]
        imgfile_tail_list=[".png",".jpg","jpeg",".bmp"]
        if os.path.exists(reference_dir):shutil.rmtree(reference_dir) 
        if os.path.exists(source_dir):shutil.rmtree(source_dir) 
        makedir(reference_dir)
        makedir(source_dir)
        # if file_tail==".zip":
        #     # accurate,similari_list,ph,ah,dh,ssim,psnr=font_pair_zip(upload_file_path1,upload_file_path2)
        #     accurate,similari_list,ph,ah,dh,ssim,psnr=font_pair_tzi(upload_file_path1,upload_file_path2)
        # elif file_tail.lower() in file_tail_list:
        #     accurate,similari_list,ph,ah,dh,ssim,psnr=font_pair_ttf(upload_file_path1,upload_file_path2)
        # elif file_tail.lower() in imgfile_tail_list:
        #     accurate,similari_list,ph,ah,dh,ssim,psnr=font_pair_img(upload_file_path1,upload_file_path2)
        if (file_tail1.lower() in file_tail_list or file_tail1.lower() in imgfile_tail_list or file_tail1.lower()==".zip")and (file_tail2.lower() in file_tail_list or file_tail2.lower() in imgfile_tail_list or file_tail2.lower()==".zip"):
            accurate,similari_list,ph,ah,dh,ssim,psnr=font_pair_tzi(upload_file_path1,upload_file_path2)
        else:
            return redirect(url_for('error'))
        if accurate:
            return "神经网络相似程度为{},感知哈希相似程度为{},均值哈希相似程度为{},差值感知相似程度为{},ssim相似程度为{},psnr相似程度为{}".format(str(accurate),str(ph[0]),str(ah[0]),str(dh[0]),str(ssim[0]),str(psnr[0]))
            # return json.dumps({'file_id': similari_list, 'filename': ph[0] , 'links_to' : ah})
        else:
            return "没有在字库中找到相应的汉字"
    else:return redirect(url_for('index'))






if __name__ == '__main__':
    # app.run(host='127.0.0.1', threaded=True, port=8080)
    app.run(host='192.168.4.149', threaded=True, port=8080)
    