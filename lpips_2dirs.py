import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='imgs/AaBaLuoKe')
parser.add_argument('-d1','--dir1', type=str, default='imgs/AaJueXingHei80J')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
img1_basename=os.path.basename(opt.dir0)
img2_basename=os.path.basename(opt.dir1)
sum_dis=0
sum_compare=0
for file in files:
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		# print('%s: %.3f'%(file,dist01))
		sum_dis+=dist01
		sum_compare+=1
		f.writelines('%s: %.6f\n'%(file,dist01))
tmp_percentage=1-(sum_dis/sum_compare).item()
if tmp_percentage<0.80:
	sum_percentage=tmp_percentage*pow(tmp_percentage,2)*100
	# sum_percentage=pow(tmp_percentage,2)*100
else:
	sum_percentage=tmp_percentage*pow(tmp_percentage,1/2)*100
	# sum_percentage=pow(tmp_percentage,1/2)*100
print("{}与{}神经网络有{:.2f}%相似".format(img1_basename,img2_basename,sum_percentage))
print("{}与{}神经网络有{:.2f}%相似".format(img1_basename,img2_basename,sum_percentage))

f.close()
