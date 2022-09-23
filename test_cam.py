import argparse
import lpips
# from lpips.grad_cam import GradCam,GuidedBackpropReLUModel,show_cams,show_gbs,preprocess_image
# from torchcam.methods import SmoothGradCAMpp
# import torchcamc
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import regnet_x_16gf
from torchcam.methods import SmoothGradCAMpp
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='squeeze',version=opt.version)
cam_extractor = SmoothGradCAMpp(loss_fn)
# grad_cam=GradCam()

if(opt.use_gpu):
	loss_fn.cuda()


img0,img1=lpips.resize_img(lpips.load_image(opt.path0),lpips.load_image(opt.path1))
# print(img0.shape,img1.shape)
h,w,c=img0.shape
img0 = lpips.im2tensor(img0) # RGB image from [-1,1]
img1 = lpips.im2tensor(img1)
# Load images
# img0 = lpips.im2tensor(lpips.load_image(opt.path0)) # RGB image from [-1,1]
# img1 = lpips.im2tensor(lpips.load_image(opt.path1))



if(opt.use_gpu):
	img0 = img0.cuda()
	img1 = img1.cuda()


# Compute distance
dist01 = loss_fn.forward(img0, img1)
print(dist01.data.item())
int(dist01.data.item())
dist01=dist01.squeeze(0).squeeze(0).squeeze(0)
activation_map = cam_extractor(0, dist01)
print('Distance: %.3f'%dist01)
