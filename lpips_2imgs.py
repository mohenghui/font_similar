import argparse
import lpips
from lpips.grad_cam import GradCam,GuidedBackpropReLUModel,show_cams,show_gbs,preprocess_image
from torchcam.methods import SmoothGradCAMpp

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='squeeze',version=opt.version)
# grad_cam=GradCam()
if(opt.use_gpu):
	loss_fn.cuda()

# Load images
img0 = lpips.im2tensor(lpips.load_image(opt.path0)) # RGB image from [-1,1]
img1 = lpips.im2tensor(lpips.load_image(opt.path1))

if(opt.use_gpu):
	img0 = img0.cuda()
	img1 = img1.cuda()
# 
cam_extractor = SmoothGradCAMpp(loss_fn,target_layer="lin6.model.1")
# Compute distance
dist01 = loss_fn.forward(img0, img1)
dist01=dist01.squeeze(0).squeeze(0).squeeze(0)
cam_extractor(0,dist01)
print('Distance: %.3f'%dist01)
