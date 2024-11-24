import argparse
from ast import arg
import logging
import os
import time
import torch
# from dataset import SonyTestDataset
from torch.utils.data import DataLoader
import scipy.io
from tqdm import tqdm  # 修改导入语句
import numpy as np
# from util import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from DataLoader import *
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from models.whole_new_dwt_RCABS import *
# from models.whole_unet_RCABS import *
# from models.whole_new_dwt_RCABS import *
# from models.copy_whole_new_dwt_RCABS import *
# from models.whole_new_dwt_RCABS_return_two import *
# from models.whole_new_dwt_RCABS_Sony_down import *
# from models.model import UNet
# from compare_model.SUnet import *
from datasets.dataset import SonyTestDataset, SonyTestFullDataset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def get_lpips_torch(x, gt):
    '''
    x,gt : tensor with shape (b,3,h,w), data_range is [0,1]
    '''
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
    x = x * 2.0 - 1.0
    gt = gt * 2.0 - 1.0
    lpips_score = lpips(x, gt)
    return lpips_score

def test(args):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    # 如果'logs'文件夹不存在，则创建一个
    logs_folder =  os.path.join(args.result_dir, 'logs')
    
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    
    # 设置日志记录
    logging.basicConfig(filename=os.path.join(logs_folder, 'test.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

    # data
    # testset = SonyTestDataset(args.input_dir, args.gt_dir)
    testset = SonyTestFullDataset(args.test_path)
    print(len(testset)) # 598
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    model = MGCC()
    # model = torch.load(args.model)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model))
    
    model.to(device)
    model.eval()

    # 获取迭代器的长度
    total_batches = len(test_loader)
    print(total_batches) # 598
    # 初始化 PSNR 和 SSIM 列表
    psnr_list = []
    ssim_list = []
    lpips_list = []
    with torch.no_grad(): ###插在此处
        # testing
        for i, databatch in tqdm(enumerate(test_loader), total=total_batches):
            etime = time.time()
            input_full, scale_full, gt_full, test_id, ratio = databatch
            # 使用 permute() 方法重新排列维度
            gt_patch = gt_full.permute(0, 3, 1, 2).to(device)
            # print(f"gt_patch的大小为{gt_patch.shape},类型为{type(gt_patch)}")  # gt_patch的大小为torch.Size([1, 3, 1024, 1024]),类型为<class 'torch.Tensor'>
            # print(f"gt_patch.max()为{gt_patch.max()}") # gt_full.max()为1.0
            # print(f"gt_patch.min()为{gt_patch.min()}") # gt_full.min()为0.0
            # torch.Size([1, 4, 512,512]) 
            # torch.Size([1, 1024, 1024, 3])
            # torch.Size([1, 1024, 1024, 3])
            # print(f"input_full的大小为{input_full.shape},类型为{type(input_full)}")  # input_full的大小为torch.Size([1, 4, 1424, 2128]),类型为<class 'torch.Tensor'>
            # print(f"scale_full的大小为{scale_full.shape},类型为{type(scale_full)}")  # scale_full的大小为torch.Size([1, 2848, 4256, 3]),类型为<class 'torch.Tensor'>
            # print(f"gt_full的大小为{gt_full.shape},类型为{type(gt_full)}")           # gt_full的大小为torch.Size([1, 2848, 4256, 3]),类型为<class 'torch.Tensor'>
            
            scale_full, gt_full = torch.squeeze(scale_full), torch.squeeze(gt_full)
            
            # print(f"scale_full的大小为{scale_full.shape},类型为{type(scale_full)}") # scale_full的大小为torch.Size([2848, 4256, 3]),类型为<class 'torch.Tensor'>
            # print(f"gt_full的大小为{gt_full.shape},类型为{type(gt_full)}") # gt_full的大小为torch.Size([2848, 4256, 3]),类型为<class 'torch.Tensor'>

            # processing
            inputs = input_full.to(device) # torch.Size([1, 4, 512, 512])
            # print(inputs.shape)
            outputs = model(inputs)        # torch.Size([1, 3, 1024, 1024])
            # long_raw,outputs = model(inputs)
            
            # 将像素值限制在[0, 1]范围内
            # long_raw = torch.clamp(long_raw, 0, 1)
            outputs = torch.clamp(outputs, 0, 1)
            gt_patch = torch.clamp(gt_patch, 0, 1)
            
            # print(f"outputs.max()为{outputs.max()}") # outputs.max()为0.8341951966285706
            # print(f"outputs.min()为{outputs.min()}") # outputs.min()为0.0013602226972579956
            # 将所有超出[0, 1]范围的值截断为[0, 1]
            # outputs = torch.clamp(outputs, 0, 1)
            # print(f"outputs.max()为{outputs.max()}") # outputs.max()为0.8341951966285706
            # print(f"outputs.min()为{outputs.min()}") # outputs.min()为0.0013602226972579956
            # 计算 PSNR 和 SSIM
            single_psnr =  peak_signal_noise_ratio(gt_patch, outputs)
            single_ssim =  structural_similarity_index_measure(gt_patch, outputs)
            single_lpips = get_lpips_torch(gt_patch, outputs)
            logging.info(f"PSNR: {single_psnr.item()},SSIM: {single_ssim.item()},LPIPS: {single_lpips.item()},Time: {time.time() - etime}")
            
            # 将 PSNR 和 SSIM 添加到列表中
            psnr_list.append(single_psnr.item())
            ssim_list.append(single_ssim.item())    
            lpips_list.append(single_lpips.item())
            
            outputs = outputs.cpu().detach() # torch.Size([1, 3, 2848, 4256])
            outputs = torch.squeeze(outputs)
            # print(f"outputs的大小为{outputs.shape},类型为{type(outputs)}") # outputs的大小为torch.Size([3, 2848, 4256]),类型为<class 'torch.Tensor'>
            outputs = outputs.permute(1, 2, 0)
            # print(f"outputs的大小为{outputs.shape},类型为{type(outputs)}") # outputs的大小为torch.Size([2848, 4256, 3]),类型为<class 'torch.Tensor'>
            
        

            # scaling can clipping
            outputs, scale_full, gt_full = outputs.numpy(), scale_full.numpy(), gt_full.numpy()
            
        
            
            scale_full = scale_full * np.mean(gt_full) / np.mean(
                scale_full)  # scale the low-light image to the same mean of the ground truth
            outputs = np.minimum(np.maximum(outputs, 0), 1)
            # print(f"scale_full的大小为{scale_full.shape},类型为{type(scale_full)}")
            # print(f"outputs的大小为{outputs.shape},类型为{type(outputs)}")
            # print(f"gt_full的大小为{gt_full.shape},类型为{type(gt_full)}")
            # scale_full的大小为(2848, 4256, 3),类型为<class 'numpy.ndarray'>
            # outputs的大小为(2848, 4256, 3),类型为<class 'numpy.ndarray'>
            # gt_full的大小为(2848, 4256, 3),类型为<class 'numpy.ndarray'>
            
            # scale_full的大小为(1024, 1024, 3),类型为<class 'numpy.ndarray'>
            # outputs的大小为(1024, 1024, 3),类型为<class 'numpy.ndarray'>
            # gt_full的大小为(1024, 1024, 3),类型为<class 'numpy.ndarray'>
            
            # saving
            if not os.path.isdir(os.path.join(args.result_dir, 'eval')):
                os.makedirs(os.path.join(args.result_dir, 'eval'))
            scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                os.path.join(args.result_dir, 'eval', '%05d_00_train_%d_scale.jpg' % (test_id[0], ratio[0])))
            scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(
                os.path.join(args.result_dir, 'eval', '%05d_00_train_%d_out.jpg' % (test_id[0], ratio[0])))
            scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                os.path.join(args.result_dir, 'eval', '%05d_00_train_%d_gt.jpg' % (test_id[0], ratio[0])))
        # 计算平均 PSNR 和 SSIM
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list) 
        logging.info(f"Average PSNR: {avg_psnr},Average SSIM: {avg_ssim},Average LPIPS: {avg_lpips}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    parser = argparse.ArgumentParser(description="evaluating model")
    parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--test_path', type=str, default='/mnt/data/hbj/data/Sony_test_list.txt')
    # parser.add_argument('--input_dir', type=str, default='/raid/hbj/data/Sony/short/')
    parser.add_argument('--input_dir', type=str, default='/raid/hbj/data/Sony/short_update/')
    parser.add_argument('--gt_dir', type=str, default='/raid/hbj/data/Sony/long/')
    # parser.add_argument('--test_path', type=str, default='/raid/hbj/data/Sony_test_list.txt')
    parser.add_argument('--test_path', type=str, default='/raid/hbj/data/Sony_update_test_list.txt')
    # parser.add_argument('--result_dir', type=str, default='/raid/hbj/Wave_GCC/eval_result_zpc/WaveUnet_MGCC/loss_lpips')
    parser.add_argument('--result_dir', type=str, default='/raid/hbj/Wave_GCC/compare/ours_eval_result/Loss/L1loss/best')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1, help='multi-threads for data loading')
    # parser.add_argument('--model', type=str, default='/raid/hbj/Wave_GCC/result/whole_pipline/new_dwt_RCABS/whole_stage/best_loss_model/bestmodel_0.020229027349131633_5410.pth')
    # parser.add_argument('--model', type=str, default='/raid/hbj/Wave_GCC/result/whole_pipline/new_dwt_RCABS/whole_stage_Sony/L1_loss/whole_stage/best_loss_model/bestmodel_0.020143511613992894_5990.pth')
    # parser.add_argument('--model', type=str, default='/raid/hbj/Wave_GCC/compare/SUnet++/best_loss_model/bestmodel_0.021346292646474533_5501.pth')
    parser.add_argument('--model', type=str, default='/raid/hbj/Wave_GCC/result/whole_pipline/new_dwt_RCABS/whole_stage_Sony/L1_loss/whole_stage/last_model/ModelSnapshot_0.022057889390921927_6000.pth')
    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    test(args)