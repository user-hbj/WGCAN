import argparse
from ast import arg
import datetime
import imp
import os
from telnetlib import PRAGMA_HEARTBEAT
import numpy as np
import time
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from datasets.dataset import *
import os, time, scipy.io, scipy.misc
# import visdom
import rawpy
import scipy
import glob
from PIL import Image
import matplotlib.pyplot as plt
# import skimage.metricss
import logging
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from utils.util import *
from models.new_dwt_second_model import *
from models.whole_new_dwt_RCABS import *
from collections import OrderedDict
from Losses.wavelet_loss import CombinedLoss

def main(args):
    
    result_dir = args.result_dir
  
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    save_val_model = os.path.join(result_dir, 'val_model')
    if not os.path.exists(save_val_model):
        os.makedirs(save_val_model)
    
    save_bestmodel = os.path.join(result_dir, 'best_loss_model')
    if not os.path.exists(save_bestmodel):
        os.makedirs(save_bestmodel)

    save_lastmodel = os.path.join(result_dir, 'last_model')
    if not os.path.exists(save_lastmodel):
        os.makedirs(save_lastmodel)
    
    save_best_psnr_model = os.path.join(result_dir, 'best_psnr_model')
    if not os.path.exists(save_best_psnr_model):
        os.makedirs(save_best_psnr_model)

    save_best_ssim_model = os.path.join(result_dir, 'best_ssim_model')
    if not os.path.exists(save_best_ssim_model):
        os.makedirs(save_best_ssim_model)

    logs_folder = os.path.join(result_dir, 'logs') 
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    
    # 设置日志记录
    logging.basicConfig(filename=os.path.join(logs_folder, 'train.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

    trainset = SonySLDataset(args.input_dir, args.gt_dir, args.ps)
    valset = SonySLEvalDataset(args.val_list, args.ps)
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MGCC().to(device)
    criterion_wave = CombinedLoss().cuda()
    criterion = torch.nn.L1Loss().cuda()
   
    if args.resume:   # 如果为True，则表示接着训练；如果为False，则表示从头开始训练
        last_info_list = process_files(save_lastmodel)[0] # 获取列表中的第一个字典
        model_name = last_info_list['文件名']
        model_loss = float(last_info_list['Loss'])
        model_epoch = int(last_info_list['epoch'])  # 将字符串转换为整数
        model_path = os.path.join(save_lastmodel, model_name)    
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)  # 确保模型在GPU上
        lastepoch = model_epoch + 1
        print(f"从{lastepoch}轮继续训练")
        best_info_list = process_files(save_bestmodel)[0]
        min_loss = float(best_info_list['Loss'])

        psnr_info_list = process_files(save_best_psnr_model)[0]
        max_psnr = float(psnr_info_list['Loss'])

        ssim_info_list = process_files(save_best_ssim_model)[0]
        max_ssim = float(ssim_info_list['Loss'])

        if model_epoch < 100:
            val_min_loss = float('inf')  # 初始化最小损失为正无穷

        else:
            eval_info_list = process_files(save_val_model)[0]
            val_min_loss = float(eval_info_list['Loss'])

    else:
        lastepoch = 1
        min_loss = float('inf')
        val_min_loss = float('inf')  # 初始化最小损失为正无穷
        max_psnr = 0
        max_ssim = 0

        
    print(f"lastepoch为 {lastepoch}")
     # 初始化一个SummaryWriter对象，将日志写入指定的目录
    log_dir = os.path.join(result_dir, 'logs_tensorboard')
    # writer = SummaryWriter(log_dir)
    #  加载之前保存的日志文件夹
    writer = SummaryWriter(log_dir, purge_step=lastepoch)

    G_opt = optim.Adam(model.parameters(), lr=args.lr)
    # lr scheduler
    scheduler = optim.lr_scheduler.StepLR(G_opt, step_size=2000, gamma=0.1)


        
    
    
    for epoch in range(lastepoch,args.num_epoch):
        model.train()
        # 如果当前时期的结果目录已经存在，跳过当前时期的训练，继续下一个时期的训练。
        if os.path.isdir(result_dir + '%04d' % epoch):
            continue
        #Calculating total loss
        etime = time.time()
        eloss = 0
        epsnr = 0
        essim = 0
        
        count = 0
        # 随机排列训练数据的索引，以确保随机性 开始遍历训练数据，对每一张图像进行训练
        for i, databatch in enumerate(train_loader):
            input_patch, Long_patch, gt_patch, id, ratio = databatch
            
            batch_length = input_patch.size(0)
            # print(batch_length)
            count += batch_length
            print(count)
           
            input_patch = input_patch.cuda()
            gt_patch = gt_patch.cuda()
            
    
            outputs = model(input_patch)
            
            # 将像素值限制在[0, 1]范围内
            outputs = torch.clamp(outputs, 0, 1)
            
            # 检查张量的最小值和最大值
            min_val = torch.min(outputs)
            max_val = torch.max(outputs)
            # print(f"outputs最小值为{min_val},最大值为{max_val}")
            
            # 检查张量的最小值和最大值
            min_val = torch.min(Long_patch)
            max_val = torch.max(Long_patch)
            
            # print(f"Long_patch最小值为{min_val},最大值为{max_val}")
            loss = criterion_wave(outputs, gt_patch)
            # print(f"小波损失的值为{loss_wave}")
            # loss_L1 = criterion(outputs, gt_patch)
            # print(f"L1损失的值为{loss_L1}")
            
            # loss = alpha * loss_L1 + beta * loss_wave
            # print(f"Loss损失的值为{loss}")
            
            G_opt.zero_grad()
            loss.backward()
            G_opt.step()
            
            eloss = eloss + loss.item()   # Total Loss
            
            # 计算 PSNR 和 SSIM
            single_psnr =  peak_signal_noise_ratio(gt_patch, outputs)
            single_ssim =  structural_similarity_index_measure(gt_patch, outputs)
            
            epsnr = epsnr + single_psnr.item()  # Total psnr
            essim = essim + single_ssim.item()  # Total SSIM


            # if epoch % args.save_freq == 0:
            #     if not os.path.isdir(os.path.join(result_dir, '%04d' % epoch)):
            #         os.makedirs(os.path.join(result_dir, '%04d' % epoch))
                
            #     gt_patch = gt_patch.detach().cpu().numpy()
            #     outputs = outputs.detach().cpu().numpy()
            #     train_id = train_id.item()
            #     ratio = ratio.item()
               
            #     print(f"train_id的值为{train_id}")
            #     print(f"ratio的值为{ratio}")
                
            #     # print(len(gt_patch))
            #     for i in range(len(gt_patch)):
            #         # 转换为 uint8 类型，并移动通道维度到最后一个维度
            #         temp_gt = np.transpose((gt_patch[i] * 255).clip(0, 255).astype(np.uint8), (1, 2, 0))
            #         temp_out = np.transpose((outputs[i] * 255).clip(0, 255).astype(np.uint8), (1, 2, 0))

            #         # 合并原始图像和输出图像
            #         temp = np.concatenate((temp_gt, temp_out), axis=1)
                     
            #         # 保存图像
            #         filename = '%05d_00_train_%d.jpg' % (train_id, ratio)  # 修改保存文件名格式
            #         Image.fromarray(temp).save(os.path.join(result_dir, '%04d' % epoch, filename))

        # 更新学习率
        scheduler.step()    
        
        # Saving Snapshot of Model with different name for each 100 epoch  
        # if np.mod(epoch, args.model_save_freq) == 0:
        #     ModelName = result_dir + "ModelSnapshot_"+str(epoch)+"_epoch.pth"
        #     torch.save(model.state_dict(), ModelName)
        #     print("模型已保存，当前是第{}个epoch".format(epoch))
        
        # esize = len(train_ids)
        print(f"训练一轮总的个数为{count}")
        aloss = eloss/count
        apsnr = epsnr/count
        assim = essim/count
        temp_loss = aloss
        temp_psnr = apsnr
        temp_ssim = assim
        
        # 写入TensorBoard
        writer.add_scalar('train_loss', aloss, epoch)
        writer.add_scalar('train_PSNR', apsnr, epoch)
        writer.add_scalar('train_SSIM', assim, epoch)
             
        if(epoch % 100 == 0):
            model.eval()
            
            v_loss = 0
            v_psnr = 0
            v_ssim = 0
            v_count = 0
            with torch.no_grad():  # 在验证过程中不需要计算梯度
                # 随机排列训练数据的索引，以确保随机性 开始遍历训练数据，对每一张图像进行训练
                for i, databatch in enumerate(val_loader):
                    input_patch, Long_patch, gt_patch, ratio = databatch
                
                    batch_length = input_patch.size(0)
                    # print(batch_length)
                    v_count += batch_length
                    print(v_count)
                
                    input_patch = input_patch.cuda()
                    Long_patch = Long_patch.cuda()
                    gt_patch = gt_patch.cuda()
                    
                    
                    # 将 input_patch 传递给第一阶段模型
                    # first_stage_output = first_stage_model(input_patch)
                    
                    outputs = model(input_patch)
                    
                    # 将像素值限制在[0, 1]范围内
                    outputs = torch.clamp(outputs, 0, 1)
                    
                    loss = criterion_wave(outputs, gt_patch)
                    # loss_L1 = criterion(outputs, gt_patch)
                    
                    # loss = alpha * loss_L1 + beta * loss_wave
                    v_loss = v_loss + loss.item()  # Total Loss
                    # print(f"Long_patch的大小为{Long_patch.shape},类型为{type(Long_patch)}")
                    # print(f"outputs的大小{outputs.shape},类型为{type(outputs)}")
                    
                    # 计算 PSNR 和 SSIM
                    single_psnr =  peak_signal_noise_ratio(outputs, gt_patch)
                    single_ssim =  structural_similarity_index_measure(outputs, gt_patch)
                    
                    v_psnr = v_psnr + single_psnr.item() # Total psnr
                    v_ssim = v_ssim + single_ssim.item()  # Total SSIM
                    
            val_loss = v_loss/v_count
            val_psnr = v_psnr/v_count
            val_ssim = v_ssim/v_count
            val_temp_loss = val_loss
            print(f"验证一轮总的个数为{v_count}")
            
            # 更新最小损失和最大PSNR
            if val_temp_loss < val_min_loss:
                val_min_loss = val_temp_loss
                # 保存最好的权重
                if os.path.exists(save_val_model):
                    # 先删除再进行保存
                    for filename in os.listdir(save_val_model):
                        if filename.endswith('.pth'):
                            # 删除该文件
                            file_path = os.path.join(save_val_model, filename)
                            os.remove(file_path)
                    val_bestmodel = os.path.join(save_val_model, "valbestmodel_{}_{}.pth".format(temp_loss, epoch))
                    torch.save(model.state_dict(), val_bestmodel) 
                      
        if temp_loss < min_loss:
            min_loss = temp_loss
            min_loss_epoch = epoch
            # 保存最好的权重
            if os.path.exists(save_bestmodel):
                # 先删除再进行保存
                for filename in os.listdir(save_bestmodel):
                    if filename.endswith('.pth'):
                        # 删除该文件
                        file_path = os.path.join(save_bestmodel, filename)
                        os.remove(file_path)
                bestmodel = os.path.join(save_bestmodel, "bestmodel_{}_{}.pth".format(temp_loss, epoch))
                torch.save(model.state_dict(), bestmodel)   
                  
        if temp_psnr > max_psnr:
            max_psnr = temp_psnr
            # 保存最好的权重
            if os.path.exists(save_best_psnr_model):
                # 先删除再进行保存
                for filename in os.listdir(save_best_psnr_model):
                    if filename.endswith('.pth'):
                        # 删除该文件
                        file_path = os.path.join(save_best_psnr_model, filename)
                        os.remove(file_path)
                bestpsnrmodel = os.path.join(save_best_psnr_model, "bestpsnrmodel_{}_{}.pth".format(temp_psnr, epoch))
                torch.save(model.state_dict(), bestpsnrmodel) 
        
        if temp_ssim > max_ssim:
            max_ssim = temp_ssim
            # 保存最好的权重
            if os.path.exists(save_best_ssim_model):
                # 先删除再进行保存
                for filename in os.listdir(save_best_ssim_model):
                    if filename.endswith('.pth'):
                        # 删除该文件
                        file_path = os.path.join(save_best_ssim_model, filename)
                        os.remove(file_path)
                bestssimmodel = os.path.join(save_best_ssim_model, "bestssimmodel_{}_{}.pth".format(temp_ssim, epoch))
                torch.save(model.state_dict(), bestssimmodel) 
        
        print(f"\nEpoch = {epoch}. \tLoss = {aloss}, \tPSNR = {apsnr}, \tSSIM = {assim},\tTime = {time.time() - etime}")
        # 在时期完成后
        logging.info(f"Epoch {epoch} - Loss: {aloss}, PSNR: {apsnr},SSIM: {assim},Time: {time.time() - etime}")
        
        # 保存最后一轮的权重
        # 检查目录中是否有.pth结尾的文件
        if os.path.exists(save_lastmodel):
            # 先删除再进行保存
            for filename in os.listdir(save_lastmodel):
                if filename.endswith('.pth'):
                    # 删除该文件
                    file_path = os.path.join(save_lastmodel, filename)
                    os.remove(file_path)
            lastmodel = os.path.join(save_lastmodel, "ModelSnapshot_{}_{}.pth".format(temp_loss, epoch))
            torch.save(model.state_dict(), lastmodel)

    # 在训练结束后输出最小损失和最大PSNR
    print(f"Min Loss: {min_loss} (Epoch {min_loss_epoch}), Max PSNR: {max_psnr},Max SSIM: {max_ssim} ")
    # 在整个训练过程结束时
    logging.info(f"Training complete. Min Loss: {min_loss} (Epoch {min_loss_epoch}), Max PSNR: {max_psnr} ,Max SSIM: {max_ssim}")
    writer.close()        

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/raid/hbj/data/Sony/short/')
    parser.add_argument('--gt_dir', type=str, default='/raid/hbj/data/Sony/long/')
    parser.add_argument('--val_list', type=str, default='/raid/hbj/data/Sony_val_list.txt')
    parser.add_argument('--result_dir', type=str, default='/raid/hbj/Wave_GCC/result/whole_pipline/new_dwt_RCABS/whole_stage_Sony/WaveLoss')
    
    parser.add_argument('--ps', type=int, default=512)
    # parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=6001)
    parser.add_argument('--model_save_freq', type=int, default=200)
    parser.add_argument('--resume', type=bool, default=True, help='continue training')
    parser.add_argument('--workers', type=int, default=1)
    # parser.add_argument('--model', type=str, default='/raid/hbj/Wave_GCC/result/wave_net/first_stage/new_dwt_conv/best_model/bestmodel_0.022891917407674635_3752.pth')
    args = parser.parse_args()        
    main(args)
    