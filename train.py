from __future__ import print_function
import os
from math import log10
from collections import OrderedDict

import torchvision.utils as vutils

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader

from dataset import DatasetFromFolder
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
from pix2pix_pro import Pix2PixPro
import util.util as util
from util.visualizer import Visualizer
import time

from options.train_options import TrainOptions, EPOCH_MAPPING, MINIBATCH_MAPPING

opt = TrainOptions().parse()

import random
random.seed(2143155159)

if __name__ == '__main__':

    cudnn.benchmark = True
    print("GAN_MODE: {}".format(opt.gan_mode))

    visualizer = Visualizer(opt)

    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    ## creating checkpoint dir
    if not os.path.exists("result"):
            os.mkdir("result")
    if not os.path.exists(os.path.join("result", opt.project_name)):
        os.mkdir(os.path.join("result", opt.project_name))

    print('===> Loading datasets')
    root_path = opt.dataset_root


    pix2pix_pro = Pix2PixPro()
    pix2pix_pro.initialize(opt)

    pix2pix_pro = nn.DataParallel(pix2pix_pro, device_ids=opt.gpu_ids)
    pix2pix_pro.cuda()

    current_resolution = opt.start_resolution

    print("===> start epoch:{} start resolution:{}".format(opt.start_epoch, opt.start_resolution))

    for epoch in range(opt.start_epoch, opt.niter + opt.niter_decay + 1):
        random.seed(2143155159 * (epoch+1))

        net_status = "stable"
        if  epoch == 1 and current_resolution == 256:
            net_status = "stable"
        elif epoch > EPOCH_MAPPING[current_resolution]:
            if current_resolution != 256:
                current_resolution *= 2
                net_status = "fadein"
                print("==> fadeIn! current_resolution:{}".format(current_resolution))
                psnr_cache = 0

        else:
            net_status = "stable"

        pix2pix_pro.module.set_config(current_resolution, net_status, 1.0)
        batch_size = MINIBATCH_MAPPING[current_resolution] * opt.batch_rate
        if net_status == "fadein":
            batch_size = int(batch_size/2)
            if(batch_size < 1):
                batch_size = 1

        train_set = DatasetFromFolder(root_path + opt.dataset + "/train", opt.direction, current_resolution, is_train=True)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=batch_size,
                                          shuffle=True)

        test_set = DatasetFromFolder(root_path + opt.dataset + "/test", opt.direction, current_resolution, is_train=False)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=batch_size,
                                         shuffle=False)

        # train
        for iteration, batch in enumerate(training_data_loader, 1):
            if iteration % 10 == 0:
                iter_start_time = time.time()
            if net_status == "fadein":
                net_alpha = 1.0 - (iteration + 1) / len(train_set)
                pix2pix_pro.module.set_config(current_resolution, "fadein", net_alpha)

            # forward
            real_a, real_b, real_256_a, other_a = f.interpolate(batch[2], current_resolution).cuda(), batch[1].cuda(), batch[2].cuda(), batch[6].cuda()

            losses, generated = pix2pix_pro(real_256_a, real_a, real_b, other_a)
            losses = {k: v.mean() if not isinstance(v, int) else v for k, v in losses.items()}

            loss_d = ( losses["D_fake"] + losses["D_real"] + losses["D_other"] ) / 3
            loss_d_tf = ( losses["D_tf_fake"] + losses["D_tf_real"] ) / 2

            loss_g = losses["G_GAN"] + losses["G_GAN_tf"] + losses["G_FM"] + losses.get("G_FM_tf", 0) + losses.get("G_Input", 0) + losses["G_L1"]


            pix2pix_pro.module.optimizer_g.zero_grad()
            loss_g.backward()
            pix2pix_pro.module.optimizer_g.step()

            pix2pix_pro.module.optimizer_d.zero_grad()
            loss_d.backward()
            pix2pix_pro.module.optimizer_d.step()

            pix2pix_pro.module.optimizer_d_tf.zero_grad()
            loss_d_tf.backward()
            pix2pix_pro.module.optimizer_d_tf.step()

            ############## Display results and errors ##########
            ### print out errors
            if iteration % 10 == 0:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in losses.items()}
                t = (time.time() - iter_start_time) / 10
                visualizer.print_current_errors(epoch, iteration, errors, t)
                total_steps = epoch * len(training_data_loader) + iteration
                visualizer.plot_current_errors(errors, total_steps)

                visuals = OrderedDict([('input_label', util.tensor2im(real_a.data[0])),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(real_b.data[0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

        pix2pix_pro.module.update_learning_rate()
        pix2pix_pro.module.save("latest")

        # test
        avg_psnr = 0
        for iteration,batch in enumerate(testing_data_loader):
            real_a, real_b, real_256_a, real_256_b = \
                batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()#, batch[4].to(device)#, batch[5].to(device)#, batch[6].to(device)

            prediction = pix2pix_pro.module.inference(real_256_a.detach())

            mse = np.mean((prediction.cpu().numpy() - real_b.cpu().numpy()) ** 2)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr

            vutils.save_image(torch.cat((real_256_a, f.interpolate(prediction.float(), 256), real_256_b), 3),
                              "{}/{}/test_result_epoch_{}_iter_{}".format(opt.checkpoints_dir, opt.project_name, str(epoch), str(iteration)) + ".jpg",
                              nrow=4, normalize=True, padding=0)
            del prediction, real_a, real_b, real_256_a, real_256_b
        result_img = Image.new('RGB', (0, 0), (0, 0, 0))
        for p in range(99):
            try:
                img_name = "{}/{}/test_result_epoch_{}_iter_{}".format(opt.checkpoints_dir, opt.project_name, str(epoch), str(p)) + ".jpg"
                im = Image.open(img_name)
                result_img = util.get_concat_v_blank(result_img, im)
                os.remove(img_name)
            except:
                break
        result_img.save("{}/{}/test_result_epoch_{}".format(opt.checkpoints_dir, opt.project_name, str(epoch)) + ".jpg")
        
        avg_psnr = avg_psnr / len(testing_data_loader)
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))

        if epoch % 10 == 0 and epoch != 0:
            pix2pix_pro.module.save(epoch)