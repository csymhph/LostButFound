import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.unet import Unet

def ifft2(img, norm='ortho'):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img), norm=norm))

def fft2(img, norm='ortho'):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img), norm=norm))

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        input, target, maximum, _, _ = data
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
   
        output=model(input)
    
        # Mask output and target, when epoch exceeds 30
        #if epoch>30:
         #   loss = loss_type(output, target, maximum, masked=True)
        #else:

        loss = loss_type(output, target, maximum, masked=False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model_list, data_loader_list):
    
    reconstructions = []
    targets = []
    inputs = []
    start = time.perf_counter()
    
    for iter, model in enumerate(model_list):
        model.eval()
        reconstructions.append(defaultdict(dict))
        targets.append(defaultdict(dict))
        inputs.append(defaultdict(dict))

        with torch.no_grad():
            for i, data in enumerate(data_loader_list[iter]):
                input, target, _, fnames, slices = data
                input = input.cuda(non_blocking=True)
                output = model(input)

                for j in range(output.shape[0]):
                    reconstructions[iter][fnames[j]][int(slices[j])] = output[j].cpu().numpy()
                    targets[iter][fnames[j]][int(slices[j])] = target[j].numpy()
                    inputs[iter][fnames[j]][int(slices[j])] = input[j].cpu().numpy()

        for fname in reconstructions[iter]:
            reconstructions[iter][fname] = np.stack(
                [out for _, out in sorted(reconstructions[iter][fname].items())]
            )
        for fname in targets[iter]:
            targets[iter][fname] = np.stack(
                [out for _, out in sorted(targets[iter][fname].items())]
            )
        for fname in inputs[iter]:
            inputs[iter][fname] = np.stack(
                [out for _, out in sorted(inputs[iter][fname].items())]
            )
    
    combined_recon = defaultdict(dict)
    combined_target = defaultdict(dict)
    combined_input = defaultdict(dict)
   
    for fname in reconstructions[0]:
        for slices in range(reconstructions[0][fname].shape[0]):
            combined_recon[fname][slices] = np.sqrt(np.abs(reconstructions[0][fname][slices] ** 2 + reconstructions[1][fname][slices] ** 2 + reconstructions[2][fname][slices] ** 2 + reconstructions[3][fname][slices] ** 2 + 2 * reconstructions[4][fname][slices] - 2 * reconstructions[5][fname][slices]))
            
        combined_recon[fname] = np.stack(
            [out for _, out in sorted(combined_recon[fname].items())]
        )

    for fname in targets[0]:
        for slices in range(reconstructions[0][fname].shape[0]):
            combined_target[fname][slices] = np.sqrt(np.abs(targets[0][fname][slices] ** 2 + targets[1][fname][slices] ** 2 + targets[2][fname][slices] ** 2 + targets[3][fname][slices] ** 2 + 2 * targets[4][fname][slices] - 2 * targets[5][fname][slices]))
        
        combined_target[fname] = np.stack(
            [out for _, out in sorted(combined_target[fname].items())]
        )
        
    for fname in inputs[0]:
        for slices in range(reconstructions[0][fname].shape[0]):
            combined_input[fname][slices] = np.sqrt(np.abs(inputs[0][fname][slices] ** 2 + inputs[1][fname][slices] ** 2 + inputs[2][fname][slices] ** 2 + inputs[3][fname][slices] ** 2 + 2 * inputs[4][fname][slices] - 2 * inputs[5][fname][slices]))
            
        combined_input[fname] = np.stack(
            [out for _, out in sorted(combined_input[fname].items())]
        )
        metric_loss = sum([ssim_loss(combined_target[fname], combined_recon[fname]) for fname in combined_recon])
    num_subjects = len(combined_recon)
    
    return metric_loss, num_subjects, combined_recon, combined_target, combined_input, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model_list, optimizer_list, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model_1': model_list[0].state_dict(),
            'model_2': model_list[1].state_dict(),
            'model_3': model_list[2].state_dict(),
            'model_4': model_list[3].state_dict(),
            'model_5': model_list[3].state_dict(),
            'model_6': model_list[3].state_dict(),
            'optimizer_1': optimizer_list[0].state_dict(),
            'optimizer_2': optimizer_list[1].state_dict(),
            'optimizer_3': optimizer_list[2].state_dict(),
            'optimizer_4': optimizer_list[3].state_dict(),
            'optimizer_5': optimizer_list[3].state_dict(),
            'optimizer_6': optimizer_list[3].state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    #model = Unet(in_chans = args.in_chans, out_chans = args.out_chans, drop_prob = args.drop_prob)
    model_1 = Unet(in_chans = args.in_chans, out_chans = args.out_chans, drop_prob = args.drop_prob)
    model_2 = Unet(in_chans = args.in_chans, out_chans = args.out_chans, drop_prob = args.drop_prob)
    model_3 = Unet(in_chans = args.in_chans, out_chans = args.out_chans, drop_prob = args.drop_prob)
    model_4 = Unet(in_chans = args.in_chans, out_chans = args.out_chans, drop_prob = args.drop_prob)
    model_5 = Unet(in_chans = args.in_chans, out_chans = args.out_chans, drop_prob = args.drop_prob)
    model_6 = Unet(in_chans = args.in_chans, out_chans = args.out_chans, drop_prob = args.drop_prob)
    
    
    model_1.to(device=device)
    model_2.to(device=device)
    model_3.to(device=device)
    model_4.to(device=device)
    model_5.to(device=device)
    model_6.to(device=device)
    
    model_list = [model_1, model_2, model_3, model_4, model_5, model_6]
    
    loss_type = SSIMLoss().to(device=device)
    #optimizer = torch.optim.NAdam(model.parameters(), args.lr)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda epoch: 0.95 ** epoch )
    
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)

    best_val_loss = 1.
    start_epoch = 0

    train_loader_list = [create_data_loaders(data_path_1 = '/root/Datastorage_train/Real/Real/', data_path_2 = 'init', args = args, shuffle=True),
                        create_data_loaders(data_path_1 = '/root/Datastorage_train/Real/Imaginary/', data_path_2 = 'init', args = args, shuffle=True),
                        create_data_loaders(data_path_1 = '/root/Datastorage_train/Imaginary/Real/', data_path_2 = 'init', args = args, shuffle=True),
                        create_data_loaders(data_path_1 = '/root/Datastorage_train/Imaginary/Imaginary/', data_path_2 = 'init', args = args, shuffle=True),
                        create_data_loaders(data_path_1 = '/root/Datastorage_train/Comp_1/', data_path_2 = 'init', args = args, shuffle=True),
                        create_data_loaders(data_path_1 = '/root/Datastorage_train/Comp_2/', data_path_2 = 'init', args = args, shuffle=True)
        ]
#     train_loader_list = [create_data_loaders(data_path_1 = '/root/Datastorage_check/Real/Real/', data_path_2 = 'init', args = args, shuffle=True),
#                         create_data_loaders(data_path_1 = '/root/Datastorage_check/Real/Imaginary/', data_path_2 = 'init', args = args, shuffle=True),
#                         create_data_loaders(data_path_1 = '/root/Datastorage_check/Imaginary/Real/', data_path_2 = 'init', args = args, shuffle=True),
#                         create_data_loaders(data_path_1 = '/root/Datastorage_check/Imaginary/Imaginary/', data_path_2 = 'init', args = args, shuffle=True),
#                         create_data_loaders(data_path_1 = '/root/Datastorage_check/Comp_1/', data_path_2 = 'init', args = args, shuffle=True),
#                         create_data_loaders(data_path_1 = '/root/Datastorage_check/Comp_2/', data_path_2 = 'init', args = args, shuffle=True)
#         ]
    
    
    
    val_loader_list = [
        create_data_loaders(data_path_1 = '/root/Datastorage_val/Real/Real/', data_path_2 = 'init', args = args), 
        create_data_loaders(data_path_1 = '/root/Datastorage_val/Real/Imaginary/', data_path_2 = 'init', args = args), 
        create_data_loaders(data_path_1 = '/root/Datastorage_val/Imaginary/Real/', data_path_2 = 'init', args = args), 
        create_data_loaders(data_path_1 = '/root/Datastorage_val/Imaginary/Imaginary/', data_path_2 = 'init', args = args),
        create_data_loaders(data_path_1 = '/root/Datastorage_val/Comp_1/', data_path_2 = 'init', args = args),
        create_data_loaders(data_path_1 = '/root/Datastorage_val/Comp_2/', data_path_2 = 'init', args = args)
        ]
#     val_loader_list = [
#         create_data_loaders(data_path_1 = '/root/Datastorage_check/Real/Real/', data_path_2 = 'init', args = args), 
#         create_data_loaders(data_path_1 = '/root/Datastorage_check/Real/Imaginary/', data_path_2 = 'init', args = args), 
#         create_data_loaders(data_path_1 = '/root/Datastorage_check/Imaginary/Real/', data_path_2 = 'init', args = args), 
#         create_data_loaders(data_path_1 = '/root/Datastorage_check/Imaginary/Imaginary/', data_path_2 = 'init', args = args),
#         create_data_loaders(data_path_1 = '/root/Datastorage_check/Comp_1/', data_path_2 = 'init', args = args),
#         create_data_loaders(data_path_1 = '/root/Datastorage_check/Comp_2/', data_path_2 = 'init', args = args)
#         ]




    optimizer_1 = torch.optim.NAdam(model_1.parameters(), args.lr)
    optimizer_2 = torch.optim.NAdam(model_2.parameters(), args.lr)
    optimizer_3 = torch.optim.NAdam(model_3.parameters(), args.lr)
    optimizer_4 = torch.optim.NAdam(model_4.parameters(), args.lr)
    optimizer_5 = torch.optim.NAdam(model_4.parameters(), args.lr)
    optimizer_6 = torch.optim.NAdam(model_4.parameters(), args.lr)
    optimizer_list = [optimizer_1, optimizer_2, optimizer_3, optimizer_4, optimizer_5, optimizer_6]
  
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[0], T_max=5, eta_min=0)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[1], T_max=5, eta_min=0)
    scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[2], T_max=5, eta_min=0)
    scheduler_4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[3], T_max=5, eta_min=0)
    scheduler_5 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[4], T_max=5, eta_min=0)
    scheduler_6 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[5], T_max=5, eta_min=0)
    scheduler_list = [scheduler_1, scheduler_2, scheduler_3, scheduler_4, scheduler_5, scheduler_6]
       
    
    '''
    일단 val_loader_i 를 통해 각각의 인풋을 가져오고, 각각 해당하는 model을 통과시킨 뒤 나오는 아웃풋을 (acc, 환자번호)가 같은 것들끼리 합성시켜서 최종      reconstructions에 집어넣는다. 마찬가지로 label도 그렇게 합성시킨 뒤 둘을 비교하여 .ssim metric을 만든다.
    
    즉, 위의 data_path_train, data_path_val 모두 각각의 real/imaginary 파일에만 접근하도록 설정하여야 함.
    '''
    train_loss_list = []
    val_loss_list = []

    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss_1, train_time_1 = train_epoch(args, epoch, model_1, train_loader_list[0], optimizer_list[0], loss_type)
        train_loss_2, train_time_2 = train_epoch(args, epoch, model_2, train_loader_list[1], optimizer_list[1], loss_type)
        train_loss_3, train_time_3 = train_epoch(args, epoch, model_3, train_loader_list[2], optimizer_list[2], loss_type)
        train_loss_4, train_time_4 = train_epoch(args, epoch, model_4, train_loader_list[3], optimizer_list[3], loss_type)
        train_loss_5, train_time_5 = train_epoch(args, epoch, model_4, train_loader_list[4], optimizer_list[4], loss_type)
        train_loss_6, train_time_6 = train_epoch(args, epoch, model_4, train_loader_list[5], optimizer_list[5], loss_type)
        
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model_list=model_list,data_loader_list=val_loader_list)
        
        scheduler_1.step()
        scheduler_2.step()
        scheduler_3.step()
        scheduler_4.step()
        scheduler_5.step()
        scheduler_6.step()

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = args.val_loss_dir / "val_loss_log"
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")
        
        train_loss = (train_loss_1 + train_loss_2 + train_loss_3 + train_loss_4 + train_loss_5 + train_loss_6 ) / len(model_list)
        train_time = (train_time_1 + train_time_2 + train_time_3 + train_time_4 + train_time_5 + train_time_6 )
        
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model_list, optimizer_list, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
    
    y1 = np.array([])
    y2 = np.array([])
    
    for i in train_loss_list:
        i = i.to('cpu')
        y1 = np.append(y1, i)
    
    for i in val_loss_list:
        i = i.to('cpu')
        y2 = np.append(y2, i)
    
    x = np.arange(0, args.num_epochs)
    
    plt.plot(x, y1, 'r-.', label = 'train')
    plt.plot(x, y2, 'b-.', label = 'val')
    plt.legend()
     
    plt.title('Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    
    plt.savefig('loss1.png', dpi=300)
