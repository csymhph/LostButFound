import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_kspace_loaders
from utils.model.unet import Unet
from utils.common import fftc as ff

def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    
    return torch.sqrt((data ** 2).sum(dim))

def test(args, model_list, data_loader):
    
    reconstructions = []
    inputs = []
    
    
    '''
    Things to do from now:
    
    import kspace data from data_loader, and process 6 images.
    coilwise RSS calculation involved
    
    After generating 6 input images, put them through trained model.
    This returns 6 aliasing-free output images.
    
    Combine 6 output images, and return final output as reconstructions
    '''
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            real = kspace[:,:,:,:,0]
            imag = kspace[:,:,:,:,1]
            
            for slice in kspace.shape[0]
                image1 = rss(torch.real(ff.ifft2c(real[slice])))
                image2 = rss(torch.imag(ff.ifft2c(real[slice])))
                image3 = rss(torch.real(ff.ifft2c(imag[slice])))
                image4 = rss(torch.imag(ff.ifft2c(imag[slice])))
                image5 = torch.sum(torch.real(ff.ifft2c(imag[slice]))*torch.imag(ff.ifft2c(real[slice])), dim=0)
                image5 = image5 * (image5>0)
                image6 = torch.sum(torch.imag(ff.ifft2c(imag[slice]))*torch.real(ff.ifft2c(real[slice])), dim=0)
                image6 = image6 * (image6>0)


            for iter, model in model_list:
                model.eval()
                
                if iter ==0:
                    output=model(image1)
                    
                if iter ==1:
                    output=model(image2)
                    
                if iter ==2:
                    output=model(image3)
                    
                if iter ==3:
                    output=model(image4)
                    
                if iter ==4:
                    output=model(image5)
                    
                if iter ==5:
                    output=model(image6)



                for i in range(output.shape[0]):
                    reconstructions[iter][fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                    inputs[iter][fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
    return reconstructions, inputs


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model_1 = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model_2 = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model_3 = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model_4 = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model_5 = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model_6 = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model_1.load_state_dict(checkpoint['model_1'])
    model_2.load_state_dict(checkpoint['model_2'])
    model_3.load_state_dict(checkpoint['model_3'])
    model_4.load_state_dict(checkpoint['model_4'])
    model_5.load_state_dict(checkpoint['model_5'])
    model_6.load_state_dict(checkpoint['model_6'])
    
    model_list = [
        model_1, model_2, model_3, model_4, model_5, model_6
    ]
    
    forward_loader = create_kspace_loaders(data_path_1 = args.data_path, data_path_2 = 'init', args = args, isforward = True)
    
    reconstructions, inputs = test(args, model_list, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
