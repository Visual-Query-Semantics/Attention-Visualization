from visualizer import get_local
get_local.activate()
from visualizer.display import visualize_grid_to_grid_with_cls, \
                               visualize_grid_to_grid 

import torch
import torchvision.transforms as transforms

from PIL import Image

import models_mae



MAE_CKPT_DIR ='./pretrained_ckpts/mae/mae_pretrain_vit_large.pth'



def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model



def main():
    chkpt_dir = MAE_CKPT_DIR
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')
    
    image1 = Image.open('./assets/dogcat.jpg')
    image2 = Image.open('./assets/rabbit.jpg')
    
    #plt.axis('off')
    #plt.imshow(image)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    _transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    input_tensor1 = _transforms(image1).unsqueeze(0)
    input_tensor2 = _transforms(image2).unsqueeze(0)
    
    print(input_tensor1.shape)
    
    get_local.clear()
    with torch.no_grad():
        model_mae(input_tensor1, mask_ratio=0)
    
    cache = get_local.cache
    print(list(cache.keys()))
    
    attention_maps = cache['Attention.forward']
    
    print(len(attention_maps))
    
    print(attention_maps[0].shape)
    
    #visualize_grid_to_grid_with_cls(attention_maps[11][0,4,:,:], 105, image1)
    
    #visualize_grid_to_grid(attention_maps[11][0,11,1:,1:], 87, image1)
    
    visualize_grid_to_grid(attention_maps[23][0,6,1:,1:], 87, image1)
    
    
if __name__ == '__main__':
    main()    