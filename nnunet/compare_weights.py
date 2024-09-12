import torch

pretrained_weights = '/home/GRAMES.POLYMTL.CA/p118739/data/nnUNet_results/Dataset146_nako_manual_inference_plus_spider_143/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
current_weights = '/home/GRAMES.POLYMTL.CA/p118739/data/nnUNet_results/Dataset348_DiscsVertebrae/nnUNetTrainerFT__new_plans__3d_fullres/fold_0/checkpoint_final.pth'

pretrained_model = torch.load(pretrained_weights, map_location=torch.device('cpu'))['network_weights']
current_model = torch.load(current_weights, map_location=torch.device('cpu'))['network_weights']

for key, _ in pretrained_model.items():
    layer_pretrained = pretrained_model[key]
    layer_current = current_model[key]
    print(key)
    print((layer_pretrained == layer_current).all())
        


