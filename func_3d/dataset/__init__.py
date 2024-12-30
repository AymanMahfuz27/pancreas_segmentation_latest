from .btcv import BTCV
from .amos import AMOS
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .pancreas import PancreasDataset




def get_dataloader(args):
    # transform_train = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_train_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'pancreas':
        # e.g. patch_size could be read from args or fixed
        patch_z = 32
        patch_y = 256
        patch_x = 256
        pancreas_train_dataset = PancreasDataset(
            args,
            args.data_path,
            mode='Training',
            prompt=args.prompt,
            patch_size=(patch_z, patch_y, patch_x),
            margin=10,
            do_augmentation=True  # augment only on train
        )
        pancreas_test_dataset = PancreasDataset(
            args,
            args.data_path,
            mode='Validation',
            prompt=args.prompt,
            patch_size=(patch_z, patch_y, patch_x),
            margin=10,
            do_augmentation=False # no augmentation on val
        )

        nice_train_loader = DataLoader(
            pancreas_train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        nice_test_loader = DataLoader(
            pancreas_test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )



    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader