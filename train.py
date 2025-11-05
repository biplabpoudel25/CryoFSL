import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import datasets
import models
import utils
from statistics import mean
import time

torch.cuda.empty_cache()  # Frees up unused memory


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=True, pin_memory=True, num_workers=8)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')

    # Get the total number of samples in each dataset
    train_samples = len(train_loader.dataset) if train_loader is not None else 0
    val_samples = len(val_loader.dataset) if val_loader is not None else 0

    print(f"Training dataset: {train_samples} samples")
    print(f"Validation dataset: {val_samples} samples")

    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    return model, optimizer, epoch_start, lr_scheduler


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(train_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    pbar = tqdm(total=len(train_loader), desc='Training')
    loss_list = []

    for batch in train_loader:
        for k, v in batch.items():
            if k != 'filename':
                batch[k] = v.to(device)

        inp = batch['inp']
        gt = batch['gt']

        model.set_input(inp, gt)
        model.optimize_parameters()

        loss_list.append(model.loss_G.item())
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss_list[-1]:.4f}'})

    pbar.close()
    torch.cuda.empty_cache()

    return mean(loss_list)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_

    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.to(device)

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint['model'], strict=False)

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    params = count_trainable_parameters(model)
    log_info = ['Number of trainable parameters: {}'.format(params)]
    log(', '.join(log_info))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    min_train_loss = float('inf')
    best_loss_epoch = -1
    best_epoch = -1
    best_metric_value = None
    timer = utils.Timer()

    total_train_time = 0  # Initialize total training time

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        train_start_time = time.time()
        train_loss_G = train(train_loader, model)
        train_end_time = time.time()

        total_train_time += train_end_time - train_start_time

        if train_loss_G < min_train_loss:
            min_train_loss = train_loss_G
            best_loss_epoch = epoch
            save(config, model, save_path, 'best_loss')

        lr_scheduler.step()

        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, metric1, metric2 = eval_psnr(val_loader, model, eval_type=config.get('eval_type'))

            log_info.append('val: {}={:.4f}'.format(metric1, result1))
            writer.add_scalars(metric1, {'val': result1}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric2, result2))
            writer.add_scalars(metric2, {'val': result2}, epoch)

            if config['eval_type'] != 'ber':
                if result1 > max_val_v:
                    max_val_v = result1
                    best_epoch = epoch
                    best_metric_value = result1
                    save(config, model, save_path, 'best')
                    log_info.append('\nNew best checkpoint saved with {} = {:.4f}'.format(metric1, result1))
                    writer.add_scalar('best_checkpoint', result1, epoch)
            else:
                if result2 < max_val_v:
                    max_val_v = result2
                    best_epoch = epoch
                    best_metric_value = result2
                    save(config, model, save_path, 'best')
                    log_info.append('\nNew best checkpoint saved with {} = {:.4f}'.format(metric2, result2))
                    writer.add_scalar('best_checkpoint', result2, epoch)

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log(', '.join(log_info))
            writer.flush()

    # Log final summary
    final_log_info = []
    if best_epoch != -1:
        metric_name = metric1 if config['eval_type'] != 'ber' else metric2
        final_log_info.append('Training completed - Best model saved at epoch {} with {} = {:.4f}'.format(
            best_epoch, metric_name, best_metric_value))
    else:
        final_log_info.append('Training completed - No best model was saved during training')

    final_log_info.append('Total training time: {:.4f} seconds'.format(total_train_time))
    log(', '.join(final_log_info))
    writer.flush()


def eval_psnr(loader, model, eval_type=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    metric_fn = utils.calc_kvasir
    metric1, metric2 = 'dice', 'iou'

    pbar = None

    pred_list = []
    gt_list = []

    val_metric1 = 0
    val_metric2 = 0
    cnt = 0

    for batch in loader:
        for k, v in batch.items():
            if k != 'filename':
                batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        result1, result2 = metric_fn(pred, batch['gt'])
        val_metric1 += (result1 * pred.shape[0])
        val_metric2 += (result2 * pred.shape[0])
        cnt += pred.shape[0]
        if pbar is not None:
            pbar.update(1)
    val_metric1 = torch.tensor(val_metric1).cuda()
    val_metric2 = torch.tensor(val_metric2).cuda()
    cnt = torch.tensor(cnt).cuda()

    if pbar is not None:
        pbar.close()
    torch.cuda.empty_cache()

    return val_metric1.item() / cnt, val_metric2.item() / cnt, metric1, metric2


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


def override_config(config, args):
    """Override parts of config with user args (if provided)."""
    # Training dataset overrides
    if args.train_images is not None:
        config["train_dataset"]["dataset"]["args"]["root_path_1"] = args.train_images
    if args.train_labels is not None:
        config["train_dataset"]["dataset"]["args"]["root_path_2"] = args.train_labels

    # Validation dataset overrides
    if args.val_images is not None:
        config["val_dataset"]["dataset"]["args"]["root_path_1"] = args.val_images
    if args.val_labels is not None:
        config["val_dataset"]["dataset"]["args"]["root_path_2"] = args.val_labels

    # Epochs override
    if args.epochs is not None:
        config["epoch_max"] = args.epochs

    # SAM checkpoint override
    if args.sam_ckpt is not None:
        config["sam_checkpoint"] = args.sam_ckpt

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/cod-sam-vit-l.yaml")
    parser.add_argument('--name', default='10028_few_shot')
    parser.add_argument('--shot', type=int, default=5, choices=[1, 5, 10], required=False,
                        help='Number of shots for few-shot training (1, 5, or 10)')
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument('--resume', type=str,
                        default=None, help="Path to trained checkpoint, leave empty to load SAM ckpt")

    # Arguments for training and validation images and labels
    parser.add_argument('--train_images', type=str, default="./data/train/images/", help= 'path/to/train/images')
    parser.add_argument('--train_labels', type=str, default="./data/train/labels/", help= 'path/to/train/labels')
    parser.add_argument('--val_images', type=str, default="./data/valid/images/", help= 'path/to/valid/images')
    parser.add_argument('--val_labels', type=str, default="./data/valid/labels/", help= 'path/to/valid/labels')
    parser.add_argument('--epochs', type=int, default=5000, help="Number of epochs")
    parser.add_argument('--sam_ckpt', type=str,
                        default="./pretrained/sam2_hiera_large.pt",
                        help="Path to SAM2 checkpoint")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = override_config(config, args)

    dir_save = 'checkpoints'
    os.makedirs(dir_save, exist_ok=True)
    shot_dir = f'{args.shot}_shot'
    save_name = args.name
    print(args.name)
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join(dir_save, shot_dir, save_name)
    os.makedirs(os.path.join(dir_save, shot_dir), exist_ok=True)
    main(config, save_path, args=args)