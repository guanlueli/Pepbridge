import sys
sys.path.append('./')
import wandb
import os
import shutil
import argparse
import torch
import torch.cuda.amp as amp
import torch.distributed as distrib
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from pepbridge.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from pepbridge.utils.data import PaddingCollate
from pepbridge.utils.train import ScalarMetricAccumulator, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, sum_weighted_losses
from models_con.pep_dataloader import PepDataset
from models_con.diffusion_model import DiffusionModel
import time

BASE_DIR = '../Pepbridge'
DATA_DIR = './data'

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=f'{BASE_DIR}/configs/learn_surf_angle.yaml')
    parser.add_argument('--base_dir', type=str, default=BASE_DIR)
    parser.add_argument('--logdir', type=str, default=f"{DATA_DIR}/logs")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='Sample1')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--name', type=str, default='pepbridge')
    return parser.parse_args()

def check_abnormal_gradients(model, threshold_high=1.0, threshold_low=1e-6):
    """Check for abnormal gradient norms"""
    problematic_layers = {'high': [], 'low': [], 'nan': []}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            
            if torch.isnan(param.grad).any():
                problematic_layers['nan'].append((name, grad_norm))
            elif grad_norm > threshold_high:
                problematic_layers['high'].append((name, grad_norm))
            elif grad_norm < threshold_low:
                problematic_layers['low'].append((name, grad_norm))
    
    return problematic_layers

def compare_initialization():
    print("=== WEIGHT INITIALIZATION COMPARISON ===")
    for name, param in model.named_parameters():
        if 'bb_update' in name:  # Focus on problematic layers
            weight_std = param.data.std().item()
            weight_mean = param.data.mean().item()
            print(f"{name}: mean={weight_mean:.6f}, std={weight_std:.6f}")

def check_data_ranges(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bool:
                # For boolean tensors, show count of True/False
                true_count = value.sum().item()
                total_count = value.numel()
                false_count = total_count - true_count
                print(f"{key} (bool): True={true_count}, False={false_count}, "
                      f"shape={value.shape}")
            elif value.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                # For integer tensors
                print(f"{key} (int): min={value.min().item()}, max={value.max().item()}, "
                      f"shape={value.shape}")
            elif value.dtype in [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128]:
                # For float/complex tensors
                print(f"{key} (float): min={value.min():.4f}, max={value.max():.4f}, "
                      f"mean={value.mean():.4f}, std={value.std():.4f}, shape={value.shape}")
            else:
                # For other types
                print(f"{key}: dtype={value.dtype}, shape={value.shape}")
        else:
            print(f"{key}: {type(value)} = {value}")

if __name__ == '__main__':
    
    args = parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)
    config['device'] = args.device

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        run = wandb.init(project=args.name,
                        config=config,
                        name='%s[%s]' % (
                            time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()),  # prepend time
                            f'{config_name}_{args.tag}' if args.tag else config_name
                        ),
                        dir=f'{DATA_DIR}/wandb'
                    )
        log_dir = run.dir  # This shows the directory for the current run
        print(f"Wandb logs are saved to: {log_dir}")
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix='%s[%s]' % (config_name, '1'), tag=args.tag)
        with open(os.path.join(log_dir, 'commit.txt'), 'w') as f:
            f.write('base' + '\n')
            f.write('1' + '\n')
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
            shutil.copytree(args.base_dir + '/models_con', os.path.join(log_dir, 'models_con'))
            shutil.copytree(args.base_dir + '/data', os.path.join(log_dir, 'data'))

    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    train_dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, dataset_dir = config.dataset.train.dataset_dir,
                                            name = config.dataset.train.name, transform=None, reset=config.dataset.train.reset)
    val_dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=args.num_workers, pin_memory=True)
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    # model = get_model(config.model).to(args.device)
    model = DiffusionModel(config.model, args.device).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    compare_initialization()
    
    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    def train(it):

        time_start = current_milli_time()
        model.train()

        batch = recursive_to(next(train_iterator), args.device)

        # check_data_ranges(batch)

        loss, loss_dict = model(batch) # get loss and metrics
        time_forward_end = current_milli_time()
        
        if torch.isnan(loss):
            print('NAN Loss!')
            torch.save({'batch':batch,'loss':loss,'loss_dict':loss_dict,'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,},os.path.join(log_dir,'nan.pt'))
            loss = torch.tensor(0.,requires_grad=True).to(loss.device)

        loss.backward()
        
        # rescue for nan grad
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    param.grad[torch.isnan(param.grad)] = 0

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)


        # Backward
        # if it % config.train.accum_grad ==0:
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        scalar_dict = {}
        # scalar_dict.update(metric_dict['scalar'])
        scalar_dict.update({
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger)

    def validate(it):
        time_start = current_milli_time()
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            model.eval()
            time_forward_end = current_milli_time()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)

                # Forward pass
                loss, loss_dict = model(batch)
                scalar_accum.add(name='loss', value=loss, batchsize=len(batch['aa']), mode='mean')
            
        # avg_loss = scalar_accum.get_average('loss')
        summary = scalar_accum.log(it, 'val', logger=logger, writer=writer)
        for k, v in summary.items():
            wandb.log({f'val/{k}': v}, step=it)
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(loss)
        else:
            scheduler.step()

        time_backward_end = current_milli_time()
        # Logging
        scalar_dict = {}
        # scalar_dict.update(metric_dict['scalar'])
        scalar_dict.update({
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        log_losses(loss, loss_dict, scalar_dict, it=it, tag='val', logger=logger)
        
        return loss

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            # if it % config.train.val_freq == 0:
            #     avg_val_loss = validate(it)
                # if not args.debug:
            if it % config.train.val_freq == 0:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    # 'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')