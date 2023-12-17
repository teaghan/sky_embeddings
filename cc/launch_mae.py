import numpy as np
import os
import sys
import argparse
import configparser

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    ## Optional arguments
    
    # Job params
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch iters after which to evaluate val set and display output.", 
                        type=int, default=1000)
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=10)
    parser.add_argument("-n", "--num_runs", 
                        help="Number of jobs to run for this simulation.", 
                        type=int, default=2)
    parser.add_argument("-acc", "--account", 
                        help="Compute Canada account to run jobs under.", 
                        type=str, default='def-sfabbro')
    parser.add_argument("-mem", "--memory", 
                        help="Memory per job in GB.", 
                        type=int, default=48)
    parser.add_argument("-ncp", "--num_cpu", 
                        help="Number of CPU cores per job.", 
                        type=int, default=12)
    parser.add_argument("-jt", "--job_time", 
                        help="Number of hours per job.", 
                        type=int, default=3)
    
    # Config params
    parser.add_argument("-sfn", "--source_data_file", 
                        help="Source data file for training.", 
                        type=str, default='gaia_grid_crossref.h5') 
    parser.add_argument("-tfn", "--target_data_file", 
                        help="Target data file for training.", 
                        type=str, default='gaia_observed_crossref.h5') 
    parser.add_argument("-wfn", "--wave_grid_file", 
                        help="Wave grid file.", 
                        type=str, default='gaia_wavegrid.npy')
    parser.add_argument("-mmk", "--multimodal_keys",  type=str, nargs='+',
                        help="Dataset keys for labels in data file.", 
                        default="[]") 
    parser.add_argument("-umk", "--unimodal_keys",  type=str, nargs='+',
                        help="Dataset keys for labels in data file.", 
                        default="['teff', 'feh', 'logg', 'alpha']") 
    parser.add_argument("-tvs", "--target_val_survey", 
                        help="Survey for target domain label data.", 
                        type=str, default='APOGEE') 
    parser.add_argument("-cn", "--continuum_normalize", 
                        help="Whether or not to continuum normalize each spectrum.", 
                        type=str, default='False')
    parser.add_argument("-dbm", "--divide_by_median", 
                        help="Whether or not to divide each spectrum by its median value.", 
                        type=str, default='False')
    parser.add_argument("-ado", "--apply_dropout", 
                        help="Whether or not to dropout chunks of flux value in the source spectra during training.", 
                        type=str, default='False')
    parser.add_argument("-an", "--add_noise_to_source", 
                        help="Whether or not to add noise to source spectra during training.", 
                        type=str, default='True')
    parser.add_argument("-mnf", "--max_noise_factor", 
                        help="Maximum fraction of continuum to set random noise to.", 
                        type=float, default=0.1)
    parser.add_argument("-ssm", "--std_min", 
                        help="Threshold for standard deviation of a channel in the spectrum (if lower, that channel will not be used).", 
                        type=float, default=0.00001)
    parser.add_argument("-au", "--augs", type=str, nargs='+',
                        help="Names of the augmentation to use during training.", 
                        default=[])
    parser.add_argument("-aum", "--aug_means", 
                        help="Mean value of each augmentation used for normalization.", 
                        default=[])
    parser.add_argument("-aus", "--aug_stds", 
                        help="Standard deviation of each augmentation used for normalization.", 
                        default=[])
    parser.add_argument("-umm", "--unimodal_means", 
                        help="Mean value of each label used for normalization.", 
                        default=[5375, -2.0, 2.5, 0.2])
    parser.add_argument("-ums", "--unimodal_stds", 
                        help="Standard deviation of each label used for normalization.", 
                        default=[1600, 1.8, 1.9, 0.4])
    parser.add_argument("-sm", "--spectra_mean", 
                        help="Mean flux value in spectra.", 
                        type=float, default=1)
    parser.add_argument("-ss", "--spectra_std", 
                        help="Standard dev of flux values in spectra.", 
                        type=float, default=0.15)
    
    parser.add_argument("-upa", "--use_prev_ae", 
                        help="Whether or not to skip the MAE training by using a previously trained model.", 
                        type=str, default='False')
    parser.add_argument("-pan", "--prev_ae_name", 
                        help="The name of the previously trained auto-encoder.", 
                        type=str, default='None')
    parser.add_argument("-bs", "--batch_size", 
                        help="Training batchsize.", 
                        type=int, default=128)
    parser.add_argument("-lr", "--lr", 
                        help="Initial learning rate.", 
                        type=float, default=0.001)
    parser.add_argument("-lrf", "--final_lr_factor", 
                        help="Final lr will be lr/lrf.", 
                        type=float, default=1000.0)
    parser.add_argument("-tlw", "--target_loss_weight", 
                        help="Loss weighting placed on samples from target domain.", 
                        type=float, default=10.0)
    parser.add_argument("-wd", "--weight_decay", 
                        help="Weight decay for AdamW optimizer.", 
                        type=float, default=0.01)
    parser.add_argument("-ti", "--total_batch_iters", 
                        help="Total number of batch iterations for training.", 
                        type=int, default=100000)
    parser.add_argument("-mr", "--mask_ratio", 
                        help="Fraction of patches that will be masked during training.", 
                        type=float, default=0.5)
    
    parser.add_argument("-lpop", "--lp_optimizer", 
                        help="Training optimizer for linear probe.", 
                        type=str, default='adamw')
    parser.add_argument("-lpbs", "--lp_batch_size", 
                        help="Training batchsize for linear probe.", 
                        type=int, default=1024)
    parser.add_argument("-lplr", "--lp_lr", 
                        help="Initial learning rate for linear probe.", 
                        type=float, default=0.1)
    parser.add_argument("-lplrf", "--lp_final_lr_factor", 
                        help="Final lr will be lr/lrf for linear probe.", 
                        type=float, default=500.0)
    parser.add_argument("-lpwd", "--lp_weight_decay", 
                        help="Weight decay for AdamW optimizer for linear probe.", 
                        type=float, default=0.0)
    parser.add_argument("-lpti", "--lp_total_batch_iters", 
                        help="Total number of batch iterations for training for linear probe.", 
                        type=int, default=20000)
    parser.add_argument("-lpdo", "--lp_dropout", 
                        help="Dropout ratio for linear probe.", 
                        type=float, default=0.95)
    parser.add_argument("-lpel", "--num_enc_layers", 
                        help="The number of layers in the encoder to finetune (0, 1, or 2).", 
                        type=int, default=2)
    parser.add_argument("-lpls", "--label_smoothing", 
                        help="Label smoothing for cross-entropy loss.", 
                        type=float, default=0.0)
    
    parser.add_argument("-ssz", "--spectrum_size", 
                        help="Number of flux values in spectrum.", 
                        type=int, default=800)
    parser.add_argument("-psz", "--patch_size", 
                        help="Number of flux values in each patch of the spectrum.", 
                        type=int, default=50)
    parser.add_argument("-eed", "--encoder_embed_dim", 
                        help="Dimension of encoder.", 
                        type=int, default=64)
    parser.add_argument("-edp", "--encoder_depth", 
                        help="Depth of encoder.", 
                        default=6)
    parser.add_argument("-enh", "--encoder_num_heads", 
                        help="Number of heads in encoder (should multiply evenly into encoder_embed_dim).", 
                        type=int, default=4)
    parser.add_argument("-ded", "--decoder_embed_dim", 
                        help="Dimension of decoder.", 
                        type=int, default=128)
    parser.add_argument("-ddp", "--decoder_depth", 
                        help="Depth of decoder.", 
                        default=4)
    parser.add_argument("-dnh", "--decoder_num_heads", 
                        help="Number of heads in decoder (should multiply evenly into decoder_embed_dim).", 
                        type=int, default=8)
    parser.add_argument("-mlr", "--mlp_ratio", 
                        help="Ratio of internal dimension of each block.", 
                        type=int, default=4)
    
    parser.add_argument("-co", "--comment", 
                        help="Comment for config file.", 
                        default='Original.')
    
    # Parse arguments
    args = parser.parse_args()

    return args

# Directories
cur_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cur_dir, '../data')
model_dir = os.path.join(cur_dir, '../models/')
training_script1 = os.path.join(cur_dir, '../train_mae.py')
testing_script1 = os.path.join(cur_dir, '../test_mae.py')
training_script2 = os.path.join(cur_dir, '../train_linear_probe3.py')
testing_script2 = os.path.join(cur_dir, '../test_linear_probe3.py')

# Read command line arguments
args = parseArguments()

# Configuration filename
config_fn = os.path.join(cur_dir, '../configs', args.model_name+'.ini')
if os.path.isfile(config_fn):
    good_to_go = False
    while not good_to_go: 
        user_input = input('This config file already exists, would you like to:\n'+
                           '-Overwrite the file (o)\n' + 
                           '-Run the existing file for another %i runs (r)\n' % (args.num_runs) + 
                           '-Or cancel (c)?\n')
        if (user_input=='o') or (user_input=='r') or (user_input=='c'):
            good_to_go = True
        else:
            print('Please choose "o" "r" or "c"')
else:
    user_input = 'o' 

if user_input=='c':
    sys.exit()  
elif user_input=='o':
    # Create new configuration file
    config = configparser.ConfigParser()
    
    config['DATA'] = {'source_data_file': args.source_data_file, 
                      'target_data_file': args.target_data_file, 
                      'wave_grid_file': args.wave_grid_file, 
                      'multimodal_keys': args.multimodal_keys,
                      'unimodal_keys': args.unimodal_keys,
                      'target_val_survey': args.target_val_survey,
                      'continuum_normalize': args.continuum_normalize,
                      'divide_by_median': args.divide_by_median,
                      'apply_dropout': args.apply_dropout,
                      'add_noise_to_source': args.add_noise_to_source,
                      'max_noise_factor': args.max_noise_factor,
                      'std_min': args.std_min,
                      'augs': args.augs,
                      'aug_means': args.aug_means,
                      'aug_stds': args.aug_stds,
                     'unimodal_means': args.unimodal_means,
                      'unimodal_stds': args.unimodal_stds,
                      'spectra_mean': args.spectra_mean,
                      'spectra_std': args.spectra_std}
    
    config['TRAINING'] = {'use_prev_ae': args.use_prev_ae,
                          'prev_ae_name': args.prev_ae_name,
                          'batch_size': args.batch_size,
                          'lr': args.lr,
                          'final_lr_factor': args.final_lr_factor,
                          'target_loss_weight': args.target_loss_weight,
                          'weight_decay': args.weight_decay,
                          'total_batch_iters': args.total_batch_iters,
                          'mask_ratio': args.mask_ratio}
    
    config['LINEAR PROBE TRAINING'] = {'optimizer': args.lp_optimizer,
                                       'batch_size': args.lp_batch_size,
                                       'lr': args.lp_lr,
                                       'final_lr_factor': args.lp_final_lr_factor,
                                       'weight_decay': args.lp_weight_decay,
                                       'total_batch_iters': args.lp_total_batch_iters,
                                       'dropout': args.lp_dropout,
                                       'num_enc_layers': args.num_enc_layers,
                                       'label_smoothing': args.label_smoothing}
    
    config['MAE ARCHITECTURE'] = {'spectrum_size': args.spectrum_size,
                              'patch_size': args.patch_size,
                              'encoder_embed_dim': args.encoder_embed_dim,
                              'encoder_depth': args.encoder_depth,
                              'encoder_num_heads': args.encoder_num_heads,
                              'decoder_embed_dim': args.decoder_embed_dim,
                              'decoder_depth': args.decoder_depth,
                              'decoder_num_heads': args.decoder_num_heads,
                              'mlp_ratio': args.mlp_ratio}
        
    config['Notes'] = {'comment': args.comment}

    with open(config_fn, 'w') as configfile:
        config.write(configfile)
        
    source_data_file = args.source_data_file
    target_data_file = args.target_data_file
    wave_grid_file = args.wave_grid_file
    
    # Delete existing model file
    model_filename =  os.path.join(model_dir, args.model_name+'.pth.tar')
    if os.path.exists(model_filename):
        os.remove(model_filename)

elif user_input=='r':
    config = configparser.ConfigParser()
    config.read(config_fn)
    source_data_file = os.path.join(data_dir, config['DATA']['source_data_file'])
    target_data_file = os.path.join(data_dir, config['DATA']['target_data_file'])
    wave_grid_file = os.path.join(data_dir, config['DATA']['wave_grid_file'])

todo_dir = os.path.join(cur_dir, '../scripts/todo')
done_dir = os.path.join(cur_dir, '../scripts/done')
stdout_dir = os.path.join(cur_dir, '../scripts/stdout')

# Create script directories
if not os.path.exists(os.path.join(cur_dir,'../scripts')):
    os.mkdir(os.path.join(cur_dir,'../scripts'))
if not os.path.exists(todo_dir):
    os.mkdir(todo_dir)
if not os.path.exists(done_dir):
    os.mkdir(done_dir)
if not os.path.exists(stdout_dir):
    os.mkdir(stdout_dir)

# Create script file
script_fn = os.path.join(todo_dir, args.model_name+'.sh')
with open(script_fn, 'w') as f:
    f.write('#!/bin/bash\n\n')
    f.write('# Module loads\n')
    for line in open(os.path.join(cur_dir,'module_loads.txt'), 'r').readlines():
        f.write(line)
    f.write('\n\n')
    f.write('# Copy files to slurm directory\n')
    f.write('cp %s $SLURM_TMPDIR\n' % (os.path.join(data_dir, source_data_file)))
    f.write('cp %s $SLURM_TMPDIR\n' % (os.path.join(data_dir, target_data_file)))
    f.write('cp %s $SLURM_TMPDIR\n\n' % (os.path.join(data_dir, wave_grid_file)))
    f.write('# Run MAE training\n')
    f.write('python %s %s -v %i -ct %0.2f -dd $SLURM_TMPDIR/\n' % (training_script1, 
                                                                   args.model_name,
                                                                   args.verbose_iters, 
                                                                   args.cp_time))
    f.write('\n# Run MAE testing\n')
    f.write('python %s %s -dd $SLURM_TMPDIR/\n' % (testing_script1,
                                                   args.model_name))
    
    f.write('\n# Run Linear Probe training\n')
    f.write('python %s %s -v %i -ct %0.2f -dd $SLURM_TMPDIR/\n' % (training_script2, 
                                                                   args.model_name,
                                                                   args.verbose_iters, 
                                                                   args.cp_time))
    f.write('\n# Run Linear Probe testing\n')
    f.write('python %s %s -dd $SLURM_TMPDIR/\n' % (testing_script2,
                                                   args.model_name))

# Compute-canada goodies command
cmd = 'python %s ' % (os.path.join(cur_dir, 'queue_cc.py'))
cmd += '--account "%s" --todo_dir "%s" ' % (args.account, todo_dir)
cmd += '--done_dir "%s" --output_dir "%s" ' % (done_dir, stdout_dir)
cmd += '--num_jobs 1 --num_runs %i --num_gpu 1 ' % (args.num_runs)
cmd += '--num_cpu %i --mem %sG --time_limit "00-0%i:00"' % (args.num_cpu, args.memory, args.job_time)

# Execute jobs
os.system(cmd)
