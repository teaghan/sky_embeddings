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
                        type=int, default=2000)
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
                        type=int, default=16)
    parser.add_argument("-ngpu", "--num_gpu", 
                        help="Number of GPUs per job.", 
                        type=int, default=2)
    parser.add_argument("-ncp", "--num_cpu", 
                        help="Number of CPU cores per job.", 
                        type=int, default=24)
    parser.add_argument("-jt", "--job_time", 
                        help="Number of hours per job.", 
                        type=int, default=3)
    
    # Config params
    parser.add_argument("-tfn", "--train_data_file", 
                        help="Filename for training samples.", 
                        type=str, default='HSC_zspec_GRIZY_64_train_new.h5') 
    parser.add_argument("-vfn", "--val_data_file", 
                        help="Filename for validation samples.", 
                        type=str, default='HSC_zspec_GRIZY_64_val_new.h5') 
    parser.add_argument("-lk", "--label_keys", 
                        help="List of label keys in the datasets that will be the target labels.", 
                        type=str, nargs='+', default="['zspec']")
    parser.add_argument("-lm", "--label_means", 
                        help="List of mean values for the different labels (for normalization purposes).", 
                        default=[0.64])
    parser.add_argument("-ls", "--label_stds", 
                        help="List of standard deviation values for the different labels (for normalization purposes).", 
                        default=[0.64])

    parser.add_argument("-tm", "--train_method", 
                        help="Training method ('ft' for finetuning, 'lp' for linear probing, or 'full' for fully supervised).", 
                        type=str, default='lp')
    parser.add_argument("-pt", "--pretained_mae", 
                        help="Name of pretrained MAE model (or 'None' for fresh weights).", 
                        type=str, default='mim_1')
    parser.add_argument("-nt", "--num_train", 
                        help="Number of training samples (use -1 to use the entire dataset).", 
                        type=int, default=-1)
    parser.add_argument("-bs", "--batch_size", 
                        help="Training batchsize.", 
                        type=int, default=128)
    parser.add_argument("-ti", "--total_batch_iters", 
                        help="Total number of batch iterations for training.", 
                        type=int, default=1.5e5)
    parser.add_argument("-ld", "--layer_decay", 
                        help="Layerwise weight decay parameter for optimizer (only applies when training method is finetuning).", 
                        type=float, default=0.5)
    parser.add_argument("-wd", "--weight_decay", 
                        help="Weight decay for optimizer.", 
                        type=float, default=0.0)
    parser.add_argument("-lr", "--init_lr", 
                        help="Initial learning rate.", 
                        type=float, default=0.0005)
    parser.add_argument("-lrf", "--final_lr_factor", 
                        help="Final lr will be lr/lrf.", 
                        type=float, default=1e5)
    parser.add_argument("-aug", "--augment", 
                        help="Whether or not to apply augmentations during training.", 
                        type=str, default='True')
    
    parser.add_argument("-ims", "--img_size", 
                        help="Number of rows and columns in each image sample (must be less than or equal to the img_size used in the pretraining).", 
                        type=int, default=64)
    parser.add_argument("-gp", "--global_pool", 
                        help="Global pooling method for model head (either 'token', 'avg', or 'map').", 
                        type=str, default='map')
    parser.add_argument("-do", "--dropout", 
                        help="Amount of dropout to apply before head.", 
                        type=float, default=0.0)
    
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
training_script = os.path.join(cur_dir, '../train_predictor.py')
testing_script = os.path.join(cur_dir, '../test_predictor.py')

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

    config['DATA'] = {'train_data_file': args.train_data_file, 
                      'val_data_file': args.val_data_file, 
                      'label_keys': args.label_keys, 
                      'label_means': args.label_means, 
                      'label_stds': args.label_stds}
    
    config['TRAINING'] = {'train_method': args.train_method,
                          'pretained_mae': args.pretained_mae,
                          'num_train': args.num_train,
                          'batch_size': args.batch_size,
                          'total_batch_iters': args.total_batch_iters,
                          'layer_decay': args.layer_decay,
                          'weight_decay': args.weight_decay,
                          'init_lr': args.init_lr,
                          'final_lr_factor': args.final_lr_factor,
                          'augment': args.augment}
    
    config['ARCHITECTURE'] = {'img_size': args.img_size,
                              'global_pool': args.global_pool,
                              'dropout': args.dropout}
        
    config['Notes'] = {'comment': args.comment}

    with open(config_fn, 'w') as configfile:
        config.write(configfile)
        
    train_data_file = args.train_data_file
    val_data_file = args.val_data_file
    
    # Delete existing model file
    model_filename =  os.path.join(model_dir, args.model_name+'.pth.tar')
    if os.path.exists(model_filename):
        os.remove(model_filename)

elif user_input=='r':
    config = configparser.ConfigParser()
    config.read(config_fn)
    train_data_file = os.path.join(data_dir, config['DATA']['train_data_file'])
    val_data_file = os.path.join(data_dir, config['DATA']['val_data_file'])

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
    f.write('cp %s $SLURM_TMPDIR\n' % (os.path.join(data_dir, train_data_file)))
    f.write('cp %s $SLURM_TMPDIR\n\n' % (os.path.join(data_dir, val_data_file)))
    f.write('# Run predictor training\n')
    f.write('python %s %s -v %i -ct %0.2f -dd $SLURM_TMPDIR/\n\n' % (training_script, 
                                                                   args.model_name,
                                                                   args.verbose_iters, 
                                                                   args.cp_time))
    f.write('# Run predictor testing\n')
    f.write('python %s %s -dd $SLURM_TMPDIR/\n' % (testing_script, args.model_name))

# Compute-canada goodies command
cmd = 'python %s ' % (os.path.join(cur_dir, 'queue_cc.py'))
cmd += '--account "%s" --todo_dir "%s" ' % (args.account, todo_dir)
cmd += '--done_dir "%s" --output_dir "%s" ' % (done_dir, stdout_dir)
cmd += '--num_jobs 1 --num_runs %i --num_gpu %i ' % (args.num_runs, args.num_gpu)
#cmd += '--num_cpu "auto" --mem %sG --time_limit "00-0%i:00"' % (args.memory, args.job_time)
cmd += '--num_cpu %i --mem %sG --time_limit "00-0%i:00"' % (args.num_cpu, args.memory, args.job_time)

# Execute jobs
os.system(cmd)
