from comet_ml import Experiment
import os
import torch
import numpy as np
import pdb
import pandas as pd
import json
from core.data_utils import get_all_loaders
from core.cka_utils import calculate_CKA
from core.train_methods_mnist import train_task_sequentially, eval_single_epoch
from core.utils import save_np_arrays, setup_experiment, log_comet_metric, get_random_string
from core.utils import save_task_model_by_policy, load_task_model_by_policy, flatten_params
from core.utils import assign_weights, get_norm_distance, ContinualMeter

# DATASET = 'rot-mnist', 'imb-rot-mnist', or 'noisy-rot-mnist'
DATASET = 'rot-mnist'
HIDDENS = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRIAL_ID =  os.environ.get('NNI_TRIAL_JOB_ID', get_random_string(5))
EXP_DIR = './checkpoints/{}'.format(TRIAL_ID)

if 'rot-mnist' in DATASET:
    config = {'exp_name': '-ocs',
              # ---COMMON----
              'num_tasks': 20, 'per_task_rotation': 9, 'trial': TRIAL_ID, 'exp_dir': EXP_DIR,\
              'memory_size': 200, 'dataset': DATASET, 'device': DEVICE, 'momentum': 0.8,\
              'mlp_hiddens': HIDDENS, 'dropout': 0.2, 'lr_decay': 0.75,\
              'n_classes': 10,\

               # ----Seq Model-----
               'seq_lr': 0.005, 'stream_size': 100, 'seq_epochs': 1,\

               # ------OCS------
               'ocspick': True, 'batch_size': 10, 'is_r2c': True, 'r2c_iter': 100, \
               'tau': 1000.0, 'ref_hyp': 10., 'select_type': 'ocs_select', 'coreset_base': False,
               }
    if 'imb-rot-mnist' in DATASET:
        config['lr_decay'] = 0.8
        config['ref_hyp'] = 50.

    if 'debug' in config['exp_name']:
        config['num_tasks'] = 2


import datetime
now = datetime.datetime.now()
Thistime = now.strftime('%Y-%m-%d-%H-%M-%S')
print(Thistime)       # 2018-07-28 12:11:32

config['exp_name'] = Thistime+'-'+config['dataset']+config['exp_name']+'_seqlr'+('%s'%config['seq_lr']).replace('.','p')+'_seqep'+'%s'%config['seq_epochs']+'_seqbs'+'%s'%config['stream_size']


from torch.utils.tensorboard import SummaryWriter
summary = SummaryWriter('../summary/'+config['exp_name'])

config['trial'] = TRIAL_ID
experiment = Experiment(api_key="hidden_key", \
                        project_name="rot-mnist-20", \
                        workspace="cl-modeconnectivity", disabled=True)

loaders = get_all_loaders(config['dataset'], config['num_tasks'],\
                         config['batch_size'], config['stream_size'],\
                         config['memory_size'], config.get('per_task_rotation'))

def main():
    print('Started the trial >>', TRIAL_ID, 'for experiment 1')
    print(config)
    # init and save
    setup_experiment(experiment, config)

    accs_max = [0. for _ in range(config['num_tasks'])]
    for task in range(1, config['num_tasks']+1):
        accs_max_temp = [0. for _ in range(config['num_tasks'])]
        print('---- Task {} (OCS) ----'.format(task))
        seq_model = train_task_sequentially(task, loaders, config, summary)
        save_task_model_by_policy(seq_model, task, 'seq', config['exp_dir'])

        accs_rcp_temp, losses_rcp_temp = [], []
        for prev_task in range(1, task+1):
            metrics_rcp = eval_single_epoch(seq_model, loaders['sequential'][prev_task]['val'], config)
            accs_rcp_temp.append(metrics_rcp['accuracy'])
            losses_rcp_temp.append(metrics_rcp['loss'])
            print('OCS >> ', prev_task, metrics_rcp)
            accs_max_temp[prev_task-1] = metrics_rcp['accuracy'] if accs_max[prev_task-1] < metrics_rcp['accuracy'] else accs_max[prev_task-1]

        print("OCS >> (average accuracy): {}".format(np.mean(accs_rcp_temp)))
        print("OCS >> (Forgetting): {}".format(np.sum(np.array(accs_max[:task-1])-np.array(accs_rcp_temp[:task-1]))/(task-1)))

        summary.add_scalar('cl_average_accuracy', np.mean(accs_rcp_temp), task-1)
        print('maximum per-task accuracy >>): {}'.format(accs_max))
        accs_max = accs_max_temp
        print()

    print(config)
    experiment.end()


if __name__ == "__main__":
    main()
