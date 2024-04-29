from model import *
import torch_utils as tu
from torch_utils import Write_csv,earlystopping
from data.data_process import *
from data.get_data import build_dataloader
import torch
import torch.nn as nn
import numpy as np
import test
import yaml
from datetime import datetime

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
class EXP():
    def __init__(self,args):
        assert args.resume_dir==args.output_dir
        self.agrs=args
        tu.dist.init_distributed_mode(args)
        # get_data
        (adj, self.train_dataloader,self.val_dataloader, self.test_dataloader,
         self.train_sampler,self.val_sampler,self.test_sampler) = build_dataloader(args)
        self.adj=adj # get adj

        # get_model
        self.build_model(args, adj)
        self.model.to(device)

        self.model = tu.dist.ddp_model(self.model, [args.local_rank])
        if args.dp_mode:
            self.model = nn.DataParallel(self.model)
            print('using dp mode')

        criterion = nn.MSELoss()
        self.criterion=criterion

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer=optimizer

        # Weight decay: cos decay
        lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch,eta_min=args.lr / 1000)
        self.lr_optimizer=lr_optimizer

        # Early stop
        if args.output_dir==None or args.output_dir=='None' or args.output_dir=='none':
            args.output_dir = None
            tu.config.create_output_dir(args)
            args.resume_dir=args.output_dir

        output_path = os.path.join(args.output_dir,args.model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        path = os.path.join(output_path, args.data_name + '_best_model.pkl')
        self.output_path = output_path
        self.early_stopping = earlystopping.EarlyStopping(path=path, optimizer=self.optimizer,
                                                          scheduler=self.lr_optimizer, patience=args.patience)
        resume_path = os.path.join(args.resume_dir,args.model_name)
        if not os.path.exists(resume_path):
            raise print('No corresponding path to read the pre-trained weights was found')
        resume_path = os.path.join(resume_path, args.data_name + '_best_model.pkl')
        self.resume_path = resume_path

        if args.resume:
            print('Loading pre-trained models')
            try:
                dp_mode = args.args.dp_mode
            except AttributeError as e:
                dp_mode = True
            hparam_path = os.path.join(args.output_dir, 'hparam.yaml')
            with open(hparam_path, 'r') as f:
                hparam_dict = yaml.load(f, yaml.FullLoader)
                args.output_dir = hparam_dict['output_dir']

            # Read the best weights
            self.load_best_model(path=self.resume_path,args=args, distributed=dp_mode)

    '''modelling'''
    def build_model(self,args,adj):
        if args.model_name == 'PA2MGCN':
            args.heads = 1 # Used in Patch Attention
            args.if_T_i_D=True # Whether to use date coding
            args.if_D_i_W=True # Whether to use weekly codes
            args.if_node=True # Is it a node
            args.day_of_week_size=7 # Select n days of the week
            args.time_of_day_size=args.points_per_hour*24 #How many time steps are recorded in a day
            args.num_layer=3
            self.model = PA2MGCN(num_nodes=args.num_nodes,pred_len=args.pred_len, num_features=args.num_features, supports=adj, args=args)


        else:
            raise NotImplementedError

    '''Code under one epoch'''
    def train_test_one_epoch(self,args,dataloader,adj,save_manager: tu.save.SaveManager,epoch,mode='train',max_iter=float('inf'),**kargs):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
        elif mode == 'test' or mode =='val':
            self.model.eval()
        else:
            raise NotImplementedError

        metric_logger = tu.metric.MetricMeterLogger() # Initialise a dictionary to record the loss results of the corresponding training

        # Dataloader
        for index, unpacked in enumerate(
                metric_logger.log_every(dataloader, header=mode, desc=f'{mode} epoch {epoch}')):
            if index > max_iter:
                break
            seqs, seqs_time,targets,targets_time = unpacked # (B,L,C,N)
            seqs, targets = seqs.cuda().float(), targets.cuda().float()
            seqs_time, targets_time = seqs_time.cuda().float(), targets_time.cuda().float()
            seqs,targets=seqs.permute(0,2,3,1),targets.permute(0,2,3,1)# (B,L,C,N)
            seqs_time, targets_time = seqs_time.permute(0, 2, 3, 1), targets_time.permute(0, 2, 3, 1) #(B,C,N=1,L)
            # TODO The input and output dimensions of the model are both (B,C,N,L). The feature dimension of the output defaults to 1
            self.adj = np.array(self.adj)
            pred = self.model(seqs,self.adj,seqs_time=seqs_time,targets_time=targets_time)

            # 计算损失 TODO By default the first feature dimension is calculated
            targets=targets[:, 0:1, ...]
            if pred.shape[1]!=1:
                pred=pred[:,0:1,...]

            loss = self.criterion(pred.to(targets.device), targets) # 0 means that the feature takes only one feature, the flow rate (refer to the source code of DGCN).
            # Calculate MSE, MAE losses
            mse = torch.mean(torch.sum((pred - targets) ** 2, dim=1).detach())
            mae = torch.mean(torch.sum(torch.abs(pred - targets), dim=1).detach())

            metric_logger.update(loss=loss, mse=mse, mae=mae)  # Updated training records

            step_logs = metric_logger.values()
            step_logs['epoch'] = epoch
            save_manager.save_step_log(mode, **step_logs)  # Save the training loss for each batch

            if mode == 'train':
                loss.backward()
                # gradient cropping
                if args.clip_max_norm > 0:  # Crop value greater than 0
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        epoch_logs = metric_logger.get_finish_epoch_logs()
        epoch_logs['epoch'] = epoch
        save_manager.save_epoch_log(mode, **epoch_logs)  # Saving the training loss for each epoch

        return epoch_logs

    def train(self):
        args=self.agrs
        if args.resume!=True:
            tu.config.create_output_dir(args)  # Creating a directory for output
            print('output dir: {}'.format(args.output_dir))
            start_epoch = 0
        else:
            start_epoch=self.start_epoch

        # The following hyperparameters are saved
        save_manager = tu.save.SaveManager(args.output_dir, args.model_name, 'mse', compare_type='lt', ckpt_save_freq=30)
        save_manager.save_hparam(args)

        max_iter = float('inf')

        # Here is the start of the official training
        for epoch in range(start_epoch, args.end_epoch):
            if tu.dist.is_dist_avail_and_initialized():
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            tu.dist.barrier()

            # train
            self.train_test_one_epoch(args,self.train_dataloader,self.adj, save_manager, epoch, mode='train')

            self.lr_optimizer.step()

            # val
            val_logs = self.train_test_one_epoch(args, self.val_dataloader, self.adj, save_manager, epoch, mode='val')

            # test
            test_logs = self.train_test_one_epoch(args,self.test_dataloader,self.adj, save_manager, epoch,mode='test')


            # Early Stop Mechanism
            self.early_stopping(val_logs['mse'], model=self.model, epoch=epoch)
            if self.early_stopping.early_stop:
                break
        # Training complete. Read the best weights.
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True
        output_path = os.path.join(self.output_path, args.data_name + '_best_model.pkl')
        self.load_best_model(path=output_path, args=args, distributed=dp_mode)


    def ddp_module_replace(self,param_ckpt):
        return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}

    # TODO Load the best model
    def load_best_model(self, path, args=None, distributed=True):

        ckpt_path = path
        if not os.path.exists(ckpt_path):
            print('The path {0} does not exist and the parameters of the model are randomly initialised'.format(ckpt_path))

        ckpt = torch.load(ckpt_path)

        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_optimizer.load_state_dict(ckpt['lr_scheduler'])
        self.start_epoch=ckpt['epoch']

    def test(self):
        args=self.agrs
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True

        # 读取最好的权重
        if args.resume:
            self.load_best_model(path=self.resume_path, args=args, distributed=dp_mode)
        star = datetime.now()
        metric_dict=test.test(args,self.model,test_dataloader=self.test_dataloader,adj=self.adj)
        end=datetime.now()
        test_cost_time=(end-star).total_seconds()
        print("test took: {0} seconds.".format(test_cost_time))
        mae=metric_dict['mae']
        mse=metric_dict['mse']
        rmse=metric_dict['rmse']
        mape=metric_dict['mape']


        # 创建csv文件记录训练结果
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                          'batch_size', 'seed', 'MAE', 'MSE', 'RMSE','MAPE','seq_len',
                           'pred_len', 'd_model', 'd_ff','test_cost_time',
                           # 'e_layers', 'd_layers',
                            'info','output_dir']]
            Write_csv.write_csv(log_path, table_head, 'w+')

        time = datetime.now().strftime('%Y%m%d-%H%M%S')  # Get the current system time
        a_log = [{'dataset': args.data_name, 'model': args.model_name, 'time': time,
                  'LR': args.lr,
                  'batch_size': args.batch_size,
                  'seed': args.seed, 'MAE': mae, 'MSE': mse,'RMSE':rmse,"MAPE":mape,'seq_len': args.seq_len,
                  'pred_len': args.pred_len,'d_model': args.d_model, 'd_ff': args.d_ff,
                  'test_cost_time': test_cost_time,
                  # 'e_layers': args.e_layers, 'd_layers': args.d_layers,
                  'info': args.info,'output_dir':args.output_dir}]
        Write_csv.write_csv_dict(log_path, a_log, 'a+')





