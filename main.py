import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pyramidnet import pyramidnet

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=500, type=int,help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batchsize', default=128, type=int,help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,help='total number of layers (default: 28)')
parser.add_argument('--depth', default=32, type=int, help='depth of the network (default: 32)')
parser.add_argument('--alpha', default=48, type=int,help='number of new channel increases per depth (default: 12)')
parser.add_argument('--block', default='basic',type=str,help='to use bottleneck_block (default: basic)')
parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')
parser.add_argument('--k', type=int, default=5,help='number of experts')
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--tensorboard',help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--mos',help='mixture of softmaxes', action='store_true')
parser.add_argument('--expname', default='PyramidNet', type=str,help='name of experiment')
parser.add_argument('--seed', type=int, default=1234,help='random seed')
parser.add_argument('--val_freq', type=int, default=1,help='number of epoch before validation script')
parser.add_argument('--rd', type=float, default=0.3,help='recurrent dropout')

best_acc = 0

def main():
    global args, best_acc 
    args = parser.parse_args()
  
    if args.tensorboard: configure("runs/%s"%(args.expname))

    
    # Normlization for input to the newtwork
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
            transforms.ToTensor(),
            ## Extra padding for random crop as described in paper
            transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), requires_grad=False, volatile=True),(4,4,4,4),mode='reflect').data.squeeze()), 
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # channel first or channel last 
            normalize,])

    transform_test = transforms.Compose([transforms.ToTensor(),normalize])  


    # Dataset loader 

    if args.dataset == 'cifar100':
        
        train_data=datasets.CIFAR100('../data',transform=transform_train,download=True)
        val_data= datasets.CIFAR100('../data',transform=transform_test,train=False,download=True)


        train_loader = torch.utils.data.DataLoader(train_data,  batch_size=args.batchsize, shuffle=True,num_workers=8,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data,batch_size=args.batchsize, shuffle=False, num_workers=8,pin_memory=True) 
        num_class=100       
    
    elif args.dataset == 'cifar10':
        
        train_data=datasets.CIFAR10('../data',transform=transform_train,download=True)
        val_data= datasets.CIFAR10('../data',transform=transform_test,train=False,download=True)


        train_loader = torch.utils.data.DataLoader(train_data , batch_size=args.batchsize, shuffle=True,num_workers=8,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data,batch_size=args.batchsize, shuffle=False, num_workers=8,pin_memory=True)
        num_class=10        
    
    else:
        raise Exception ('write like this:{}'.format(args.dataset))



    # Defining the main model


    model=pyramidnet(args.block,args.alpha,args.depth,args.mos,num_class,args.k,args.rd)

    
    
    print(model)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))




    # define loss function (criterion) and optimizer
    if args.mos:
        criterion=nn.NLLLoss()
        
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum, nesterov = args.nesterov,weight_decay=args.weight_decay)

    
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
   
    
    if args.cuda:
        model=model.cuda()
        criterion=criterion.cuda()


    
    # to optimise code according to hardware
    # since our image size is constant
    cudnn.benchmark =True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    
   # Training 


    for epoch in range(args.start_epoch,args.epochs):
        
        model.train()    # trainig mode
        
        losses = AverageMeter()
        train_acc = AverageMeter()

        start_time = time.time()   # calculation time needed for one epoch for training

        
        adjust_learning_rate(optimizer, epoch)  

        for images,labels in train_loader:
            
            if args.cuda:
                images=images.cuda() # load Tensor on GPU 
                labels=labels.cuda()
            
            images = Variable(images)
            labels  = Variable(labels)

            # To release gradient of previous time step
            optimizer.zero_grad()
            

            output = model(images)  # forward pass
            loss = criterion(output,labels)
            loss.backward()    # backward passs  #  applying backprop

            
            optimizer.step()    # updating weights

            losses.update(loss.data[0], images.size(0)) 

            _, predicted = torch.max(output.data, 1)


            accuracy= ((predicted == labels.data).sum()*1.0/labels.size(0))     # calculating accuracy # (0.1) otherwise you will get  0 all the time. 
            

            train_acc.update(accuracy, images.size(0))

        
        time_taken = time.time()-start_time

        print('Epoch: {0} Loss: {1} Accuracy: {2} Time: {3} '.format(epoch,losses.avg,train_acc.avg,time_taken))

        
        # log in to tensorboard
        if args.tensorboard:
            log_value('train_loss', losses.avg, epoch)
            log_value('train_error', train_acc.avg, epoch)


        if epoch%args.val_freq==0:
            
            model.eval()   # evaluate mode
        
            val_acc = AverageMeter() 
            val_loss = AverageMeter()
            start_time = time.time()  # calculation time needed for one epoch for validation

            for input, target in val_loader:

                if args.cuda:                       
                    input =input.cuda()     # load Tensor on GPU 
                    target=target.cuda()
                

                input = Variable(input,volatile=True)    # volatile=True memory optimization
                target= Variable(target,volatile=True)    

                
                output = model(input)    # forward pass
                loss = criterion(output,target) 

                val_loss.update(loss.data[0], input.size(0))


                _, predicted = torch.max(output.data, 1)  # calculating accuracy

                val_accuracy= ((predicted == target.data).sum()*1.0/target.size(0))
                val_acc.update(val_accuracy,input.size(0))


            time_taken = time.time()-start_time  

            print('Val_loss: {0} Val_Accuracy: {1} Val_Time: {2} '.format(val_loss.avg,val_acc.avg,time_taken))

            if args.tensorboard:
                log_value('val_loss', val_loss.avg, epoch)
                log_value('val_acc', val_acc.avg, epoch)
        

            is_best = (val_acc.avg >= best_acc)
            best_acc = max(val_acc.avg,best_acc)

            print ('Current best accuracy (error):', best_acc )   # saving the best model 
            save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'best_acc': best_acc}, is_best)


    print ('Best accuracy (error): ', best_acc)
    

    




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.expname) + 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    main()


















            


    







        




