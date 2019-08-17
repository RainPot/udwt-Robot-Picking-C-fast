from configs.rrnet_config import Config
from operators.distributed_wrapper import DistributedWrapper
from operators.train.rrnet_operator import RRNetTrainOperator
from operators.train.ctnet_operator import CTNetTrainOperator


if __name__ == '__main__':
    op = RRNetTrainOperator if Config.Model.use_rr else CTNetTrainOperator
    dis_operator = DistributedWrapper(Config, op)
    dis_operator.train()
    print('Training is Done!')
