from configs.rrnet_config import Config
from operators.test.rrnet_operator import RRNetTestOperator
from operators.test.ctnet_operator import CTNetTestOperator
from datasets import make_test_dataloader


if __name__ == "__main__":
    OP = RRNetTestOperator if Config.Model.use_rr else CTNetTestOperator
    op = OP(Config)

    dataloader = make_test_dataloader(Config)

    print("=> Perform inference ...")
    op.evaluation_process(dataloader)
    print("=> Done!")
