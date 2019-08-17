import os
from utils.metrics.metrics import evaluate_results
from configs.rrnet_config import Config
from operators.test.rrnet_operator import RRNetTestOperator
from operators.test.ctnet_operator import CTNetTestOperator
from datasets import make_val_dataloader


if __name__ == "__main__":
    print("Perform testing ...")
    OP = RRNetTestOperator if Config.Model.use_rr else CTNetTestOperator
    op = OP(Config)

    dataloader = make_val_dataloader(Config)
    op.evaluation_process(dataloader)

    result_dir = Config.result_dir
    gt_dir = os.path.join(Config.data_root, 'val_data', 'annotations')
    evaluate_results(result_dir, gt_dir)
