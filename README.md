## update

20220306: 写到方向分类模型的数据集构建 cls/datasets/baseset.py
20220309: 完成数据预处理的可扩展
20220313: 完成cls的训练，添加val_step和metric, 需要前几个step的可视化， 需要搞清楚validation_step在多gpu时的情况
20220318: validation_step 和 validation_epoch_end 在多gpu时，也是每个gpu都会跑的，现在在val的时候metric取了各个gpu的平均
20220319: validation_step_end 在多gpu时会收集各个gpu的结果。可以得到准确的val