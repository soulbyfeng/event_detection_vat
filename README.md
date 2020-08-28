
#### **事件识别 使用bert预训练模型 结合vat 实现半监督事件检测**

一。bert预训练作用：
1.分训练数据、测试数据
2.使用随机词向量进行训练，在测试集上获得各项指标
3.使用bert预训练词向量 训练，在测试集上获得各项指标
4.使用glove预训练词向量 训练，在测试集上获得各项指标
5.对步骤2，3，4进行指标分析，得到最好指标的预训练模型

使用随机词向量训练
    python main_ed.py --use_vat False -emb_dim 256  --output output/baseline.bin -embed_type baseline --use_cuda True --num_epochs 50 --batch_size 256

使用bert词向量训练
python main_ed.py --use_vat False -emb_dim 768  --output output/bert_baseline.bin -embed_type bert --use_cuda True --num_epochs 50 --batch_size 256

使用glove词向量训练
python main_ed.py --use_vat False -emb_dim 300  --output output/glove_baseline.bin -embed_type glove --use_cuda True --num_epochs 50 --batch_size 256

每个指令执行完会走一遍验证集测试，可以直接看到测试效果，方便对比

二。vat作用 ：
1.分训练数据、测试数据、无标签数据
2.使用训练数据进行对比模型训练 ，用测试集测试准确率召回率f1指标   该步骤已经在一中执行，这里可以无需执行
3.使用训练数据+无标签数据进行vat+ed训练，用测试集验证对比和步骤2指标差距，凸显vat的作用


vat 对比试验 参考论文 ed without triggers
`python main_ed.py --use_vat False  --output output/baseline.bin --use_cuda False --num_epochs 50 --batch_size 256`
注： 此步骤已经在步骤一中执行这里可以不执行

vat 训练
`python main_ed.py --use_vat True  --output output/bert_vat.bin --use_cuda False --num_epochs 50 --batch_size 256`

如果无法执行，直接点击执行 调整 main_ed.py 中的参数 后直接run即可
