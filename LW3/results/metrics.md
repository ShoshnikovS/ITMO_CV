Сводка: {"resnet18": {"val_acc": 0.928, "val_f1": 0.927, "test_acc": 0.883, "test_f1": 0.881}, "small_cnn": {"val_acc": 0.815, "val_f1": 0.812, "test_acc": 0.667, "test_f1": 0.681}}

resnet18 / val: accuracy=0.928, macro-F1=0.927
| класс | precision | recall | F1 | число |
|---|---|---|---|---|
| daisy | 0.96 | 0.94 | 0.95 | 127 |
| dandelion | 0.97 | 0.95 | 0.96 | 180 |
| roses | 0.88 | 0.91 | 0.89 | 129 |
| sunflowers | 0.94 | 0.96 | 0.95 | 140 |
| tulips | 0.89 | 0.89 | 0.89 | 160 |

resnet18 / test: accuracy=0.883, macro-F1=0.881
| класс | precision | recall | F1 | число |
|---|---|---|---|---|
| daisy | 0.87 | 1.00 | 0.93 | 13 |
| dandelion | 0.93 | 1.00 | 0.97 | 14 |
| roses | 0.90 | 0.90 | 0.90 | 10 |
| sunflowers | 0.89 | 0.89 | 0.89 | 9 |
| tulips | 0.82 | 0.64 | 0.72 | 14 |

small_cnn / val: accuracy=0.815, macro-F1=0.812
| класс | precision | recall | F1 | число |
|---|---|---|---|---|
| daisy | 0.88 | 0.77 | 0.82 | 127 |
| dandelion | 0.80 | 0.91 | 0.85 | 180 |
| roses | 0.75 | 0.68 | 0.72 | 129 |
| sunflowers | 0.91 | 0.88 | 0.89 | 140 |
| tulips | 0.76 | 0.80 | 0.78 | 160 |

small_cnn / test: accuracy=0.667, macro-F1=0.681
| класс | precision | recall | F1 | число |
|---|---|---|---|---|
| daisy | 1.00 | 0.92 | 0.96 | 13 |
| dandelion | 0.56 | 0.36 | 0.43 | 14 |
| roses | 1.00 | 0.80 | 0.89 | 10 |
| sunflowers | 0.35 | 0.78 | 0.48 | 9 |
| tulips | 0.73 | 0.57 | 0.64 | 14 |

Ошибок resnet18 на test: 7 из 60