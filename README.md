# Datamining-homework4
The last week homework for dataming

## Datasets 
1. wine_benchmarks.zip
2. skin_benchmarks.zip

## Main process .py files
1. /wine/wine.py
2. /skin/skin.py

## Process results
> wine
1. /wine/wineDiffResults.txt
2. /wine/wineBenchmarkResults.txt
> skin
1. /skin/skinDiffResults.txt
2. /skin/skinBenchmarkResults.txt  

## Analysis reports
> wine
1. /wine/wineDiffShow.ipynb
2. /wine/wineBenchmarkShow.ipynb
> skin
1. /skin/skinDiffShow.ipynb
2. /skin/skinBenchmarkShow.ipynb

## Analysis reports html
```
预分析说明：
在数据集的 meta_data 中，有多个csv文件：
其中，`wine.original.csv`应该是最原始的数据，`wine.diff.csv`是经过预处理的全数据集。
通过对 wine 数据集的初步观察，数据中`quality`低于 5 分（包括5分）的酒为 anomaly 类，高于 5 分的酒为 nomal 类
且其它各项数据经过归一化处理，所有的batchmark数据csv文件中的数据格式与`wine.diff.csv`相同
因此，我们使用`wine.diff.csv`文件抽样来训练一个离群点检测器，用于检测每个benchmark中的离群点信息
之后，我们对于每个benchmark进行采样，通过采样数据训练benchmark独立的离群点检测器，用于检测单个benchmark中的自离群点信息
skin 数据集也做类似处理
```
> wine
1. [/wine/wineDiffShow.html](https://bingfenghan-bit.github.io/Datamining-homework4/wine/wineDiffShow.html)
2. [/wine/wineBenchmarkShow.html](https://bingfenghan-bit.github.io/Datamining-homework4/wine/wineBenchmarkShow.html)
> skin
1. [/skin/skinDiffShow.html](https://bingfenghan-bit.github.io/Datamining-homework4/skin/skinDiffShow.html)
2. [/skin/skinBenchmarkShow.html](https://bingfenghan-bit.github.io/Datamining-homework4/skin/skinBenchmarkShow.html)