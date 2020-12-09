
# 命名规范

## 配置文件

```
<core algorithm>-<backbone>-<dataset>-<modality>-<data format>-<sample strategy>-<clip_len>x<frame_interval>x<num_clips>
```

## 示例一

配置文件`tsn_r50_ucf101_rgb_raw_dense_1x16x4.yaml`

* `core algorithm: tsn`
* `backbone: r50(resnet_50)`
* `dataset: ucf101`
* `modality: rgb`
* `data format: raw(raw frame)`
* `sample strategy: dense(dense sample)`
* `clip_len: 1`
* `frame_interval: 16`
* `num_clips: 4`

##　示例二

配置文件`tsn_r50_pretrained_fix_bn_ucf101_rgb_raw_dense_1x16x4.yaml`

* `core algorithm: tsn`
* `backbone: r50_pretrained_fix_bn`
* `dataset: ucf101`
* `modality: rgb`
* `data format: raw(raw frame)`
* `sample strategy: dense(dense sample)`
* `clip_len: 1`
* `frame_interval: 16`
* `num_clips: 4`