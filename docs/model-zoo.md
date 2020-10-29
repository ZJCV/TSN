
# Model Zoo

## [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-jqnz{background-color:#F3F6F6;color:#404040;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-rt64{background-color:#F3F6F6;color:#9B59B6;text-align:center;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj"><span style="font-weight:bold">config</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">resolution(TxHxW)</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">gpus</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">backbone</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">pretrain</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">top1 acc</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">top5 acc</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">testing protocol</span></th>
    <th class="tg-wa1i"><span style="font-weight:bold">inference_time(video/s)</span></th>
    <th class="tg-wa1i"><span style="font-weight:bold">gpu_mem(M)</span></th>
    <th class="tg-wa1i"><span style="font-weight:bold">ckpt</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-rt64"><a href="https://cloud.zhujian.tech:9300/s/MwAMXHsXQgAZRwD" target="_blank" rel="noopener noreferrer">tsn_r50_ucf101_rgb_raw_dense_1x16x4</a></td>
    <td class="tg-jqnz">4x256x256</td>
    <td class="tg-jqnz">2</td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">tsn</span></td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">ImageNet</span></td>
    <td class="tg-jqnz">80.881</td>
    <td class="tg-jqnz"><span style="font-weight:400;font-style:normal">95.48</span></td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">1 clips x 1 crop</span></td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">x</span></td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">x</span></td>
    <td class="tg-rt64"><a href="https://cloud.zhujian.tech:9300/s/ZKXim94beK4a9EJ">ckpt</a></td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td class="tg-0wh7"><a href="https://cloud.zhujian.tech:9300/s/bY7jRPpAD9mKqkW" target="_blank" rel="noopener noreferrer">tsn_r50_ucf101_rgb_raw_seg_1x1x3</a></td>
    <td class="tg-tiqg">4x256x256</td>
    <td class="tg-tiqg">2</td>
    <td class="tg-tiqg"><span style="background-color:#F3F6F6">tsn</span></td>
    <td class="tg-tiqg"><span style="background-color:#F3F6F6">imagenet</span></td>
    <td class="tg-tiqg">81.589</td>
    <td class="tg-tiqg"><span style="font-weight:400;font-style:normal">95.964</span></td>
    <td class="tg-tiqg"><span style="background-color:#F3F6F6">1 clips x 1 crop</span></td>
    <td class="tg-tiqg"><span style="background-color:#F3F6F6">x</span></td>
    <td class="tg-tiqg">x</td>
    <td class="tg-0wh7"><a href="https://cloud.zhujian.tech:9300/s/xqbSpLFcQkJADbz" target="_blank" rel="noopener noreferrer">ckpt</a></td>
  </tr>
</tbody>
</table>
