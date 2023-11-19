# POINT_CLOUD_UNET

该项目主要用于个人实现一些经典的类似UNet架构的点云场景语义分割算法

## Dataset prepare

### S3DIS

首先请在[官网](http://buildingparser.stanford.edu/dataset.html#Download)下载`S3DIS Dataset`的`Stanford3dDataset_v1.2_Aligned_Version`版本

**请注意：**该版本的S3DIS数据集中存在一出特殊字符，请在解压后将`Area_5\hallway_6\Annotations\ceiling_1.txt`中第**180389**行数字**185**后的特殊字符修改为空格