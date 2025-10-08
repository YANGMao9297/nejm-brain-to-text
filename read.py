import h5py
import numpy as np
import os

# 文件路径
file_path = r"brain-to-text-25\t15_copyTask_neuralData\hdf5_data_final\t15.2023.08.11\data_train.hdf5"

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误: 文件不存在 - {file_path}")
else:
    print(f"正在读取文件: {file_path}\n")
    
    # 打开 HDF5 文件
    with h5py.File(file_path, 'r') as f:
        print("=" * 60)
        print("HDF5 文件结构:")
        print("=" * 60)
        
        # 递归函数来显示 HDF5 文件的结构
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                if obj.shape == () or (len(obj.shape) == 1 and obj.shape[0] <= 10):
                    print(f"  Value: {obj[()]}")
                print()
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                print()
        
        # 遍历文件中的所有对象
        f.visititems(print_structure)
        
        print("=" * 60)
        print("顶层键值:")
        print("=" * 60)
        for key in f.keys():
            print(f"- {key}")
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"  类型: Dataset, Shape: {item.shape}, Dtype: {item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"  类型: Group")
        
        print("\n" + "=" * 60)
        print("示例: 读取具体数据")
        print("=" * 60)
        
        # 示例：如果你想读取某个特定的数据集，可以这样做：
        # 这里我们先列出所有键，你可以根据需要修改
        if len(f.keys()) > 0:
            first_key = list(f.keys())[0]
            print(f"\n读取第一个键 '{first_key}' 的数据:")
            data = f[first_key][()]
            print(f"数据类型: {type(data)}")
            print(f"数据形状: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            if hasattr(data, 'shape') and len(data.shape) <= 2 and np.prod(data.shape) <= 100:
                print(f"数据内容:\n{data}")
            else:
                print(f"数据太大，只显示前几个元素...")
                if hasattr(data, 'shape') and len(data.shape) > 0:
                    print(data.flat[:10] if np.prod(data.shape) > 10 else data)

print("\n完成!")
