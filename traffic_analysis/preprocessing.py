from pathlib import Path
import numpy as np
import pandas as pd
import gzip
import json
from joblib import Parallel, delayed
from scapy.compat import raw
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
import json
from utils import should_omit_packet, read_pcap, PREFIX_TO_APP_ID, PREFIX_TO_TRAFFIC_ID
import ast
import os
from utils import load_config
import argparse

def get_labels_from_path(path):
    """
    从文件名提取前缀，应用标签和流量标签
    """
    prefix = path.name.split(".")[0].lower()
    app_label = PREFIX_TO_APP_ID.get(prefix)
    traffic_label = PREFIX_TO_TRAFFIC_ID.get(prefix)
    return prefix,app_label, traffic_label

def remove_ether_header(packet):
    """移除数据包的以太网头部，以获取内部的IP包等信息。

    Args:
        packet (dpkt.ethernet.Ethernet): 待处理的数据包。

    Returns:
        packet (dpkt.ethernet.Ethernet): 移除以太网头部后的数据包。
    """
    if Ether in packet:
        return packet[Ether].payload# 提取以太网头部之后的有效载荷

    return packet

def mask_ip(packet):
    """将IP数据包中的源IP地址和目的IP地址掩码为0.0.0.0，以便去除敏感信息。

    Args:
        packet (dpkt.ip.IP): IP数据包。

    Returns:
        packet (dpkt.ip.IP): 掩码后的IP数据包。
    """
    # 检查数据包是否为IP包，如果是，则将源IP和目标IP设置为0.0.0.0
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"

    return packet

def pad_udp(packet):
    """为UDP数据包添加固定长度的填充，保证数据包长度一致。

    Args:
        packet (dpkt.udp.UDP): UDP数据包。

    Returns:
        packet (dpkt.udp.UDP): 填充后的UDP数据包。
    """
    if UDP in packet:
        # 获取UDP包之后的所有层
        layer_after = packet[UDP].payload.copy()

        # 构建填充层，填充12个字节
        pad = Padding()
        pad.load = "\x00" * 12

        # 去掉原始UDP包的有效载荷，添加填充层后重新组合
        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet

def packet_to_sparse_array(packet, max_length=1500):
    """将数据包转换为稀疏矩阵格式，方便进行机器学习处理。

    Args:
        packet (bytes): 数据包的字节序列。
        max_length (int): 稀疏矩阵的最大长度，默认1500。

    Returns:
        scipy.sparse.csr_matrix: 稀疏矩阵表示的数据包。
    """
    # 将包转换为二进制数组
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length] / 255
    # 如果数据包长度小于max_length，则进行填充
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        # 用0填充剩余部分
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    # 转换为稀疏矩阵
    arr = sparse.csr_matrix(arr)
    return arr

def load_poison_records(file_path):
    """加载指定文本文件中的投毒记录，并打印每行数据及其索引"""
    records = []
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            stripped_line = line.strip()
            if stripped_line:  # 确保记录非空
                records.append(stripped_line)
            #print(f"Loaded line {index}: '{stripped_line}'")  # 打印加载的每行数据
    return records

def transform_packet(packet):
    """
    处理单个数据包，进行必要的转换和处理
    """
    # 判断数据包是否应被忽略
    if should_omit_packet(packet):
        return None
    # 依次进行去除Ethernet头部、填充UDP包、掩码IP等处理
    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    # 转换为稀疏矩阵
    arr = packet_to_sparse_array(packet)

    return arr
def write_json_gzip(rows, output_path):
    try:
        # 确保输出路径的文件夹存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试以gzip格式打开文件并写入
        with open(output_path, 'wb') as f:
            with gzip.open(f, 'wt', encoding='utf-8') as gz_out:
                for row in rows:
                    gz_out.write(f"{json.dumps(row)}\n")

        # 检查文件是否为gzip格式
        with open(output_path, 'rb') as f:
            try:
                with gzip.open(f, 'rt') as test_gzip:
                    test_gzip.read(1)  # 尝试读取文件中的一些数据
                print(f"File {output_path} written successfully as gzip format.")
            except Exception as e:
                print(f"File {output_path} is not in gzip format. Error: {e}")

    except Exception as e:
        print(f"Error writing gzip file {output_path}: {e}")

def transform_pcap(path,records_dir,output_path: Path = None, output_batch_size=10000):
    """
    处理整个PCAP文件，将数据包转换并保存为JSON格式
    """
    # 检查是否已处理过该文件
    if Path(str(output_path.absolute()) + "_SUCCESS").exists():
        print(output_path, "Done")
        return

    print("Processing", path)
    rows = []
    batch_index = 0
    prefix,app_label, traffic_label=get_labels_from_path(path)
    records_path = Path(records_dir) / f"{path.stem}_state_info.txt"
    if records_path.exists():
        omitted_indices = []
        poisoned_indices = set()  # 用于存储投毒记录里中毒的索引
        # 加载投毒记录文件（如果存在），解析出中毒的索引
        if records_path.exists():
            with open(records_path, 'r') as file:
                for idx, line in enumerate(file):
                    try:
                        record = ast.literal_eval(line.strip())
                        if record.get("Is Poisoned"):
                            poisoned_indices.add(idx)  # 添加被投毒的索引
                    except Exception as e:
                        print(f"Error parsing line in {records_path}: {line.strip()}. Error: {e}")

        # 遍历PCAP文件中的每个包，进行处理
        for i, packet in enumerate(read_pcap(path)):
            arr = transform_packet(packet)
            if arr is not None:
                is_poisoned = (i in poisoned_indices)  # 检查当前索引是否在投毒记录中
                row = {
                    "r_id":i,
                    "app_label": app_label,
                    "traffic_label": traffic_label,
                    "feature": arr.todense().tolist()[0],
                    "is_poisoned": is_poisoned,
                }
                rows.append(row)
            else:
                omitted_indices.append(i)

             # 每处理batch_size个包就写一次输出文件
            if rows and i > 0 and i % output_batch_size == 0:
                part_output_path = Path(str(output_path.absolute()) + f"_part_{batch_index:04d}.json.gz")
                write_json_gzip(rows, part_output_path)

                batch_index += 1
                rows.clear()

    else:
        # 没有投毒记录的情况
        for i, packet in enumerate(read_pcap(path)):
            arr = transform_packet(packet)
            if arr is not None:
                row = {
                    "r_id":i,
                    "app_label": app_label,
                    "traffic_label": traffic_label,
                    "feature": arr.todense().tolist()[0],
                    "is_poisoned": False,#没有投毒记录的文件
                }
                rows.append(row)
            
            # 每处理batch_size个包就写一次输出文件
            if rows and i > 0 and i % output_batch_size == 0:
                part_output_path = Path(str(output_path.absolute()) + f"_part_{batch_index:04d}.json.gz")
                write_json_gzip(rows, part_output_path)

                batch_index += 1

                rows.clear()

    # 最后一次处理剩余数据
    if rows:
        part_output_path = Path(
            str(output_path.absolute()) + f"_part_{batch_index:04d}.json.gz"
        )
        write_json_gzip(rows, part_output_path)


    # 写入处理成功的标记文件
    with Path(str(output_path.absolute()) + "_SUCCESS").open("w") as f:
        f.write("")

    print(output_path, "Done")
def main():
    parser = argparse.ArgumentParser(description='运行中毒训练集相关操作，需指定配置文件路径')
    parser.add_argument('config_path', type=str, help='配置文件的路径')
    args = parser.parse_args()

    config = load_config(args.config_path)  # 加载配置
    print(config['preprocessing'])

    task=config['preprocessing']['task'] # 任务类型，可以是'app'或'traffic'
    target_label=config['preprocessing']['train_target_label']
    njob=config['preprocessing']['njob'] # 并行处理的任务数量
    dataset_folder=config['preprocessing']['dataset_folder']
    poison_dir=config['preprocessing']['poison_dir'] #中毒数据的上层文件夹，方便管理
    results_dir=config['preprocessing']['results_dir']   #结果文件夹
    
    source = f"{poison_dir}/poisoned_{dataset_folder}_{task}"
    target = f"{poison_dir}/processed_poisoned_{dataset_folder}_{task}"
    records_dir = f"{source}_{target_label}"

    data_dir_path = Path(source)
    target_dir_path = Path(target)
    records_dir=Path(records_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # 如果是单进程运行，顺序处理文件
    if njob == 1:
        for pcap_path in sorted(data_dir_path.iterdir()):
            transform_pcap(
                pcap_path,records_dir,target_dir_path / (pcap_path.name + ".transformed")
            )
    else:
        # 启动多进程并行处理
        Parallel(n_jobs=njob)(
            delayed(transform_pcap)(
                pcap_path,records_dir,target_dir_path / (pcap_path.name + ".transformed")
            )
            for pcap_path in sorted(data_dir_path.iterdir())
        )

# 程序入口
if __name__ == "__main__":
    main()
