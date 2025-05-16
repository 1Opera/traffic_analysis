from pathlib import Path
import click
import numpy as np
import pandas as pd
import gzip
import json
from joblib import Parallel, delayed
import dpkt
import socket
from scipy.sparse import csr_matrix
from utils import should_omit_packet, read_pcap, PREFIX_TO_APP_ID, PREFIX_TO_TRAFFIC_ID

def remove_ether_header(packet):
    """移除数据包的以太网头部，以获取内部的IP包等信息。

    Args:
        packet (dpkt.ethernet.Ethernet): 待处理的数据包。

    Returns:
        packet (dpkt.ethernet.Ethernet): 移除以太网头部后的数据包。
    """
    if isinstance(packet, dpkt.ethernet.Ethernet):
        return packet.data
    return packet

def mask_ip(packet):
    """将IP数据包中的源IP地址和目的IP地址掩码为0.0.0.0，以便去除敏感信息。

    Args:
        packet (dpkt.ip.IP): IP数据包。

    Returns:
        packet (dpkt.ip.IP): 掩码后的IP数据包。
    """
    if isinstance(packet, dpkt.ip.IP):
        packet.src = b'\x00\x00\x00\x00'
        packet.dst = b'\x00\x00\x00\x00'
    return packet

def pad_udp(packet):
    """为UDP数据包添加固定长度的填充，保证数据包长度一致。

    Args:
        packet (dpkt.udp.UDP): UDP数据包。

    Returns:
        packet (dpkt.udp.UDP): 填充后的UDP数据包。
    """
    if isinstance(packet, dpkt.udp.UDP):
        udp = packet
        original_data = udp.data
        udp.data = (b'\x00' * 12) + original_data
    return packet

def packet_to_sparse_array(packet, max_length=1500):
    """将数据包转换为稀疏矩阵格式，方便进行机器学习处理。

    Args:
        packet (bytes): 数据包的字节序列。
        max_length (int): 稀疏矩阵的最大长度，默认1500。

    Returns:
        scipy.sparse.csr_matrix: 稀疏矩阵表示的数据包。
    """
    bytes_packet = bytes(packet)
    arr = np.frombuffer(bytes_packet, dtype=np.uint8)[:max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    arr = csr_matrix(arr)
    return arr

def transform_packet(packet):
    """对数据包执行一系列转换，包括移除以太网头部、掩码IP、填充UDP，并转换为稀疏矩阵。

    Args:
        packet: 原始数据包。

    Returns:
        scipy.sparse.csr_matrix: 转换后的稀疏矩阵，若数据包应被忽略，则返回None。
    """
    if should_omit_packet(packet):
        return None
    packet = remove_ether_header(packet)
    packet = mask_ip(packet)
    packet = pad_udp(packet)
    arr = packet_to_sparse_array(packet)
    return arr

def transform_pcap(path, output_path: Path = None, output_batch_size=10000):
    """处理单个PCAP文件，将数据包转换为特征，并存储到压缩的JSON文件中。

    Args:
        path (Path): PCAP文件的路径。
        output_path (Path): 转换后数据的输出路径。
        output_batch_size (int): 每批处理的数据包数量，默认10000。
    """
    if Path(str(output_path.absolute()) + "_SUCCESS").exists():
        print(output_path, "Done")
        return
    print("Processing", path)
    rows = []
    batch_index = 0
    for i, packet in enumerate(read_pcap(path)):
        arr = transform_packet(packet)
        if arr is not None:
            #prefix = path.name.split(".")[0].lower()
            prefix = path.stem.split(".")[0].lower()
            app_label = PREFIX_TO_APP_ID.get(prefix)
            traffic_label = PREFIX_TO_TRAFFIC_ID.get(prefix)
            row = {
                "app_label": app_label,
                "traffic_label": traffic_label,
                "feature": arr.todense().tolist()[0],
            }
            rows.append(row)
        if rows and i > 0 and i % output_batch_size == 0:
            part_output_path = Path(str(output_path.absolute()) + f"_part_{batch_index:04d}.json.gz")
            with part_output_path.open("wb") as f, gzip.open(f, "wt") as f_out:
                for row in rows:
                    f_out.write(f"{json.dumps(row)}\n")
            batch_index += 1
            rows.clear()
    if rows:
        part_output_path = Path(str(output_path.absolute()) + f"_part_{batch_index:04d}.json.gz")
        with part_output_path.open("wb") as f, gzip.open(f, "wt") as f_out:
            for row in rows:
                f_out.write(f"{json.dumps(row)}\n")
    with Path(str(output_path.absolute()) + "_SUCCESS").open("w") as f:
        f.write("")
    print(output_path, "Done")
@click.command()
@click.option(
    "-s",
    "--source",
    help="path to the directory containing raw pcap files",
    required=True,
)
@click.option(
    "-t",
    "--target",
    help="path to the directory for persisting preprocessed files",
    required=True,
)
@click.option("-n", "--njob", default=1, help="num of executors", type=int)
def main(source, target, njob):
    """主函数用于处理PCAP文件的批量转换。此函数根据给定的源目录和目标目录，
    对源目录中的所有PCAP文件执行转换操作，并将结果存储在目标目录中。

    Args:
        source (str): 包含原始PCAP文件的源目录路径。
        target (str): 用于存储转换后文件的目标目录路径。
        njob (int): 指定并行执行任务的数量。如果njob为1，则不使用并行处理。
    """
    data_dir_path = Path(source)# 创建Path对象，表示源目录
    target_dir_path = Path(target)# 创建Path对象，表示目标目录
    target_dir_path.mkdir(parents=True, exist_ok=True) # 确保目标目录存在，如果不存在则创建
    if njob == 1:# 如果不使用并行处理，逐个处理每个PCAP文件
        for pcap_path in sorted(data_dir_path.iterdir()):
            transform_pcap(
                pcap_path, target_dir_path / (pcap_path.name + ".transformed")
            )
    else: 
        # 使用joblib的Parallel来并行处理PCAP文件
        Parallel(n_jobs=njob)(
            delayed(transform_pcap)(
                pcap_path, target_dir_path / (pcap_path.name + ".transformed")
            )
            for pcap_path in sorted(data_dir_path.iterdir())
        )
if __name__ == "__main__":
    main()
