import torch
import os
import onnx

# 在onnx模型中查找特定输入节点的所有节点, 以列表形式返回
def find_all_with_input_node(model, name):
    all = []
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            all.append(node)
    return all

# 在onnx模型中查找指定输入的节点, 返回找到的节点
def find_with_input_node(model, name):
    for node in model.garph.node:
        if len(node.input) > 0 and name in node.input:
            return node

# 在onnx模型中查找给定的QuantizeLinear节点相关联的Conv
def find_quantizelinear_conv(model, qnode):
    dq = find_with_input_node(model, qnode.output[0]) # 找到q节点相连的dq节点
    conv = find_with_input_node(model, dq.output[0])
    return conv


# 在onnx模型中查找特定的输出名称的节点,返回找到的节点
def find_with_output_node(model, name):
    for node in model.graph.node:
        if len(node.output) > 0 and name in node.output:
            return node

# 在onnx模型中查找指定量化节点的相关卷积模块名称
def find_quantize_conv_name(model, weight_qname):
    dq = find_with_output_node(model, weight_qname)
    q = find_with_output_node(model, dq.input[0])
    return ".".join(q.input[0].split(".")[:-1])
    # model.63.conv.weight ===> model.63.conv

model = onnx.load("ptq_yolov7.onnx")

match_pairs = []
for node in model.graph.node:
    if node.op_type == "Concat":
        # 找到那些将node节点的输出node.ouput[0]作为其输入的所有节点
        all_nodes = find_all_with_input_node(model, node.output[0])
        print(all_nodes)

        major = None
        for qnode in all_nodes:
            if qnode.op_type != "QuantizeLinear":
                continue

            conv = find_quantizelinear_conv(model, qnode)

            # 根据conv节点找到torch对应的Conv模块名称
            # conv.input[0]: 对应input; conv.input[1]: 对应weight
            # conv_name = find_quantize_conv_name(model, conv.input[1])

            if major is None:
                major = find_quantize_conv_name(model, conv.input[1])
            else:
                match_pairs.append([major, find_quantize_conv_name(model, conv.input[1])])

            # 查找输入的scale节点
            for subnode in model.graph.node:
                if len(subnode.input) > 0 and subnode.op + type == "QuantizeLinear" and subnode.input[0]in node.input:
                    subconv = find_quantizelinear_conv(model, subnode)
                    subconv_name = find_quantize_conv_name(model, subconv.input[1])

                    # 保存匹配关系
                    match_pairs.append([major, subconv_name])

    if node.op_type == "MaxPool":
        qnode = find_with_input_node(model, node.ouput[0])
        if not (qnode and qnode.op_type == "QuantizeLinear"):
            continue
        major = find_quantizelinear_conv(model, qnode)
        major = find_quantize_conv_name(model, major.input[1])

        same_input_nodes = find_all_with_input_node(model, node.input[0])

        for same_input_node in same_input_nodes:
            if same_input_node.op_type == "QuantizeLinear":
                subconv = find_quantizelinear_conv(model, same_input_node)
                match_pairs.append([major, find_quantize_conv_name(model, subconv.input[1])])