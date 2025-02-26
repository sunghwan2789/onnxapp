#!/usr/bin/env python
# -*- coding: utf-8 -*-

from onnx import checker, helper, save, TensorProto

input_X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
output_Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
output_Y1 = helper.make_tensor_value_info('Y1', TensorProto.FLOAT, [None, None])
output_Y2 = helper.make_tensor_value_info('Y2', TensorProto.FLOAT, [None, None])
output_Y3 = helper.make_tensor_value_info('Y3', TensorProto.FLOAT, [None, None])
output_Y4 = helper.make_tensor_value_info('Y4', TensorProto.FLOAT, [None, None])
output_Y5 = helper.make_tensor_value_info('Y5', TensorProto.FLOAT, [None, None])
output_Y6 = helper.make_tensor_value_info('Y6', TensorProto.FLOAT, [None, None])
output_Y7 = helper.make_tensor_value_info('Y7', TensorProto.FLOAT, [None, None])
output_Y8 = helper.make_tensor_value_info('Y8', TensorProto.FLOAT, [None, None])
output_Y9 = helper.make_tensor_value_info('Y9', TensorProto.FLOAT, [None, None])


graph = helper.make_graph(
    [
        helper.make_node('MatMul', ['X', 'X'], ['Y']),
        helper.make_node('MatMul', ['Y', 'Y'], ['Y1']),
        helper.make_node('MatMul', ['Y1', 'Y1'], ['Y2']),
        helper.make_node('MatMul', ['Y2', 'Y2'], ['Y3']),
        helper.make_node('MatMul', ['Y3', 'Y3'], ['Y4']),
        helper.make_node('MatMul', ['Y4', 'Y4'], ['Y5']),
        helper.make_node('MatMul', ['Y5', 'Y5'], ['Y6']),
        helper.make_node('MatMul', ['Y6', 'Y6'], ['Y7']),
        helper.make_node('MatMul', ['Y7', 'Y7'], ['Y8']),
        helper.make_node('MatMul', ['Y8', 'Y8'], ['Y9']),
        # helper.make_node('Slice', ['Y9'], ['Yo'], axes=[0], ends=[1], starts=[0]),
    ],
    'MatMul',
    [input_X],
    [output_Y9],
    # [output_Y],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
checker.check_model(model)

save(model, 'matmul.onnx')
