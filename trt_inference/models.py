# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import OrderedDict
from copy import deepcopy
from diffusers.models import AutoencoderKL, UNet2DConditionModel
import numpy as np
from onnx import shape_inference
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from cuda import cudart
import onnx
import os


class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix=''):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def remove_casts(self):
        nRemoveCastNode = 0
        for node in self.graph.nodes:
            # Remove Cast nodes before qkv gemm
            if node.op in ["Add", "Transpose"] and len(
                node.outputs[0].outputs) == 3 and node.o().op == "Cast" and node.o(1).op == "Cast" and node.o(
                2).op == "Cast":
                for i in range(len(node.outputs[0].outputs)):
                    matMulNode = node.o(i, 0).o()
                    matMulNode.inputs[0] = node.outputs[0]
                    nRemoveCastNode += 1

            # Remove double cast nodes after Softmax Node
            if node.op == "Softmax" and node.o().op == "Cast" and node.o().o().op == "Cast":
                node.o().o().o().inputs[0] = node.outputs[0]
                nRemoveCastNode += 1

        self.cleanup()
        return nRemoveCastNode

    def remove_parallel_swish(self):
        mRemoveSwishNode = 0
        for node in self.graph.nodes:
            if node.op == "Gemm" and len(node.outputs[0].outputs) > 6:
                swishOutputTensor = None
                for nextNode in node.outputs[0].outputs:
                    if nextNode.op == "Mul":
                        if swishOutputTensor is None:
                            swishOutputTensor = nextNode.outputs[0]
                        else:
                            nextGemmNode = nextNode.o(0)
                            assert nextGemmNode.op == "Gemm", "Unexpected node type for nextGemmNode {}".format(
                                nextGemmNode.name)
                            nextGemmNode.inputs = [swishOutputTensor, nextGemmNode.inputs[1], nextGemmNode.inputs[2]]
                            nextNode.outputs.clear()
                            mRemoveSwishNode += 1

        self.cleanup()
        return mRemoveSwishNode

    def clip_cast_fix(self):
        '''
        Clip is executed in fp32 sometimes on windows, but it has fp16 inputs. It leads to an error.
        In this fix we insert cast to fp16 before Clip.
        It is a WAR to make it work on Windows, while we are looking for a better solution.
        '''
        mCastInserted = 0
        for node in self.graph.nodes:
            if node.op == "Clip":
                # Create cast output variable
                cast_out = gs.Variable(name="Cast_out_{}".format(mCastInserted), dtype=np.float16)
                # Create cast output variable
                cast = gs.Node(op="Cast", name="Cast_{}".format(mCastInserted), attrs={"to": onnx.TensorProto.FLOAT16},
                               inputs=[node.i().outputs[0]], outputs=[cast_out])
                node.inputs[0] = cast_out
                self.graph.nodes.append(cast)
                mCastInserted += 1

        self.cleanup()
        return mCastInserted

    def resize_fix(self):
        '''
        This function loops through the graph looking for Resize nodes that uses scales for resize (has 3 inputs).
        It substitutes found Resize with Resize that takes the size of the output tensor instead of scales.
        It adds Shape->Slice->Concat
                Shape->Slice----^     subgraph to the graph to extract the shape of the output tensor.
        This fix is required for the dynamic shape support.
        '''
        mResizeNodes = 0
        for node in self.graph.nodes.copy():
            if node.op == "Resize" and len(node.inputs) == 3:
                name = node.name + "/"

                add_node = node.o().o().i(1)
                # temp hack
                if add_node.op != "Add":
                    add_node = node.o().o().o().o().o().o().i(1)
                    assert add_node.op == "Add"
                div_node = node.i()

                shape_hw_out = gs.Variable(name=name + "shape_hw_out", dtype=np.int64, shape=[4])
                shape_hw = gs.Node(op="Shape", name=name + "shape_hw", inputs=[add_node.outputs[0]],
                                   outputs=[shape_hw_out])

                const_zero = gs.Constant(name=name + "const_zero", values=np.array([0], dtype=np.int64))
                const_two = gs.Constant(name=name + "const_two", values=np.array([2], dtype=np.int64))
                const_four = gs.Constant(name=name + "const_four", values=np.array([4], dtype=np.int64))

                slice_hw_out = gs.Variable(name=name + "slice_hw_out", dtype=np.int64, shape=[2])
                slice_hw = gs.Node(op="Slice", name=name + "slice_hw",
                                   inputs=[shape_hw_out, const_two, const_four, const_zero], outputs=[slice_hw_out])

                shape_bc_out = gs.Variable(name=name + "shape_bc_out", dtype=np.int64, shape=[2])
                shape_bc = gs.Node(op="Shape", name=name + "shape_bc", inputs=[div_node.outputs[0]],
                                   outputs=[shape_bc_out])

                slice_bc_out = gs.Variable(name=name + "slice_bc_out", dtype=np.int64, shape=[2])
                slice_bc = gs.Node(op="Slice", name=name + "slice_bc",
                                   inputs=[shape_bc_out, const_zero, const_two, const_zero], outputs=[slice_bc_out])

                concat_bchw_out = gs.Variable(name=name + "concat_bchw_out", dtype=np.int64, shape=[4])
                concat_bchw = gs.Node(op="Concat", name=name + "concat_bchw", attrs={"axis": 0},
                                      inputs=[slice_bc_out, slice_hw_out], outputs=[concat_bchw_out])

                none_var = gs.Variable.empty()

                resize_bchw = gs.Node(op="Resize", name=name + "resize_bchw", attrs=node.attrs,
                                      inputs=[node.inputs[0], none_var, none_var, concat_bchw_out],
                                      outputs=[node.outputs[0]])

                self.graph.nodes.extend([shape_hw, slice_hw, shape_bc, slice_bc, concat_bchw, resize_bchw])

                node.inputs = []
                node.outputs = []

                mResizeNodes += 1

        self.cleanup()
        return mResizeNodes

    def adjustAddNode(self):
        nAdjustAddNode = 0
        for node in self.graph.nodes:
            # Change the bias const to the second input to allow Gemm+BiasAdd fusion in TRT.
            if node.op in ["Add"] and isinstance(node.inputs[0], gs.ir.tensor.Constant):
                tensor = node.inputs[1]
                bias = node.inputs[0]
                node.inputs = [tensor, bias]
                nAdjustAddNode += 1

        self.cleanup()
        return nAdjustAddNode

    def decompose_instancenorms(self):
        nRemoveInstanceNorm = 0
        for node in self.graph.nodes:
            if node.op == "InstanceNormalization":
                name = node.name + "/"
                input_tensor = node.inputs[0]
                output_tensor = node.outputs[0]
                mean_out = gs.Variable(name=name + "mean_out")
                mean_node = gs.Node(op="ReduceMean", name=name + "mean_node", attrs={"axes": [-1]},
                                    inputs=[input_tensor], outputs=[mean_out])
                sub_out = gs.Variable(name=name + "sub_out")
                sub_node = gs.Node(op="Sub", name=name + "sub_node", attrs={}, inputs=[input_tensor, mean_out],
                                   outputs=[sub_out])
                pow_out = gs.Variable(name=name + "pow_out")
                pow_const = gs.Constant(name=name + "pow_const", values=np.array([2.0], dtype=np.float32))
                pow_node = gs.Node(op="Pow", name=name + "pow_node", attrs={}, inputs=[sub_out, pow_const],
                                   outputs=[pow_out])
                mean2_out = gs.Variable(name=name + "mean2_out")
                mean2_node = gs.Node(op="ReduceMean", name=name + "mean2_node", attrs={"axes": [-1]}, inputs=[pow_out],
                                     outputs=[mean2_out])
                epsilon_out = gs.Variable(name=name + "epsilon_out")
                epsilon_const = gs.Constant(name=name + "epsilon_const",
                                            values=np.array([node.attrs["epsilon"]], dtype=np.float32))
                epsilon_node = gs.Node(op="Add", name=name + "epsilon_node", attrs={},
                                       inputs=[mean2_out, epsilon_const], outputs=[epsilon_out])
                sqrt_out = gs.Variable(name=name + "sqrt_out")
                sqrt_node = gs.Node(op="Sqrt", name=name + "sqrt_node", attrs={}, inputs=[epsilon_out],
                                    outputs=[sqrt_out])
                div_out = gs.Variable(name=name + "div_out")
                div_node = gs.Node(op="Div", name=name + "div_node", attrs={}, inputs=[sub_out, sqrt_out],
                                   outputs=[div_out])
                constantScale = gs.Constant("InstanceNormScaleV-" + str(nRemoveInstanceNorm), np.ascontiguousarray(
                    node.inputs[1].inputs[0].attrs["value"].values.reshape(1, 32, 1)))
                constantBias = gs.Constant("InstanceBiasV-" + str(nRemoveInstanceNorm), np.ascontiguousarray(
                    node.inputs[2].inputs[0].attrs["value"].values.reshape(1, 32, 1)))
                mul_out = gs.Variable(name=name + "mul_out")
                mul_node = gs.Node(op="Mul", name=name + "mul_node", attrs={}, inputs=[div_out, constantScale],
                                   outputs=[mul_out])
                add_node = gs.Node(op="Add", name=name + "add_node", attrs={}, inputs=[mul_out, constantBias],
                                   outputs=[output_tensor])
                self.graph.nodes.extend(
                    [mean_node, sub_node, pow_node, mean2_node, epsilon_node, sqrt_node, div_node, mul_node, add_node])
                node.inputs = []
                node.outputs = []
                nRemoveInstanceNorm += 1

        self.cleanup()
        return nRemoveInstanceNorm

    def insert_groupnorm_plugin(self):
        nGroupNormPlugin = 0
        for node in self.graph.nodes:
            if node.op == "Reshape" and node.outputs != [] and \
                node.o().op == "ReduceMean" and node.o(1).op == "Sub" and node.o().o() == node.o(1) and \
                node.o().o().o().o().o().o().o().o().o().o().o().op == "Mul" and \
                node.o().o().o().o().o().o().o().o().o().o().o().o().op == "Add" and \
                len(node.o().o().o().o().o().o().o().o().inputs[1].values.shape) == 3:
                # "node.outputs != []" is added for VAE

                inputTensor = node.i().inputs[0]

                gammaNode = node.o().o().o().o().o().o().o().o().o().o().o()
                index = [type(i) == gs.ir.tensor.Constant for i in gammaNode.inputs].index(True)
                gamma = np.array(deepcopy(gammaNode.inputs[index].values.tolist()), dtype=np.float32)
                constantGamma = gs.Constant("groupNormGamma-" + str(nGroupNormPlugin), np.ascontiguousarray(
                    gamma.reshape(
                        -1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

                betaNode = gammaNode.o()
                index = [type(i) == gs.ir.tensor.Constant for i in betaNode.inputs].index(True)
                beta = np.array(deepcopy(betaNode.inputs[index].values.tolist()), dtype=np.float32)
                constantBeta = gs.Constant("groupNormBeta-" + str(nGroupNormPlugin),
                                           np.ascontiguousarray(beta.reshape(-1)))

                epsilon = node.o().o().o().o().o().inputs[1].values.tolist()[0]

                if betaNode.o().op == "Sigmoid":  # need Swish
                    bSwish = True
                    lastNode = betaNode.o().o()  # Mul node of Swish
                else:
                    bSwish = False
                    lastNode = betaNode  # Cast node after Group Norm

                if lastNode.o().op == "Cast":
                    lastNode = lastNode.o()
                inputList = [inputTensor, constantGamma, constantBeta]
                groupNormV = gs.Variable("GroupNormV-" + str(nGroupNormPlugin), np.dtype(np.float16), inputTensor.shape)
                groupNormN = gs.Node("GroupNorm", "GroupNormN-" + str(nGroupNormPlugin), inputs=inputList,
                                     outputs=[groupNormV],
                                     attrs=OrderedDict([('epsilon', epsilon), ('bSwish', int(bSwish))]))
                self.graph.nodes.append(groupNormN)

                for subNode in self.graph.nodes:
                    if lastNode.outputs[0] in subNode.inputs:
                        index = subNode.inputs.index(lastNode.outputs[0])
                        subNode.inputs[index] = groupNormV
                node.i().inputs = []
                lastNode.outputs = []
                nGroupNormPlugin += 1

        self.cleanup()
        return nGroupNormPlugin

    def insert_layernorm_plugin(self):
        nLayerNormPlugin = 0
        for node in self.graph.nodes:
            if node.op == 'ReduceMean' and \
                node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
                node.o().o(0).op == 'Pow' and node.o().o(1).op == 'Div' and \
                node.o().o(0).o().op == 'ReduceMean' and \
                node.o().o(0).o().o().op == 'Add' and \
                node.o().o(0).o().o().o().op == 'Sqrt' and \
                node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1) and \
                node.o().o(0).o().o().o().o().o().op == 'Mul' and \
                node.o().o(0).o().o().o().o().o().o().op == 'Add' and \
                len(node.o().o(0).o().o().o().o().o().inputs[1].values.shape) == 1:

                if node.i().op == "Add":
                    inputTensor = node.inputs[0]  # CLIP
                else:
                    inputTensor = node.i().inputs[0]  # UNet and VAE

                # The first axis to normalize from can be inferred from the size of the `axes`
                # parameter of (any of) the `ReduceMean` node(s)
                reduceMeanNode = node.o().o(0).o()
                assert reduceMeanNode.op == "ReduceMean"
                firstNormAxis = -1 * np.size(np.array(reduceMeanNode.attrs["axes"]))

                gammaNode = node.o().o().o().o().o().o().o()
                index = [type(i) == gs.ir.tensor.Constant for i in gammaNode.inputs].index(True)
                gamma = np.array(deepcopy(gammaNode.inputs[index].values.tolist()), dtype=np.float32)
                constantGamma = gs.Constant("LayerNormGamma-" + str(nLayerNormPlugin), np.ascontiguousarray(
                    gamma.reshape(
                        -1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

                betaNode = gammaNode.o()
                index = [type(i) == gs.ir.tensor.Constant for i in betaNode.inputs].index(True)
                beta = np.array(deepcopy(betaNode.inputs[index].values.tolist()), dtype=np.float32)
                constantBeta = gs.Constant("LayerNormBeta-" + str(nLayerNormPlugin),
                                           np.ascontiguousarray(beta.reshape(-1)))

                inputList = [inputTensor, constantGamma, constantBeta]
                layerNormV = gs.Variable("LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), inputTensor.shape)
                layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=inputList,
                                     attrs=OrderedDict([('epsilon', 1.e-5), ('axis', firstNormAxis)]),
                                     outputs=[layerNormV])
                self.graph.nodes.append(layerNormN)
                nLayerNormPlugin += 1

                if betaNode.outputs[0] in self.graph.outputs:
                    index = self.graph.outputs.index(betaNode.outputs[0])
                    self.graph.outputs[index] = layerNormV
                else:
                    if betaNode.o().op == "Cast":
                        lastNode = betaNode.o()
                    else:
                        lastNode = betaNode
                    for subNode in self.graph.nodes:
                        if lastNode.outputs[0] in subNode.inputs:
                            index = subNode.inputs.index(lastNode.outputs[0])
                            subNode.inputs[index] = layerNormV
                    lastNode.outputs = []

        self.cleanup()
        return nLayerNormPlugin

    def insert_splitgelu_plugin(self):
        nSplitGeLUPlugin = 0
        for node in self.graph.nodes:
            if node.op == "Erf":
                inputTensor = node.i().i().i().outputs[0]
                lastNode = node.o().o().o().o()
                outputShape = inputTensor.shape
                outputShape[2] = outputShape[2] // 2

                splitGeLUV = gs.Variable("splitGeLUV-" + str(nSplitGeLUPlugin), np.dtype(np.float32), outputShape)
                splitGeLUN = gs.Node("SplitGeLU", "splitGeLUN-" + str(nSplitGeLUPlugin), inputs=[inputTensor],
                                     outputs=[splitGeLUV])
                self.graph.nodes.append(splitGeLUN)

                for subNode in self.graph.nodes:
                    if lastNode.outputs[0] in subNode.inputs:
                        index = subNode.inputs.index(lastNode.outputs[0])
                        subNode.inputs[index] = splitGeLUV
                lastNode.outputs = []
                nSplitGeLUPlugin += 1

        self.cleanup()
        return nSplitGeLUPlugin

    def insert_seq2spatial_plugin(self):
        nSeqLen2SpatialPlugin = 0
        for node in self.graph.nodes:
            if node.op == "Transpose" and node.o().op == "Conv":
                transposeNode = node
                reshapeNode = node.i()
                assert reshapeNode.op == "Reshape", "Unexpected node type for reshapeNode {}".format(reshapeNode.name)
                residualNode = reshapeNode.i(0)
                assert residualNode.op == "Add", "Unexpected node type for residualNode {}".format(residualNode.name)
                biasNode = residualNode.i(0)
                assert biasNode.op == "Add", "Unexpected node type for biasNode {}".format(biasNode.name)
                biasIndex = [type(i) == gs.ir.tensor.Constant for i in biasNode.inputs].index(True)
                bias = np.array(deepcopy(biasNode.inputs[biasIndex].values.tolist()), dtype=np.float32)
                biasInput = gs.Constant("AddAddSeqLen2SpatialBias-" + str(nSeqLen2SpatialPlugin),
                                        np.ascontiguousarray(bias.reshape(-1)))
                inputIndex = 1 - biasIndex
                inputTensor = biasNode.inputs[inputIndex]
                residualInput = residualNode.inputs[1]
                outputTensor = transposeNode.outputs[0]
                outputShapeTensor = transposeNode.i().i().i(1).i(1).i(1).i().inputs[0]
                seqLen2SpatialNode = gs.Node("SeqLen2Spatial", "AddAddSeqLen2Spatial-" + str(nSeqLen2SpatialPlugin),
                                             inputs=[inputTensor, biasInput, residualInput, outputShapeTensor],
                                             outputs=[outputTensor])
                self.graph.nodes.append(seqLen2SpatialNode)
                biasNode.inputs.clear()
                transposeNode.outputs.clear()
                nSeqLen2SpatialPlugin += 1

        self.cleanup()
        return nSeqLen2SpatialPlugin

    def fuse_kv(self, node_k, node_v, fused_kv_idx, heads, num_dynamic=0):
        # Get weights of K
        weights_k = node_k.inputs[1].values
        # Get weights of V
        weights_v = node_v.inputs[1].values
        # Input number of channels to K and V
        C = weights_k.shape[0]
        # Number of heads
        H = heads
        # Dimension per head
        D = weights_k.shape[1] // H

        # Concat and interleave weights such that the output of fused KV GEMM has [b, s_kv, h, 2, d] shape
        weights_kv = np.dstack([weights_k.reshape(C, H, D), weights_v.reshape(C, H, D)]).reshape(C, 2 * H * D)

        # K and V have the same input
        input_tensor = node_k.inputs[0]
        # K and V must have the same output which we feed into fmha plugin
        output_tensor_k = node_k.outputs[0]
        # Fix shape of the QKV output tensor for correct visualization
        output_tensor_k.shape[-1] = 2 * D * H
        # Create tensor
        constant_weights_kv = gs.Constant("Weights_KV_{}".format(fused_kv_idx), np.ascontiguousarray(weights_kv))

        # Create fused KV node
        fused_kv_node = gs.Node(op="MatMul", name="MatMul_KV_{}".format(fused_kv_idx),
                                inputs=[input_tensor, constant_weights_kv], outputs=[output_tensor_k])
        self.graph.nodes.append(fused_kv_node)

        # Connect the output of fused node to the inputs of the nodes after K and V
        node_v.o(num_dynamic).inputs[0] = output_tensor_k
        node_k.o(num_dynamic).inputs[0] = output_tensor_k
        for _ in range(0, num_dynamic):
            node_v.o().inputs.clear()
            node_k.o().inputs.clear()

        # Clear inputs and outputs of K and V to ge these nodes cleared
        node_k.outputs.clear()
        node_v.outputs.clear()
        node_k.inputs.clear()
        node_v.inputs.clear()

        self.cleanup()
        return fused_kv_node

    def fuse_qkv(self, node_q, node_k, node_v, fused_qkv_idx, heads, num_dynamic=0):
        # Get weights of Q
        weights_q = node_q.inputs[1].values
        # Get weights of K
        weights_k = node_k.inputs[1].values
        # Get weights of V
        weights_v = node_v.inputs[1].values

        # Input number of channels to Q, K and V
        C = weights_k.shape[0]
        # Number of heads
        H = heads
        # Hidden dimension per head
        D = weights_k.shape[1] // H

        # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
        weights_qkv = np.dstack(
            [weights_q.reshape(C, H, D), weights_k.reshape(C, H, D), weights_v.reshape(C, H, D)]).reshape(C, 3 * H * D)

        input_tensor = node_k.inputs[0]  # K and V have the same input
        # Q, K and V must have the same output which we feed into fmha plugin
        output_tensor_k = node_k.outputs[0]
        # Fix shape of the QKV output tensor for correct visualization
        output_tensor_k.shape[-1] = 3 * D * H
        # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
        constant_weights_qkv = gs.Constant("Weights_QKV_{}".format(fused_qkv_idx), np.ascontiguousarray(weights_qkv))

        # Created a fused node
        fused_qkv_node = gs.Node(op="MatMul", name="MatMul_QKV_{}".format(fused_qkv_idx),
                                 inputs=[input_tensor, constant_weights_qkv], outputs=[output_tensor_k])
        self.graph.nodes.append(fused_qkv_node)

        # Connect the output of the fused node to the inputs of the nodes after Q, K and V
        node_q.o(num_dynamic).inputs[0] = output_tensor_k
        node_k.o(num_dynamic).inputs[0] = output_tensor_k
        node_v.o(num_dynamic).inputs[0] = output_tensor_k
        for i in range(0, num_dynamic):
            node_q.o().inputs.clear()
            node_k.o().inputs.clear()
            node_v.o().inputs.clear()

        # Clear inputs and outputs of Q, K and V to ge these nodes cleared
        node_q.outputs.clear()
        node_k.outputs.clear()
        node_v.outputs.clear()

        node_q.inputs.clear()
        node_k.inputs.clear()
        node_v.inputs.clear()

        self.cleanup()
        return fused_qkv_node

    def insert_fmhca(self, node_q, node_kv, final_node, mhca_idx, heads, num_dynamic=0):
        # Get inputs and outputs for the fMHCA plugin
        # We take an output of reshape that follows the Q GEMM
        output_q = node_q.o().inputs[0]
        output_kv = node_kv.o().inputs[0]
        output_final_node = final_node.outputs[0]

        # Clear the inputs of the nodes that follow the Q and KV GEMM
        # to delete these subgraphs (it will be substituted by fMHCA plugin)
        node_kv.outputs[0].outputs[0].inputs.clear()
        node_kv.outputs[0].outputs[0].inputs.clear()
        node_q.outputs[0].outputs[0].inputs.clear()
        node_q.outputs[0].outputs[0].inputs.clear()
        # node_q.o().o().inputs.clear()
        # if dynamic_batch:
        #     node_q.o(0, 1).inputs.clear()

        weights_kv = node_kv.inputs[1].values
        dims_per_head = weights_kv.shape[1] // (heads * 2)

        # Reshape dims
        shape_q = gs.Constant("Shape_Q_{}".format(mhca_idx),
                              np.ascontiguousarray(np.array([0, 0, heads, dims_per_head], dtype=np.int64)))
        # Reshape output tensor
        output_q_reshape = gs.Variable("Reshape_Q_{}".format(mhca_idx), np.dtype(np.float16), None)
        # Create Reshape op
        reshape_q = gs.Node(op="Reshape", name="Reshape_fMHCA_Q_{}".format(mhca_idx), inputs=[output_q, shape_q],
                            outputs=[output_q_reshape])
        # Insert node
        self.graph.nodes.append(reshape_q)

        # Reshape dims
        shape_kv = gs.Constant("Shape_KV_{}".format(mhca_idx),
                               np.ascontiguousarray(np.array([0, 0, heads, 2, dims_per_head], dtype=np.int64)))
        # Reshape output tensor
        output_kv_reshape = gs.Variable("ReshapeKV_{}".format(mhca_idx), np.dtype(np.float16), None)
        # Create Reshape op
        reshape_kv = gs.Node(op="Reshape", name="Reshape_fMHCA_{}".format(mhca_idx), inputs=[output_kv, shape_kv],
                             outputs=[output_kv_reshape])
        # Insert node
        self.graph.nodes.append(reshape_kv)

        # fMHCA output tensor
        output_fmhca = gs.Variable("Output_fMHCA_{}".format(mhca_idx), np.dtype(np.float16), None)
        # Create fMHCA plugin
        fmhca = gs.Node(op="fMHCA", name="fMHCA_{}".format(mhca_idx), inputs=[output_q_reshape, output_kv_reshape],
                        outputs=[output_fmhca])
        # Insert node
        self.graph.nodes.append(fmhca)

        # # Connect input of Q reshape to output of Q GEMM
        # node_q.o().outputs[0] = output_q

        if num_dynamic > 0:
            shape_out = gs.Variable("Reshape2_fMHCA_{}_out".format(mhca_idx), np.dtype(np.int64), None)
            reshape2_input1_shape = gs.Node("Shape", "Reshape2_fMHCA_{}_shape".format(mhca_idx),
                                            inputs=[node_q.inputs[0]], outputs=[shape_out])
            self.graph.nodes.append(reshape2_input1_shape)
        else:
            # Reshape dims
            shape_out = gs.Constant("Shape_out_KV_{}".format(mhca_idx),
                                    np.ascontiguousarray(np.array([0, 0, heads * dims_per_head], dtype=np.int64)))

        # Create Reshape op to reshape fMHCA output to [2B, S, H * D]
        reshape_fmhca = gs.Node(op="Reshape", name="Reshape2_fMHA_{}".format(mhca_idx),
                                inputs=[output_fmhca, shape_out], outputs=[output_final_node])
        # Insert node
        self.graph.nodes.append(reshape_fmhca)

        # Clear outputs of transpose to get this subgraph cleared
        final_node.outputs.clear()

        self.cleanup()

    def insert_fmha(self, node_qkv, final_node, mha_idx, heads, num_dynamic=0):
        # Get inputs and outputs for the fMHA plugin
        output_qkv = node_qkv.o().inputs[0]
        output_final_node = final_node.outputs[0]

        # Clear the inputs of the nodes that follow the QKV GEMM
        # to delete these subgraphs (it will be substituted by fMHA plugin)
        node_qkv.outputs[0].outputs[2].inputs.clear()
        node_qkv.outputs[0].outputs[1].inputs.clear()
        node_qkv.outputs[0].outputs[0].inputs.clear()

        weights_qkv = node_qkv.inputs[1].values
        dims_per_head = weights_qkv.shape[1] // (heads * 3)

        # Reshape dims
        shape = gs.Constant("Shape_QKV_{}".format(mha_idx),
                            np.ascontiguousarray(np.array([0, 0, heads, 3, dims_per_head], dtype=np.int64)))

        # Reshape output tensor
        output_shape = gs.Variable("ReshapeQKV_{}".format(mha_idx), np.dtype(np.float16), None)
        # Create Reshape op to reshape QKV output to [2B, S, H, 3, D]
        reshape = gs.Node(op="Reshape", name="Reshape_{}".format(mha_idx), inputs=[output_qkv, shape],
                          outputs=[output_shape])
        # Insert node
        self.graph.nodes.append(reshape)

        # fMHA output tensor
        output_fmha = gs.Variable("Output_fMHA_{}".format(mha_idx), np.dtype(np.float16), None)
        # Create fMHA plugin
        fmha = gs.Node(op="fMHA_V2", name="fMHA_{}".format(mha_idx), inputs=[output_shape], outputs=[output_fmha])
        # Insert node
        self.graph.nodes.append(fmha)

        if num_dynamic > 0:
            shape_out = gs.Variable("Reshape2_fMHA_{}_out".format(mha_idx), np.dtype(np.int64), None)
            reshape2_input1_shape = gs.Node("Shape", "Reshape2_fMHA_{}_shape".format(mha_idx),
                                            inputs=[node_qkv.inputs[0]], outputs=[shape_out])
            self.graph.nodes.append(reshape2_input1_shape)
        else:
            # Reshape dims
            shape_out = gs.Constant("Shape_out_QKV_{}".format(mha_idx),
                                    np.ascontiguousarray(np.array([0, 0, heads * dims_per_head], dtype=np.int64)))

        # Create Reshape op to reshape fMHA output to [2B, S, H * D]
        reshape_fmha = gs.Node(op="Reshape", name="Reshape_fMHA_{}".format(mha_idx), inputs=[output_fmha, shape_out],
                               outputs=[output_final_node])
        # Insert node
        self.graph.nodes.append(reshape_fmha)

        # Clear outputs of transpose to get this subgraph cleared
        final_node.outputs.clear()

        self.cleanup()

    def mha_mhca_detected(self, node, mha):
        # Go from V GEMM down to the S*V MatMul and all way up to K GEMM
        # If we are looking for MHCA inputs of two matmuls (K and V) must be equal.
        # If we are looking for MHA inputs (K and V) must be not equal.
        default_output = False, 0, 0, 0, None, None, None, None
        if node.op == "MatMul" and len(node.outputs) == 1 and \
            ((mha and len(node.inputs[0].inputs) > 0 and node.i().op in ("Cast", "Add")) or \
             (not mha and len(node.inputs[0].inputs) == 0)):

            allowed_ops_inside_attention = ["Shape", "Gather", "Transpose",
                                            "Reshape", "Concat", "Unsqueeze",
                                            "Mul", "Cast", "ConstantOfShape", "Div"]

            iter_node = node.o()
            # Iterate while we don't hit MatMul Softmax(Q @ K') @ V
            while iter_node.op != "MatMul":
                # If we see some layer which is not allowed to be in attention, return
                if iter_node.op not in allowed_ops_inside_attention:
                    return default_output
                iter_node = iter_node.o()

            softmax_gemm_v = iter_node

            if iter_node.i().op != "Softmax":
                if iter_node.i().op == "Cast" and iter_node.i().i().op == "Softmax":
                    iter_node = iter_node.i()
                elif iter_node.i().i().op == "Cast" and iter_node.i().i().i().op == "Softmax":
                    iter_node = iter_node.i().i()
                else:
                    return default_output

            if iter_node.i().i().op != "Add":
                return default_output

            if iter_node.i().i().i().op != "Mul":
                return default_output

            if iter_node.i().i().i().i().op != "MatMul":
                return default_output

            iter_node_q = iter_node.i().i().i().i().i()
            # Iterate while we don't hit Q GEMM
            while iter_node_q.op != "MatMul":
                # If we see some layer which is not allowed to be in attention, return
                if iter_node_q.op not in allowed_ops_inside_attention:
                    return default_output
                iter_node_q = iter_node_q.i()

            iter_node_k = iter_node.i().i().i().i().i(1, 0)
            # Iterate while we don't hit K GEMM
            while iter_node_k.op != "MatMul":
                # If we see some layer which is not allowed to be in attention, return
                if iter_node_k.op not in allowed_ops_inside_attention:
                    return default_output
                iter_node_k = iter_node_k.i()

            if node.o().op == 'Shape':
                if node.o(1).op == 'Shape':
                    num_dynamic_kv = 3 if node.o(2).op == 'Shape' else 2
                else:
                    num_dynamic_kv = 1
                # For Cross-Attention, if batch axis is dynamic (in QKV), assume H*W (in Q) is dynamic as well
                num_dynamic_q = num_dynamic_kv if mha else num_dynamic_kv + 1
            else:
                num_dynamic_kv = 0
                num_dynamic_q = 0

            node_q = iter_node_q
            node_k = iter_node_k
            node_v = node

            iter_node = softmax_gemm_v.o()
            prev_node = None
            # Iterate while we don't hit Reshape followed by FC GEMM after attention
            while iter_node.op != "MatMul":
                # If we see some layer which is not allowed to be in attention, return
                if iter_node.op not in allowed_ops_inside_attention:
                    return default_output
                prev_node = iter_node
                iter_node = iter_node.o()

            if prev_node is None or prev_node.op != "Reshape":
                return default_output

            final_node = prev_node

            # Get num of heads from concated reshape
            heads = final_node.i().i().i(1, 0).inputs[1].values[0].item()
            return True, num_dynamic_q, num_dynamic_kv, heads, node_q, node_k, node_v, final_node
        return default_output

    def fuse_kv_insert_fmhca(self, mhca_index, sm):
        nodes = self.graph.nodes
        # Iterate over graph and search for MHCA pattern
        for idx, _ in enumerate(nodes):
            # fMHCA can't be at the 2 last layers of the network. It is a guard from OOB
            if idx + 1 > len(nodes) or idx + 2 > len(nodes):
                continue

            # Get anchor nodes for fusion and fMHCA plugin insertion if the MHCA is detected
            detected, num_dynamic_q, num_dynamic_kv, heads, node_q, node_k, node_v, final_node = \
                self.mha_mhca_detected(nodes[idx], mha=False)
            if detected:
                assert num_dynamic_q == 0 or num_dynamic_q == num_dynamic_kv + 1
                # Skip the FMHCA plugin for SM75 except for when the dim per head is 40.
                if sm == 75 and node_q.inputs[1].shape[1] // heads == 160:
                    continue
                # Fuse K and V GEMMS
                node_kv = self.fuse_kv(node_k, node_v, mhca_index, heads, num_dynamic_kv)
                # Insert fMHCA plugin
                self.insert_fmhca(node_q, node_kv, final_node, mhca_index, heads, num_dynamic_q)
                return True
        return False

    def fuse_qkv_insert_fmha(self, mha_index):
        nodes = self.graph.nodes
        # Iterate over graph and search for MHA pattern
        for idx, _ in enumerate(nodes):
            # fMHA can't be at the 2 last layers of the network. It is a guard from OOB
            if idx + 1 > len(nodes) or idx + 2 > len(nodes):
                continue

            # Get anchor nodes for fusion and fMHA plugin insertion if the MHA is detected
            detected, num_dynamic_q, num_dynamic_kv, heads, node_q, node_k, node_v, final_node = \
                self.mha_mhca_detected(nodes[idx], mha=True)
            if detected:
                assert num_dynamic_q == num_dynamic_kv
                # Fuse Q, K and V GEMMS
                node_qkv = self.fuse_qkv(node_q, node_k, node_v, mha_index, heads, num_dynamic_kv)
                # Insert fMHA plugin
                self.insert_fmha(node_qkv, final_node, mha_index, heads, num_dynamic_kv)
                return True
        return False

    def insert_fmhca_plugin(self, sm):
        mhca_index = 0
        while self.fuse_kv_insert_fmhca(mhca_index, sm):
            mhca_index += 1
        return mhca_index

    def insert_fmha_plugin(self):
        mha_index = 0
        while self.fuse_qkv_insert_fmha(mha_index):
            mha_index += 1
        return mha_index


def get_path(version, inpaint=False):
    local_cache_dir = './HF_cache'
    if version == "1.4":
        if inpaint:
            if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-inpainting')):
                return os.path.join(local_cache_dir, 'stable-diffusion-inpainting')
            else:
                return "runwayml/stable-diffusion-inpainting"
        else:
            if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-v1-4')):
                return os.path.join(local_cache_dir, 'stable-diffusion-v1-4')
            else:
                return "CompVis/stable-diffusion-v1-4"
    elif version == "1.5":
        if inpaint:
            if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-inpainting')):
                return os.path.join(local_cache_dir, 'stable-diffusion-inpainting')
            else:
                return "runwayml/stable-diffusion-inpainting"
        else:
            if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-v1-5')):
                return os.path.join(local_cache_dir, 'stable-diffusion-v1-5')
            else:
                return "runwayml/stable-diffusion-v1-5"
    elif version == "2.0-base":
        if inpaint:
            if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-2-inpainting')):
                return os.path.join(local_cache_dir, 'stable-diffusion-2-inpainting')
            else:
                return "stabilityai/stable-diffusion-2-inpainting"
        else:
            if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-2-base')):
                return os.path.join(local_cache_dir, 'stable-diffusion-2-base')
            else:
                return "stabilityai/stable-diffusion-2-base"
    elif version == "2.0":
        if inpaint:
            if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-2-inpainting')):
                return os.path.join(local_cache_dir, 'stable-diffusion-2-inpainting')
            else:
                return "stabilityai/stable-diffusion-2-inpainting"
        else:
            if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-2')):
                return os.path.join(local_cache_dir, 'stable-diffusion-2')
            else:
                return "stabilityai/stable-diffusion-2"
    elif version == "2.1":
        if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-2-1')):
            return os.path.join(local_cache_dir, 'stable-diffusion-2-1')
        else:
            return "stabilityai/stable-diffusion-2-1"
    elif version == "2.1-base":
        if os.path.exists(os.path.join(local_cache_dir, 'stable-diffusion-2-1-base')):
            return os.path.join(local_cache_dir, 'stable-diffusion-2-1-base')
        else:
            return "stabilityai/stable-diffusion-2-1-base"
    else:
        raise ValueError(f"Incorrect version {version}")


def get_embedding_dim(version):
    if version in ("1.4", "1.5"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    else:
        raise ValueError(f"Incorrect version {version}")


class BaseModel():
    def __init__(
        self,
        hf_token,
        text_maxlen=77,
        embedding_dim=768,
        fp16=False,
        device='cuda',
        verbose=True,
        max_batch_size=16,
        path="",
    ):
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose
        self.hf_token = hf_token
        self.path = path

        # Defaults
        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim
        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph, minimal_optimization=False):
        return onnx_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height,
        max_latent_height, min_latent_width, max_latent_width)


class CLIP(BaseModel):
    def get_model(self):
        return CLIPTextModel.from_pretrained(self.path,
                                             subfolder="text_encoder",
                                             use_auth_token=self.hf_token).to(self.device)

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
        return ['text_embeddings', 'pooler_output']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width,
                                                                            static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph, minimal_optimization=False):
        enable_optimization = not minimal_optimization

        # Remove Cast Node to optimize Attention block
        bRemoveCastNode = enable_optimization
        # Insert LayerNormalization Plugin
        bLayerNormPlugin = enable_optimization

        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info('CLIP: original')
        opt.select_outputs([0])  # delete graph output#1
        opt.cleanup()
        opt.info('CLIP: remove output[1]')
        opt.fold_constants()
        opt.info('CLIP: fold constants')
        opt.infer_shapes()
        opt.info('CLIP: shape inference')

        if bRemoveCastNode:
            num_casts_removed = opt.remove_casts()
            opt.info('CLIP: removed ' + str(num_casts_removed) + ' casts')

        if bLayerNormPlugin:
            num_layernorm_inserted = opt.insert_layernorm_plugin()
            opt.info('CLIP: inserted ' + str(num_layernorm_inserted) + ' LayerNorm plugins')

        opt.select_outputs([0], names=['text_embeddings'])  # rename network output
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info('CLIP: final')
        return opt_onnx_graph


def make_CLIP(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return CLIP(hf_token=hf_token, embedding_dim=get_embedding_dim(version),
                path=get_path(version, inpaint=inpaint), device=device, verbose=verbose, max_batch_size=max_batch_size)


class UNet(BaseModel):
    def __init__(self,
                 hf_token,
                 text_maxlen=77,
                 embedding_dim=768,
                 fp16=False,
                 device='cuda',
                 verbose=True,
                 max_batch_size=16,
                 unet_dim=4,
                 path="",
                 lora_path=None,
                 fuse_lora=True):
        super(UNet, self).__init__(hf_token, text_maxlen, embedding_dim, fp16, device, verbose, max_batch_size, path)
        self.unet_dim = unet_dim
        self.lora_path = lora_path
        self.fuse_lora = fuse_lora
        self.lora_scale = 1.0

    def get_model(self):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        unet = UNet2DConditionModel.from_pretrained(self.path,
                                                    subfolder="unet",
                                                    use_auth_token=self.hf_token,
                                                    **model_opts).to(self.device)
        if self.lora_path is not None:
            print("[UNet] loading lora weights")
            unet.load_attn_procs(self.lora_path)

            # grab all lora weights
            if self.fuse_lora:
                lora_dict = {}
                sd = unet.state_dict()
                for key in sd.keys():
                    if "lora" in key:
                        assert "transformer_blocks" in key
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-4]), ".".join(key.split(".")[-4:])
                        if attn_processor_key not in lora_dict:
                            lora_dict[attn_processor_key] = []
                        lora_dict[attn_processor_key].append(sub_key)
                # grab original weights that apply lora.
                for key in sd.keys():
                    # only replace mha lora
                    if "transformer_blocks" in key and key.endswith("weight"):
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                        if attn_processor_key in lora_dict:
                            lora_dict[attn_processor_key].append(sub_key)
                        else:
                            attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                            if attn_processor_key in lora_dict:
                                lora_dict[attn_processor_key].append(sub_key)
                assert len(lora_dict) > 0

                # merge weights
                for key in lora_dict.keys():
                    # we have 3 weights (original, down and up) x 4 tensors (q, k, v and out) in each mha
                    assert len(lora_dict[key]) == 12
                    for subkey in lora_dict[key]:
                        if not "lora" in subkey:
                            original_name = "{}.{}".format(key, subkey)
                            lora_down_name = "{}.processor.{}_lora.down.weight".format(key, subkey.split(".")[0])
                            lora_up_name = "{}.processor.{}_lora.up.weight".format(key, subkey.split(".")[0])

                            original_tensor = sd[original_name]
                            lora_down_tensor = sd[lora_down_name]
                            lora_up_tensor = sd[lora_up_name]
                            new_tensor = original_tensor + self.lora_scale * torch.matmul(lora_up_tensor, lora_down_tensor)
                            sd[original_name] = new_tensor
                            sd.pop(lora_down_name)
                            sd.pop(lora_up_name)

                # load unet without lora and update weights
                unet = UNet2DConditionModel.from_pretrained(self.path,
                                                            subfolder="unet",
                                                            use_auth_token=self.hf_token,
                                                            **model_opts).to(self.device)
                unet.load_state_dict(sd)

        return unet

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states']

    def get_output_names(self):
        return ['latent']

    def get_dynamic_axes(self):
        return {
            'sample': {0: '3B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '3B'},
            'latent': {0: '3B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(3 * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                       (3 * batch_size, self.unet_dim, latent_height, latent_width),
                       (3 * max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(3 * min_batch, self.text_maxlen, self.embedding_dim),
                                      (3 * batch_size, self.text_maxlen, self.embedding_dim),
                                      (3 * max_batch, self.text_maxlen, self.embedding_dim)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (3 * batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (3 * batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (3 * batch_size, 4, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(3 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32,
                        device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(3 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device)
        )

    def optimize(self, onnx_graph, minimal_optimization=False):
        enable_optimization = not minimal_optimization

        # Decompose InstanceNormalization into primitive Ops
        bRemoveInstanceNorm = enable_optimization
        # Remove Cast Node to optimize Attention block
        bRemoveCastNode = enable_optimization
        # Remove parallel Swish ops
        bRemoveParallelSwish = enable_optimization
        # Adjust the bias to be the second input to the Add ops
        bAdjustAddNode = enable_optimization
        # Change Resize node to take size instead of scale
        bResizeFix = enable_optimization

        # Common override for disabling all plugins below
        bDisablePlugins = minimal_optimization
        # Use multi-head attention Plugin
        bMHAPlugin = False  # True
        # Use multi-head cross attention Plugin
        bMHCAPlugin = True
        # Insert GroupNormalization Plugin
        bGroupNormPlugin = True
        # Insert LayerNormalization Plugin
        bLayerNormPlugin = True
        # Insert Split+GeLU Plugin
        bSplitGeLUPlugin = True
        # Replace BiasAdd+ResidualAdd+SeqLen2Spatial with plugin
        bSeqLen2SpatialPlugin = True

        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info('UNet: original')

        if bRemoveInstanceNorm:
            num_instancenorm_replaced = opt.decompose_instancenorms()
            opt.info('UNet: replaced ' + str(num_instancenorm_replaced) + ' InstanceNorms')

        if bRemoveCastNode:
            num_casts_removed = opt.remove_casts()
            opt.info('UNet: removed ' + str(num_casts_removed) + ' casts')

        if bRemoveParallelSwish:
            num_parallel_swish_removed = opt.remove_parallel_swish()
            opt.info('UNet: removed ' + str(num_parallel_swish_removed) + ' parallel swish ops')

        if bAdjustAddNode:
            num_adjust_add = opt.adjustAddNode()
            opt.info('UNet: adjusted ' + str(num_adjust_add) + ' adds')

        if bResizeFix:
            num_resize_fix = opt.resize_fix()
            opt.info('UNet: fixed ' + str(num_resize_fix) + ' resizes')

        opt.cleanup()
        opt.info('UNet: cleanup')
        opt.fold_constants()
        opt.info('UNet: fold constants')
        opt.infer_shapes()
        opt.info('UNet: shape inference')

        if bMHAPlugin and not bDisablePlugins:
            num_fmha_inserted = opt.insert_fmha_plugin()
            opt.info('UNet: inserted ' + str(num_fmha_inserted) + ' fMHA plugins')

        if bMHCAPlugin and not bDisablePlugins:
            props = cudart.cudaGetDeviceProperties(0)[1]
            sm = props.major * 10 + props.minor
            num_fmhca_inserted = opt.insert_fmhca_plugin(sm)
            opt.info('UNet: inserted ' + str(num_fmhca_inserted) + ' fMHCA plugins')

        if bGroupNormPlugin and not bDisablePlugins:
            num_groupnorm_inserted = opt.insert_groupnorm_plugin()
            opt.info('UNet: inserted ' + str(num_groupnorm_inserted) + ' GroupNorm plugins')

        if bLayerNormPlugin and not bDisablePlugins:
            num_layernorm_inserted = opt.insert_layernorm_plugin()
            opt.info('UNet: inserted ' + str(num_layernorm_inserted) + ' LayerNorm plugins')

        if bSplitGeLUPlugin and not bDisablePlugins:
            num_splitgelu_inserted = opt.insert_splitgelu_plugin()
            opt.info('UNet: inserted ' + str(num_splitgelu_inserted) + ' SplitGeLU plugins')

        if bSeqLen2SpatialPlugin and not bDisablePlugins:
            num_seq2spatial_inserted = opt.insert_seq2spatial_plugin()
            opt.info('UNet: inserted ' + str(num_seq2spatial_inserted) + ' SeqLen2Spatial plugins')

        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info('UNet: final')
        return onnx_opt_graph


def make_UNet(version, hf_token, device, verbose, max_batch_size, text_maxlen=77, inpaint=False, lora_path=None):
    return UNet(hf_token=hf_token, text_maxlen=text_maxlen, embedding_dim=get_embedding_dim(version), fp16=True, unet_dim=(9 if inpaint else 4),
                device=device, verbose=verbose, path=get_path(version, inpaint=inpaint), max_batch_size=max_batch_size,
                lora_path=lora_path)


class TorchVAEDecoder(torch.nn.Module):
    def __init__(self, token, device, path):
        super().__init__()
        self.path = path
        self.vae_decoder = AutoencoderKL.from_pretrained(self.path, subfolder="vae", use_auth_token=token).to(device)

    def forward(self, x):
        return self.vae_decoder.decode(x).sample


class VAE(BaseModel):
    def get_model(self):
        print(f"[VAE] loading from path: {self.path}")
        vae = TorchVAEDecoder(self.hf_token, self.device, self.path)
        return vae

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
        return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width),
                       (batch_size, 4, latent_height, latent_width),
                       (max_batch, 4, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)

    def optimize(self, onnx_graph, minimal_optimization=False):
        enable_optimization = not minimal_optimization

        # Decompose InstanceNormalization into primitive Ops
        bRemoveInstanceNorm = enable_optimization
        # Remove Cast Node to optimize Attention block
        bRemoveCastNode = enable_optimization
        # Insert GroupNormalization Plugin
        bGroupNormPlugin = enable_optimization

        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info('VAE: original')

        if bRemoveInstanceNorm:
            num_instancenorm_replaced = opt.decompose_instancenorms()
            opt.info('VAE: replaced ' + str(num_instancenorm_replaced) + ' InstanceNorms')

        if bRemoveCastNode:
            num_casts_removed = opt.remove_casts()
            opt.info('VAE: removed ' + str(num_casts_removed) + ' casts')

        opt.cleanup()
        opt.info('VAE: cleanup')
        opt.fold_constants()
        opt.info('VAE: fold constants')
        opt.infer_shapes()
        opt.info('VAE: shape inference')

        if bGroupNormPlugin:
            num_groupnorm_inserted = opt.insert_groupnorm_plugin()
            opt.info('VAE: inserted ' + str(num_groupnorm_inserted) + ' GroupNorm plugins')

        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info('VAE: final')
        return onnx_opt_graph


def make_VAE(version, hf_token, device, verbose, max_batch_size, inpaint=False, override_path=None):
    return VAE(hf_token=hf_token, embedding_dim=get_embedding_dim(version),
               device=device, verbose=verbose, path=get_path(version, inpaint=inpaint), max_batch_size=max_batch_size)


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, token, device, path):
        super().__init__()
        self.path = path
        self.vae_encoder = AutoencoderKL.from_pretrained(self.path, subfolder="vae", use_auth_token=token).to(device)

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()


class VAEEncoder(BaseModel):
    def get_model(self):
        vae_encoder = TorchVAEEncoder(self.hf_token, self.device, self.path)
        return vae_encoder

    def get_input_names(self):
        return ['images']

    def get_output_names(self):
        return ['latent']

    def get_dynamic_axes(self):
        return {
            'images': {0: 'B', 2: '8H', 3: '8W'},
            'latent': {0: 'B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, _, _, _, _ = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            'images': [(min_batch, 3, min_image_height, min_image_width), (batch_size, 3, image_height, image_width),
                       (max_batch, 3, max_image_height, max_image_width)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, 4, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)

    def optimize(self, onnx_graph, minimal_optimization=False):

        enable_optimization = not minimal_optimization

        # Decompose InstanceNormalization into primitive Ops
        bRemoveInstanceNorm = enable_optimization
        # Remove Cast Node to optimize Attention block
        bRemoveCastNode = enable_optimization
        # Insert Cast before Clip is a WAR for Windows
        bAddCastBeforeClip = enable_optimization
        # Insert GroupNormalization Plugin
        bGroupNormPlugin = enable_optimization

        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info('VAE Encoder: original')

        if bRemoveInstanceNorm:
            num_instancenorm_replaced = opt.decompose_instancenorms()
            opt.info('VAE Encoder: replaced ' + str(num_instancenorm_replaced) + ' InstanceNorms')

        if bRemoveCastNode:
            num_casts_removed = opt.remove_casts()
            opt.info('VAE Encoder: removed ' + str(num_casts_removed) + ' casts')

        if bAddCastBeforeClip:
            num_casts_inserted = opt.clip_cast_fix()
            opt.info('VAE Encoder: inserted ' + str(num_casts_inserted) + ' casts')

        opt.cleanup()
        opt.info('VAE Encoder: cleanup')
        opt.fold_constants()
        opt.info('VAE Encoder: fold constants')
        opt.infer_shapes()
        opt.info('VAE Encoder: shape inference')

        if bGroupNormPlugin:
            num_groupnorm_inserted = opt.insert_groupnorm_plugin()
            opt.info('VAE Encoder: inserted ' + str(num_groupnorm_inserted) + ' GroupNorm plugins')

        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info('VAE Encoder: final')
        return onnx_opt_graph


def make_VAEEncoder(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return VAEEncoder(hf_token=hf_token, embedding_dim=get_embedding_dim(version),
                      device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
                      max_batch_size=max_batch_size)


def make_tokenizer(version, hf_token):
    return CLIPTokenizer.from_pretrained(get_path(version),
                                         subfolder="tokenizer",
                                         use_auth_token=hf_token)
