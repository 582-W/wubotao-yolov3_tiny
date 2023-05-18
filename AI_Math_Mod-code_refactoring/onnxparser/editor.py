import onnx
from onnx import helper, numpy_helper, shape_inference, utils
from onnx import AttributeProto, TensorProto, GraphProto, ValueInfoProto, ModelProto, NodeProto
from .parser import OnnxParser

class OnnxEditor():
    
    def __init__(self) -> None:
        pass
        
    @classmethod    
    def make_node(cls, op_tpye: str, name: str, inputs: list, outputs: list, attribute: dict):
        """Construct a NodeProto.

        Arguments:
            op_type (string): The name of the operator to construct
            inputs (list of string): list of input names
            outputs (list of string): list of output names
            name (string, default None): optional unique identifier for NodeProto
            attribute (dict): the attributes of the node.  The acceptable values
                are documented in :func:`make_attribute`.
        """
        assert len(outputs) == 1
        return helper.make_node(
            op_type=op_tpye,
            inputs=inputs,
            outputs=outputs,
            name=name,
            **attribute
        )

    @classmethod
    def rm_node(cls,model,node):

        '''this function will automatically refine the structure of the graph when deleting node
        '''
        assert node.op_type != 'Constant', "constant node is weight, should not be removed individually"
        former_node_dict = OnnxParser.find_former_node(model,node)[0]
        next_node_dict = OnnxParser.find_next_node(model,node)[0]
        ##delete weight in initializer and constant node
        for weight_ in model.graph.initializer:
            if weight_.name in  node.input:
                model.graph.initializer.remove(weight_)
        for node_ in model.graph.node:
            if node_.op_type == 'Constant' and node_.output[0] in node.input:
                model.graph.node.remove(node_)
        # delete value info of output
        for value_info_ in model.graph.value_info:
            if value_info_.name in node.output:
                model.graph.value_info.remove(value_info_)

        ####change input of next layer
        new_input = []
        for node_ in model.graph.node:
            if node_.name in former_node_dict[node.name]:
                new_input.append(node_.output[0])

        if len(next_node_dict[node.name]) == 0:  ## last node
            for item in model.graph.output:
                if item.name == node.output[0]:
                    model.graph.output.remove(item)
            for item in new_input:
                for node_ in model.graph.node:
                    if item==node_.output[0] and len(OnnxParser.find_next_node(model,node_)[0][node_.name])==1:
                        model.graph.output.append(helper.make_tensor_value_info(item,TensorProto.FLOAT,shape=[]))

        else: 
            for node_ in model.graph.node:
                if node_.name in next_node_dict[node.name]:
                    node_.input.remove(node.output[0])
                    for item in new_input:
                        node_.input.insert(0,item)

        model.graph.node.remove(node)


    @classmethod
    def add_initializer(cls, model, name, values : list, shape):
        initializer_add = helper.make_tensor(name, TensorProto.FLOAT, shape, values)
        model.graph.initializer.append(initializer_add)

    
    @classmethod
    def rm_initializer(cls, model, name):
        for item in model.graph.initializer:
            if item.name == name:
                model.graph.initializer.remove(item)
                break

        
    @classmethod
    def add_init_to_graph_input(cls, model):
        '''this function should be applied before shape infer'''
        inputs = [i.name for i in model.graph.input]
        for init in model.graph.initializer:
            if init.name in inputs:
                continue
            tensor_shape = init.dims
            model.graph.input.append(helper.make_tensor_value_info(init.name,init.data_type,tensor_shape))

        return model

    @classmethod
    def shape_infer(cls,model):
        '''this function will add shape info of feature map for all layers
            but it will stop inferencing when meeting unsupported node type
            more info in https://github.com/onnx/onnx/blob/master/docs/ShapeInference.md
        '''
        model = OnnxEditor.add_init_to_graph_input(model)
        return shape_inference.infer_shapes(model)


    @classmethod
    def set_node(cls, node, name=None, input=None, output=None, op_type=None, attribute=None):
        if name:
            node.name = name 
        if input:
            node.input = input
        if output:
            node.output = output
        if op_type:
            node.op_type = op_type
        if attribute:
            node.attribute = []
            node.attribute.extend(
                helper.make_attribute(key, value)
                for key, value in sorted(attribute.items()))


    @classmethod
    def cut_submodel(cls,input_path, output_path, input_names, output_names):
        '''Function extract_model() extracts sub-model from an ONNX model. The sub-model is defined by the names of the input and output tensors exactly.
        Arguments:
            input_path (string): The path to original ONNX model.
            output_path (string): The path to save the extracted ONNX model.
            input_names (list of string): The names of the input tensors that to be extracted.
            output_names (list of string): The names of the output tensors that to be extracted.
        '''
        utils.extract_model(input_path, output_path, input_names, output_names)

    @classmethod
    def model_polish(cls,model):
        '''
            This function combines several useful utility functions together.
        '''
        onnx.checker.check_model(model)
        onnx.helper.strip_doc_string(model)
        model = shape_inference.infer_shapes(model)##add shape info of feature map for each layer
        ##model = onnx.optimizer.optimize(model)  ##unsupported in latest onnx version
        onnx.checker.check_model(model)
        return model

    
    @classmethod
    def model_check(cls,model):
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            print('The model is invalid: %s' % e)
        else:
            print('The model is valid!')
