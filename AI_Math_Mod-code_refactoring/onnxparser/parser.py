import onnx
import numpy as np


class OnnxParser():
    def __init__(self, model):
        self.model = model
        self.node_list, self.constant_node_list, self.former_layer_dict, self.next_layer_dict = OnnxParser.topo_sort(self.model)
        self.constant_node_output = [node.output[0] for node in self.constant_node_list]
        self.initializer, self.initializer_dict = OnnxParser.get_initializer(self.model)
        self.weight_name_list = list(self.initializer_dict.keys()) + self.constant_node_output
        self.input_dict, self.output_dict, self.model_graph_input, self.model_graph_output = self._get_input_output_dict()
        
    
    
    @classmethod
    def find_former_node(cls,model,node):
        former_node_dict = {}
        former_node_dict[node.name] = []
        former_node_list = []
        for input in node.input:
            for node_ in model.graph.node:
                if input in node_.output:
                    if node_.op_type != 'Constant':
                        former_node_dict[node.name].append(node_.name)
                        former_node_list.append(node_)
                    break
        
        return former_node_dict, former_node_list


    @classmethod
    def find_next_node(cls,model,node):
        next_node_dict = {}
        next_node_dict[node.name] = []
        next_node_list = []
        for output in node.output:
            for node_ in model.graph.node:
                if output in node_.input:
                    next_node_dict[node.name].append(node_.name)
                    next_node_list.append(node_)

        return next_node_dict, next_node_list

    @classmethod
    def topo_sort(cls,model):
        #sort node_list based on the model's info
        all_layer_dict = {} # layer_name -> node_id
        former_layer_dict = {}  # layer_name -> former layer's name , regardless of constant node
        next_layer_dict = {}  # layer_name -> next layer's name , regardless of constant node
        set_constant_node = set() # constant node name set
        for i in range(len(model.graph.node)):
            layer = model.graph.node[i]
            all_layer_dict[layer.name] = i
            if layer.op_type == "Constant":
                set_constant_node.add(layer.name)
                continue
            
            former_node_dict = OnnxParser.find_former_node(model, layer)[0]
            if list(former_node_dict.values())[0]:
                former_layer_dict.update(former_node_dict)
            
            next_node_dict = OnnxParser.find_next_node(model, layer)[0]
            if list(next_node_dict.values())[0]:
                next_layer_dict.update(next_node_dict)


        # find first layer
        first_layer_name_set = set(all_layer_dict.keys()) - set(former_layer_dict.keys()) - set_constant_node
        # assert len(first_layer_name_set) == 1, f"find {len(first_layer_name_set)} inputs"
        first_layer_name = list(first_layer_name_set)

        #sort layer
        layer_order_list = []
        layer_order_list.extend(list(set_constant_node)) ## put constant node in head
        cur_layer_queue = first_layer_name
        while cur_layer_queue:
            next_layer_queue = []
            while len(cur_layer_queue) > 0:
                cur_layer = cur_layer_queue.pop(0)
                if cur_layer in layer_order_list:
                    continue
                whether_traverse_all_input = True
                if cur_layer in former_layer_dict:
                    for input_layer in former_layer_dict[cur_layer]:
                        if input_layer not in layer_order_list:
                            whether_traverse_all_input = False
                            break
                if whether_traverse_all_input:
                    layer_order_list.append(cur_layer)
                    if cur_layer in next_layer_dict:
                        next_layer_queue.extend(next_layer_dict[cur_layer])
            cur_layer_queue = next_layer_queue
        assert len(layer_order_list)==len(model.graph.node), "list do not cover all nodes"
        #layer_name to node
        node_order_list = []
        for layer_name in layer_order_list:
            for j in range(len(model.graph.node)):
                if model.graph.node[j].name == layer_name:
                    node_order_list.append(model.graph.node[j])
                    break
        assert len(node_order_list)==len(model.graph.node), "list do not cover all nodes"
        return node_order_list, node_order_list[:len(set_constant_node)], former_layer_dict, next_layer_dict
    

    @classmethod
    def get_initializer(cls,model):
        initializer = model.graph.initializer
        initializer_dict = {}
        for i in range(len(initializer)):
            name = initializer[i].name
            initializer_dict[name] = i
        return initializer, initializer_dict
    
    
    def get_node_weights(self,node):
        '''
        This function assume that the order of inputs for each type of node are fixed, or errors will occur!
        It aims to obtain the weights of each node. weights are divided into two categories, 
        one is stored in model.graph.initializer,another is stored in constant nodes

        return: weight_dict: input name --> input array
        '''
        weight_dict = {}
        for input_name in node.input:
            if input_name in self.initializer_dict.keys():
                index = self.initializer_dict[input_name]
                weight_dict[input_name] = onnx.numpy_helper.to_array(self.initializer[index])
            elif input_name in self.constant_node_output:
                index = self.constant_node_output.index(input_name)
                weight_dict[input_name] = AttributeExtractor.get_info_constant(self.constant_node_list[index])
            else: #feature map of last layer
                pass
        
        return weight_dict
    

    def _get_input_output_dict(self):
        input_dict, output_dict = {}, {} # key:node.input/ node.output, value: node.name , regardless of constant node
        model_graph_input, model_graph_output = [], [] # model.graph.input / model.graph.output
        for i in range(len(self.node_list)):
            if self.node_list[i].op_type == 'Constant':
                continue
            node_name = self.node_list[i].name
            inputs = self.node_list[i].input
            outputs = self.node_list[i].output
            assert len(outputs) == 1, 'there are multi-outputs in the node' 
            output_dict[outputs[0]] = node_name
            for ele in inputs:
                if not ele in self.weight_name_list:
                    if not ele in input_dict.keys():
                        input_dict[ele] = []
                    input_dict[ele].append(node_name)
        for k in self.model.graph.input:
            # initializer name may be stored in model.graph.input, delete it!
            if k.name not in self.weight_name_list:
                model_graph_input.append(k.name)
        for k in self.model.graph.output:
            model_graph_output.append(k.name)
        
        assert len(model_graph_input) == 1, "fail to find input node or find multi-inputs"
        output_dict[model_graph_input[0]] = 'input'
        return input_dict, output_dict, model_graph_input, model_graph_output




class AttributeExtractor():
    ''' this class aims to process AttributeProto, extract attributes in the node by name
        node.attribute.name return name of the attribution
        node.attribute.f return float number, node.attribute.floats return float list
        node.attribute.i return int number, node.attribute.ints return int list
        node.attribute.s return string object
        node.attribute.t return tensor, use onnx.numpy_helper.to_array() to obtain numpy array, no matter float_data or raw_data
    '''

    def __init__(self):
        pass

    
    def get_info_conv(self,node):
        dilation, group, pad, stride = '', 1, '', ''
        for j in range(len(node.attribute)):
            para_name =  node.attribute[j].name
            if para_name == "dilations":
                dilation = node.attribute[j].ints
            elif para_name == "group":
                group = node.attribute[j].i
            elif para_name =="pads":
                pad = node.attribute[j].ints
            elif para_name =="strides":
                stride = node.attribute[j].ints
        return dilation, group, pad, stride
    
    
    def get_info_bn(self,node):
        [epsilon], [momentum] = [0.00001], [0.9]
        for attr in node.attribute:
            if attr.name == 'epsilon':
                [epsilon] = [attr.f]
            if attr.name == 'momentum':
                [momentum] = [attr.f]
        return epsilon, momentum

    
    def get_info_maxpool(self,node):
        for i in range(len(node.attribute)):
            if node.attribute[i].name == 'kernel_shape':
                kernel_shape = node.attribute[i].ints
            elif node.attribute[i].name == 'pads':
                pad = node.attribute[i].ints
            elif node.attribute[i].name == 'strides':
                stride = node.attribute[i].ints

        return kernel_shape, pad, stride

    
    def get_info_leakyrelu(self,node):
        alpha = 0.01 # default value
        if node.attribute:
            [alpha] = [attr.f for attr in node.attribute if attr.name == 'alpha']
        return alpha
    
    
    def get_info_upsample(self,node):
        mode = 'nearest'
        scales = [] # scales should be put in node's input instead of node's attribute since opset 9
        for i in range(len(node.attribute)):
            if node.attribute[i].name == 'scales':
                scales = node.attribute[i].floats
            elif node.attribute[i].name == 'mode':
                mode = node.attribute[i].s
        return mode, scales
    

    def get_info_cancat(self,node):
        [axis] = [attr.i for attr in node.attribute if attr.name == 'axis']
        return axis

    
    def get_info_clip(self,node):
        ##TODO:max and min has became the input of the node instead of attributes since opset 11
        [x_max] = [attr.f for attr in node.attribute if attr.name == 'max']
        [x_min] = [attr.f for attr in node.attribute if attr.name == 'min']
        return x_max, x_min


    def get_info_averagepool(self,node):
        for i in range(len(node.attribute)):
            if node.attribute[i].name == 'kernel_shape':
                kernel_shape = node.attribute[i].ints
            elif node.attribute[i].name == 'pads':
                pad = node.attribute[i].ints
            elif node.attribute[i].name == 'strides':
                stride = node.attribute[i].ints

        return kernel_shape, pad, stride

    @classmethod
    def get_info_constant(cls,node):
        [attr] = [a for a in node.attribute if a.name == "value"]
        w = onnx.numpy_helper.to_array(attr.t)
        return w
    
    def get_info_gather(self,node):
        [attr] = [a for a in node.attribute if a.name == "axis"]
        axis = attr.i
        return axis
    
    def get_info_unsqueeze(self,node):
        [attr] = [a for a in node.attribute if a.name == "axes"]
        axes = attr.ints
        return axes
    
    def get_info_transpose(self,node):
        [attr] = [a for a in node.attribute if a.name == "perm"]
        perm = attr.ints
        return perm
    
    def get_info_cast(self,node):
        [attr] = [a for a in node.attribute if a.name == "to"]
        to = attr.i
        return to
    
    def get_info_reshape(self,node):
        allowzero = 0
        if node.attribute:
            [attr] = [a for a in node.attribute if a.name == "allowzero"]
            allowzero = attr.i
        return allowzero
    
    def get_info_constantofshape(self,node):
        value = np.array([0])
        if node.attribute:
            [attr] = [a for a in node.attribute if a.name == "value"]
            value = onnx.numpy_helper.to_array(attr.t)
        return value
    
    def get_info_resize(self,node):
        coordinate_transformation_mode = 'half_pixel'
        cubic_coeff_a = 0.75
        exclude_outside = 0
        extrapolation_value = 0.0
        mode = 'nearst'
        nearest_mode = 'round_prefer_floor'

        for i in range(len(node.attribute)):
            if node.attribute[i].name == 'coordinate_transformation_mode':
                coordinate_transformation_mode = node.attribute[i].s
            elif node.attribute[i].name == 'cubic_coeff_a':
                cubic_coeff_a = node.attribute[i].f
            elif node.attribute[i].name == 'exclude_outside':
                exclude_outside = node.attribute[i].i
            elif node.attribute[i].name == 'extrapolation_value':
                extrapolation_value = node.attribute[i].f
            elif node.attribute[i].name == 'mode':
                mode = node.attribute[i].s
            elif node.attribute[i].name == 'nearest_mode':
                nearest_mode = node.attribute[i].s
                
        return coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode
        
