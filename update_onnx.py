# -*- coding:utf-8 -*-
import onnx

def update_onnx(src_model_file, dst_model_file, batch_input=True, 
                new_input_name='img', new_output_name='fea',
                add_node_bgr_2_rgb=True):
    
    model = onnx.load(src_model_file)
    onnx.checker.check_model(model)
        
    print(model.graph.input)
    print(model.graph.output)

    # 假设 onnx 只有一个输入和一个输出。
    
    if (batch_input):
        # 修改 onnx 为动态输入输出。
        dim_proto_i1 = model.graph.input[0].type.tensor_type.shape.dim[0]
        dim_proto_i1.dim_param = 'batchsize'
        dim_proto_o1 = model.graph.output[0].type.tensor_type.shape.dim[0]
        dim_proto_o1.dim_param = 'batchsize'
    
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    model.graph.input[0].name = new_input_name
    model.graph.output[0].name = new_output_name
    # 更新节点的输入输出到新的输入输出名
    for i in range(len(model.graph.node)):
        for j in range(len(model.graph.node[i].input)):
            if model.graph.node[i].input[j] == input_name:
                print('------------------------')
                print(model.graph.node[i].name)
                print(model.graph.node[i].input)
                print(model.graph.node[i].output)
                if (add_node_bgr_2_rgb):
                    model.graph.node[i].input[j] = new_input_name+'_rgb'
                else:
                    model.graph.node[i].input[j] = new_input_name

        for j in range(len(model.graph.node[i].output)):
            if model.graph.node[i].output[j] == output_name:
                print('------------------------')
                print(model.graph.node[i].name)
                print(model.graph.node[i].input)
                print(model.graph.node[i].output)
                model.graph.node[i].output[j] = new_output_name
    
    # 假设原模型的输入是 RGB 格式，添加节点把输入的 BGR 转成 RGB。
    if (add_node_bgr_2_rgb):
        # 添加 split 和 concat，把 BGR 转成 RGB
        new_node = onnx.helper.make_node(
            'Split',
            name=new_input_name+'_split',
            axis=1,
            inputs=[new_input_name],
            outputs=[new_input_name+'_b', new_input_name+'_g', new_input_name+'_r']
        )
        model.graph.node.append(new_node)
        new_node = onnx.helper.make_node(
            'Concat',
            name=new_input_name+'_concat',
            axis=1,
            inputs=[new_input_name+'_r', new_input_name+'_g', new_input_name+'_b'],
            outputs=[new_input_name+'_rgb']
        )
        model.graph.node.append(new_node)

    onnx.save(model, dst_model_file)

update_onnx('./models/model_IResNet_R100_Glint360K.onnx', './models/model_IResNet_R100_Glint360K_new.onnx')
exit(0)