'''
Map Pytorch module names to MLPerf TF2 reference
for weight initialization check.

'''

stage_map = {'layer1' : 'stage1',
             'layer2' : 'stage2',
             'layer3' : 'stage3',
             'layer4' : 'stage4'}

unit_map = {'0': 'unit1',
            '1': 'unit2',
            '2': 'unit3',
            '3': 'unit4',
            '4': 'unit5',
            '5': 'unit6'}

sc_map = {'downsample.0': 'conv1sc',
          'downsample.1': 'bnsc'}


def get_mlperf_name(name): 
    
    chunks = name.split(".")
    mlperf_name = ""
    if 'layer' in name:
        if 'downsample' not in name:
            mlperf_name = stage_map[chunks[3]] + "_" + unit_map[chunks[4]] + "_" + chunks[5] 
        elif 'downsample.0' in name:
            mlperf_name = stage_map[chunks[3]] + "_" + unit_map[chunks[4]] + "_" + sc_map['downsample.0'] 
        elif 'downsample.1' in name:    
            mlperf_name = stage_map[chunks[3]] + "_" + unit_map[chunks[4]] + "_" + sc_map['downsample.1']
    else: 
        if 'conv1' in name:
            mlperf_name = 'conv0'
        elif 'bn1' in name:
            mlperf_name = 'bn0'
        elif 'fc' in name:
            mlperf_name = 'fc1'
    
    return mlperf_name


