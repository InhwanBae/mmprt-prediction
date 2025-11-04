import os
import torch
from model import GroupedGraphAttentionNet


FEATURE_COLS = [
    'Age',
    'BMI',
    'Sx_duration',
    'Pre_mHKA',
    'Pre_MPTA',
    'Pre_LDFA',
    'ICRS',
    'Effusion',
    'BME',
    'SIFK'
]

GROUP_COLS = {
  'patient_demographics': [
    'Age',
    'Sex',
    'Height',
    'Weight',
    'BMI'
  ],
  'clinical_presentation': [
    'Sx_duration',
    'Chronicity',
    'Injury_mechanism',
    'Popping',
    'Givingway'
  ],
  'radiographic_parameters': [
    'Pre_mHKA',
    'Pre_MPTA',
    'Pre_LDFA',
    'Pre_JLCA',
    'Pre_slope',
    'Pre_KL',
    'MJW_extension',
    'MJW_flexion'
  ],
  'mri_findings': [
    'ICRS',
    'Shinycorner',
    'MME_absolute',
    'MME_relative',
    'Lat_PTS_MRI',
    'Med_PTS_MRI',
    'Effusion',
    'BME',
    'SIFK'
  ]
}

INPUT_DIM = len(FEATURE_COLS)
OUTPUT_DIM = 3
HIDDEN_LAYERS = (128, 16)
ATTENTION_METHOD = 'GAT'


def convert_model(model_path='best_model.pt'):
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model file '{model_path}'.")
        print("Please correct the MODEL_PATH variable to point to a valid .pt file.")
        return

    print(f"Loading model from '{model_path}'...")

    model = GroupedGraphAttentionNet(
        input_dim=INPUT_DIM,
        hidden_layer_sizes=HIDDEN_LAYERS,
        output_dim=OUTPUT_DIM,
        attention_method=ATTENTION_METHOD,
        feature_cols=FEATURE_COLS,
        group_cols=GROUP_COLS
    )
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    onnx_path = model_path.replace('.pt', '.onnx')    
    dummy_input = torch.randn(1, INPUT_DIM)
    print(f"Converting the model to ONNX format at '{onnx_path}'...")
    try:
        torch.onnx.export(
            model,                   # model being run
            dummy_input,             # dummy input
            onnx_path,               # output file name
            input_names=['input'],   # input tensor name
            output_names=['output'], # output tensor name
            dynamic_axes={           # set batch size to be variable
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11  # version of ONNX opset
        )
        print(f"conversion successful! ONNX model saved at '{onnx_path}'")
        
    except Exception as e:
        print(f"\n--- ONNX conversion failed ---")
        print(f"Error: {e}")
        print("The model's 'forward' function or architecture may not be compatible with ONNX conversion.")


if __name__ == '__main__':
    convert_model(model_path='results/dl_FU2yrs_compact_results_20251029_143535/best_model.pt')
    convert_model(model_path='results/dl_FU5yrs_compact_results_20251029_143005/best_model.pt')
    